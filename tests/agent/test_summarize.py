"""Step 3: old turns folded into one rolling summary."""
import pytest
from lingo.llm import Message, Usage

from lovelaice.agent.budget import BudgetConfig, ContextBudget
from lovelaice.agent.compaction import CompactionState, check_tool_pairs, summary_header
from lovelaice.agent.events import AssistantMessageDelta
from lovelaice.agent.harness import Harness
from lovelaice.agent.hooks import HookRegistry
from lovelaice.agent.summarize import build_summary_request, render_transcript
from lovelaice.agent.tools import ToolRegistry

from tests.agent.test_budget import case, conv


class FakeSummarizer:
    def __init__(self):
        self.calls = 0
        self.args = []

    async def __call__(self, previous, messages, cap):
        self.calls += 1
        self.args.append((previous, messages, cap))
        return f"SUMMARY-{self.calls}", Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)


def first_position_of_turn(turns, n):
    return turns.index(n)


budget = ContextBudget(BudgetConfig(window=8192))
# Six finished turns whose user/assistant text alone is over target, so
# stubbing tool results (step 1) is not enough.
LONG = "y" * 3000


def long_conv(n_finished):
    h, t = conv(*([[400, "answer"]] * n_finished), [])
    for m in h:
        if m.role == "user":
            m.content = LONG
    return h, t


@pytest.mark.asyncio
async def test_summary_folds_all_but_last_two_finished_turns():
    h, t = long_conv(6)
    fake = FakeSummarizer()
    c = case(h, t, summarize=fake)
    view, rep = await budget.prepare(**c)
    assert rep.turns_summarized == 4
    assert c["state"].summary.covers_through == first_position_of_turn(t, 5)
    assert c["state"].summary.turns_covered == 4
    assert view[1].content.startswith(summary_header(4) + "SUMMARY-1")
    assert rep.summary == "SUMMARY-1" and rep.summary_usage.total_tokens == 15
    check_tool_pairs(view)


@pytest.mark.asyncio
async def test_summary_skipped_when_steps_1_2_reach_target():
    h, t = conv([8000, "answer"], [8000, "answer"], [8000, "answer"], [])
    fake = FakeSummarizer()
    await budget.prepare(**case(h, t, summarize=fake))
    assert fake.calls == 0


@pytest.mark.asyncio
async def test_summary_is_rolling():
    fake = FakeSummarizer()
    h, t = long_conv(6)
    c = case(h, t, summarize=fake)
    await budget.prepare(**c)
    old_cut = c["state"].summary.covers_through
    h2, t2 = long_conv(9)
    c2 = case(h2, t2, state=c["state"], summarize=fake)
    await budget.prepare(**c2)
    previous, msgs, cap = fake.args[-1]
    assert previous == "SUMMARY-1"
    assert c2["state"].summary.runs == 2 and c2["state"].summary.text == "SUMMARY-2"
    assert c2["state"].summary.covers_through > old_cut
    # only the newly covered turns are sent, not the ones already summarized
    assert len(msgs) == c2["state"].summary.covers_through - old_cut


@pytest.mark.asyncio
async def test_summary_cap():
    fake = FakeSummarizer()
    h, t = long_conv(6)
    await budget.prepare(**case(h, t, summarize=fake))
    assert fake.args[-1][2] == 614                      # usable 6144 // 10

    fake2 = FakeSummarizer()
    big = ContextBudget(BudgetConfig(window=200_000))
    h, t = long_conv(6)
    await big.prepare(**case(h, t, summarize=fake2), target_tokens=100)
    assert fake2.args[-1][2] == 1024


@pytest.mark.asyncio
async def test_summary_text_clipped_to_cap():
    async def verbose(previous, messages, cap):
        return "z" * 100_000, None
    h, t = long_conv(6)
    c = case(h, t, summarize=verbose)
    await budget.prepare(**c)
    assert len(c["state"].summary.text) <= 614 * 4


def test_build_summary_request_shape():
    msgs = [Message.user("hola"), Message.assistant("¿qué tal?")]
    req = build_summary_request("PREV", msgs, 300)
    assert [m.role for m in req] == ["system", "user"]
    assert "300 tokens" in req[0].content
    assert "Previous summary:\nPREV" in req[1].content
    assert "user: hola" in req[1].content and "assistant: ¿qué tal?" in req[1].content


class StreamingLLM:
    model = "fake"

    def __init__(self):
        self._on_token = None
        self.last_tools = "unset"
        self.last_kwargs = None

    async def chat(self, messages, tools=None, **kwargs):
        self.last_tools, self.last_kwargs = tools, kwargs
        if self._on_token:
            self._on_token("leak")
        return Message.assistant("the summary")


@pytest.mark.asyncio
async def test_summary_call_streams_nothing():
    llm = StreamingLLM()
    harness = Harness(llm=llm, tools=ToolRegistry(), hooks=HookRegistry(), system_prompt="S")
    events = []
    harness.subscribe(events.append)
    text, usage = await harness.summarizer()(None, [Message.user("a"), Message.assistant("b")], 100)
    assert text == "the summary"
    assert not [e for e in events if isinstance(e, AssistantMessageDelta)]
    assert llm.last_tools is None
    # A thinking model spends its token budget reasoning first. Capping the call
    # at the summary size left Qwen 3.5 9B with an empty answer (live run,
    # 2026-10-17); the session's own max_tokens applies and the text is clipped.
    assert "max_tokens" not in llm.last_kwargs
    assert llm._on_token is not None          # streaming restored for the real answer


@pytest.mark.asyncio
async def test_empty_summary_folds_nothing():
    async def empty(previous, messages, cap):
        return "   ", None
    h, t = long_conv(6)
    c = case(h, t, summarize=empty)
    view, rep = await budget.prepare(**c)
    assert c["state"].summary is None
    assert rep.turns_summarized == 0
