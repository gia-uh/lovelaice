"""Regression checks for the configurable policy and its real model-call path."""
import math

import pytest
from lingo.llm import Message, ToolCall
from pydantic import ValidationError

from lovelaice.agent import Agent, AgentConfig, CompactionPolicy, CompactionStep
from lovelaice.agent.budget import BudgetConfig, ContextBudget
from lovelaice.agent.compaction import check_tool_pairs, fixed_chars, view_chars
from lovelaice.agent.errors import StopReason
from lovelaice.agent.events import ContextMeasured, ContextOverflow
from lovelaice.agent.loops.react_native import ReActNative
from lovelaice.agent.summarize import build_summary_request, summary_input_cost

from tests.agent.test_budget import case, conv
from tests.agent.test_model_call import ScriptedLLM, context_error, make_agent, seed
from tests.agent.test_summarize import FakeSummarizer


@pytest.mark.parametrize("bad", [
    {"trigger_percent": 0}, {"target_percent": 75}, {"trigger_percent": 101},
    {"target_percent": 50.5}, {"trigger_percent": "75"}, {"protected_turns": -1},
    {"steps": ["invalid"]}, {"steps": ["read_results", "read_results"]}, {"typo": 1},
])
def test_policy_rejects_invalid_configuration(bad):
    with pytest.raises(ValidationError):
        CompactionPolicy.model_validate(bad)


def test_policy_json_and_legacy_fractional_thresholds(tmp_path):
    policy = CompactionPolicy.model_validate_json(
        '{"trigger_percent":80,"target_percent":40,"protected_turns":0,"steps":[]}')
    assert policy.model_dump(mode="json") == {
        "trigger_percent": 80, "target_percent": 40, "protected_turns": 0, "steps": [],
    }
    agent = Agent(config=AgentConfig(model="m", api_key="fake", context_window=8192,
                                     compaction_trigger=0.725, compaction_target=0.625),
                  tools=[], loop=ReActNative(), session_path=tmp_path / "legacy.jsonl")
    config = agent.harness.budget.config
    assert (config.trigger_fraction, config.target_fraction) == (0.725, 0.625)


def test_policy_threshold_and_target_use_conversation_space():
    config = BudgetConfig(window=8192, policy=CompactionPolicy())
    assert config.usable == 6144
    assert not config.should_compact(5108, 2000, config.usable)
    assert config.should_compact(5109, 2000, config.usable)
    assert config.target_for(2000, config.usable) == 4072


async def test_policy_order_changes_what_is_removed():
    history, turns = conv([5250, "answer"], ["answer"], ["answer"], [])
    fixed = "s" * 18_000
    default = await ContextBudget(BudgetConfig(window=8192)).prepare(
        **case(history, turns, system_prompt=fixed, summarize=FakeSummarizer()))
    reverse = CompactionPolicy(steps=(CompactionStep.SUMMARIZE_OLD_TURNS,
                                      CompactionStep.FINISHED_RESULTS))
    swapped = await ContextBudget(BudgetConfig(window=8192, policy=reverse)).prepare(
        **case(history, turns, system_prompt=fixed, summarize=FakeSummarizer()))
    assert default[1].operations == {CompactionStep.FINISHED_RESULTS: 1}
    assert swapped[1].operations == {CompactionStep.SUMMARIZE_OLD_TURNS: 1}
    assert default[0] != swapped[0]


async def test_full_batch_replacement_keeps_dialogue_and_valid_remaining_calls():
    history, turns = conv([3000, 6000, "answer"], [3000])
    history[1].content = "I will look that up."
    original = [message.model_dump() for message in history]
    policy = CompactionPolicy(steps=(CompactionStep.FINISHED_TOOL_BATCHES,))
    view, report = await ContextBudget(BudgetConfig(window=4096, policy=policy)).prepare(
        **case(history, turns), target_tokens=1200)
    check_tool_pairs(view)
    assert report.operations == {CompactionStep.FINISHED_TOOL_BATCHES: 2}
    assert len([message for message in view if message.role == "tool"]) == 1
    assert "I will look that up." in view[2].content and "answer" in view[2].content
    assert "history message 2–3" in view[2].content
    assert view[-2:] == history[-2:]
    assert [message.model_dump() for message in history] == original


async def test_summary_reads_elided_original_and_budgets_each_request():
    history, turns = conv([2800, "answer"], ["answer"], ["answer"], ["answer"], [])
    history[0].content = "qué " * 1400
    history[2].content = "ORIGINAL-7219 " + history[2].content
    history[4].content = "second old turn"
    fake = FakeSummarizer()
    budget = ContextBudget(BudgetConfig(window=8192))
    kwargs = case(history, turns, summarize=fake)
    _, report = await budget.prepare(**kwargs, target_tokens=100)
    assert report.stubbed_finished == 1 and report.turns_summarized == 2
    assert len(fake.args) == 2
    assert "ORIGINAL-7219" in "".join(transcript for _, transcript, _ in fake.args)
    assert any("continued from history message 1" in text for _, text, _ in fake.args)
    for previous, transcript, cap in fake.args:
        assert summary_input_cost(build_summary_request(previous, transcript, cap)) <= 6144


async def test_incomplete_summary_keeps_only_the_last_complete_prefix():
    history, turns = conv(["answer"], ["answer"], ["answer"], ["answer"], [])
    history[0].content = "first " * 200
    history[2].content = "very long " * 4000
    calls = []

    async def fails_partway(previous, transcript, cap):
        calls.append(transcript)
        return ("Short useful summary", None) if len(calls) == 1 else ("", None)

    kwargs = case(history, turns, summarize=fails_partway)
    _, report = await ContextBudget(BudgetConfig(window=4096)).prepare(
        **kwargs, target_tokens=100)
    assert len(calls) == 2 and report.turns_summarized == 1
    assert kwargs["state"].summary.covers_through == 2


class LimitedProvider(ScriptedLLM):
    def __init__(self, limit, script=(), *, input_only=False):
        super().__init__(script)
        self.limit = limit
        self.input_only = input_only
        self.attempts = []

    async def chat(self, messages, tools=None, **kwargs):
        if messages[0].content.startswith("You compress"):
            assert summary_input_cost(messages) + kwargs["max_tokens"] <= self.limit
        else:
            tokens = math.ceil((fixed_chars(messages[0].content, tools) + view_chars(messages)) / 4)
            self.attempts.append((tokens, messages))
            if tokens > self.limit:
                if self.input_only:
                    exc = context_error(tokens, self.limit)
                    exc.args = (f"request ({tokens} tokens) exceeds maximum input tokens: {self.limit}",)
                    raise exc
                raise context_error(tokens, self.limit)
        return await super().chat(messages, tools, **kwargs)


@pytest.mark.parametrize("input_only, usable", [(False, 3072), (True, 4096)])
async def test_real_agent_retries_with_provider_limit(tmp_path, monkeypatch, input_only, usable):
    provider = LimitedProvider(4096, input_only=input_only)
    agent, events = make_agent(tmp_path, monkeypatch, provider, window=131_072)
    seed(agent, [30_000])
    assert await agent.prompt("Please continue.") == StopReason.END_TURN
    assert len(provider.attempts) == 2 and provider.attempts[0][0] > 4096
    assert provider.attempts[1][0] < usable
    assert agent.session.compaction.stubbed == [2]
    if not input_only:
        assert [ev.window for ev in events if isinstance(ev, ContextMeasured)] == [131_072, 4096]


async def test_real_agent_preserves_unread_batch_and_does_not_repeat_it(tmp_path, monkeypatch):
    reply = Message(role="assistant", content="", tool_calls=[
        ToolCall(id=f"call-{i}", name="big", arguments={"n": i}) for i in range(4)])
    provider = LimitedProvider(4096, [reply])
    agent, events = make_agent(tmp_path, monkeypatch, provider, window=4096)
    assert await agent.prompt("Get four results.") == StopReason.REFUSAL
    assert len(provider.attempts) == 2  # initial request, then one with the unread batch
    failed = provider.attempts[-1][1]
    assert [len(m.content) for m in failed if m.role == "tool"] == [8000] * 4
    assert not agent.session.compaction.stubbed and not agent.session.compaction.clipped
    overflow = next(ev for ev in events if isinstance(ev, ContextOverflow))
    assert overflow.window == 4096 and overflow.estimated is False


async def test_failed_attempt_does_not_commit_proactive_cuts(tmp_path, monkeypatch):
    provider = LimitedProvider(100)
    agent, _ = make_agent(tmp_path, monkeypatch, provider, window=8192)
    seed(agent, [25_000, 25_000, 25_000])
    before = agent.session.compaction.model_copy(deep=True)
    assert await agent.prompt("keep this request " * 200) == StopReason.REFUSAL
    assert any("result elided" in m.content for m in provider.attempts[0][1] if m.role == "tool")
    assert agent.session.compaction == before
    assert not list(tmp_path.glob("*.compaction.json"))


async def test_opt_in_clipping_can_retry_when_provider_does_not_name_its_limit(tmp_path, monkeypatch):
    error = context_error()
    error.args = ("maximum context length exceeded",)
    llm = ScriptedLLM([error, Message.assistant("ok", stop_reason="stop")])
    monkeypatch.setattr("lovelaice.agent.agent._build_llm", lambda cfg: llm)
    policy = CompactionPolicy(protected_turns=0, steps=[CompactionStep.TRUNCATE_OLD_TEXT])
    agent = Agent(config=AgentConfig(model="m", system_prompt="SYS", context_window=131_072,
                                     compaction_policy=policy),
                  tools=[], loop=ReActNative(), session_path=tmp_path / "fallback.jsonl")
    agent.session.append(Message.user("old " * 4000), turn=1)
    agent.session.append(Message.assistant("Answer"), turn=1)
    assert await agent.prompt("Do not clip this message.") == StopReason.END_TURN
    assert len(llm.calls) == 2
    assert view_chars(llm.calls[1].messages) < view_chars(llm.calls[0].messages) * 0.76
    assert llm.calls[1].messages[-1].content == "Do not clip this message."
