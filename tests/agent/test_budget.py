"""The context budget: when compaction runs and what each step cuts.

BudgetConfig(window=8192) gives usable 6,144, trigger 4,608 and target 3,072.
"""
import json

import pytest
from lingo.llm import Message, ToolCall

from lovelaice.agent.budget import BudgetConfig, ContextBudget
from lovelaice.agent.compaction import CompactionState, check_tool_pairs, render_view
from lovelaice.agent.loops.react_native import ReActNative
from lovelaice.agent.turns import assign_turns

NUDGE = ReActNative.EMPTY_TURN_NUDGE
_ids = iter(range(10_000))


def _batch(chars):
    cid = f"c{next(_ids)}"
    return [Message(role="assistant", content="",
                    tool_calls=[ToolCall(id=cid, name="web_fetch", arguments={"url": cid})]),
            Message.tool("x" * chars, tool_call_id=cid)]


def conv(*turns):
    """Each turn: a list of ints (tool result sizes in chars), 'nudge', or 'answer'.
    Every turn starts with a user message; finished turns end with an answer."""
    history = []
    for t in turns:
        history.append(Message.user("question"))
        for item in t:
            if item == "nudge":
                history.append(Message.user(NUDGE))
            elif item == "answer":
                history.append(Message.assistant("answer"))
            else:
                history.extend(_batch(item))
    items = [(m.role, m.content if isinstance(m.content, str) else "", None) for m in history]
    turns_ = assign_turns(items, NUDGE)
    return history, turns_


def case(history, turns, **kw):
    d = dict(system_prompt="SYS", tools=None, history=history, turns=turns,
             current_turn=turns[-1], state=CompactionState(), summarize=None)
    d.update(kw)
    return d


def dump(view):
    return json.dumps([m.model_dump() for m in view], sort_keys=True)


budget = ContextBudget(BudgetConfig(window=8192))


def test_config_numbers():
    c = BudgetConfig(window=8192)
    assert (c.reserve, c.usable) == (2048, 6144)
    assert BudgetConfig(window=200_000).reserve == 2048
    assert BudgetConfig(window=4096).reserve == 1024
    assert BudgetConfig(window=8192, max_tokens=1000).usable == 7192
    assert BudgetConfig(window=None).usable is None


@pytest.mark.asyncio
async def test_no_window_no_compaction():
    h, t = conv([40_000, "answer"], [40_000])
    c = case(h, t)
    view, rep = await ContextBudget(BudgetConfig(window=None)).prepare(**c)
    assert rep is None
    assert dump(view) == dump(render_view("SYS", h, CompactionState()))


@pytest.mark.asyncio
async def test_under_trigger_no_change():
    h, t = conv([4000, "answer"], [])
    c = case(h, t)
    view, rep = await budget.prepare(**c)
    assert rep is None and c["state"] == CompactionState()


@pytest.mark.asyncio
async def test_step1_stops_at_target():
    h, t = conv([8000, "answer"], [8000, "answer"], [8000, "answer"], [])
    c = case(h, t)
    view, rep = await budget.prepare(**c)
    assert rep.stubbed_finished == 2
    assert rep.after_tokens <= 3072
    assert rep.stubbed_current == 0 and rep.turns_summarized == 0 and rep.clipped == 0
    assert c["state"].stubbed == [2, 6]          # the two oldest results
    check_tool_pairs(view)


@pytest.mark.asyncio
async def test_step1_never_stubs_current_turn_with_nudge():
    h, t = conv([4000, "answer"], [16_000, "nudge", 400])
    c = case(h, t)
    view, rep = await budget.prepare(**c)
    pre_nudge = 6                                  # the 16k result, before the nudge
    assert t[pre_nudge] == c["current_turn"]
    assert rep.stubbed_finished == 1 and rep.stubbed_current == 1
    assert pre_nudge in c["state"].stubbed


@pytest.mark.asyncio
async def test_step2_keeps_latest_batch():
    h, t = conv([5600, 5600, 5600, 5600])
    c = case(h, t)
    view, rep = await budget.prepare(**c)
    last_assistant = max(p for p, m in enumerate(h) if m.role == "assistant")
    latest = [p for p, m in enumerate(h) if p > last_assistant and m.role == "tool"]
    assert not set(latest) & set(c["state"].stubbed)
    assert rep.stubbed_current == 2 and rep.after_tokens <= 3072


@pytest.mark.asyncio
async def test_clip_reaches_into_latest_batch():
    h, t = conv([48_000])
    c = case(h, t)
    view, rep = await budget.prepare(**c)
    assert rep.clipped == 1 and rep.after_tokens <= 3072
    assert "tokens removed to save context" in view[-1].content
    check_tool_pairs(view)


@pytest.mark.asyncio
async def test_fixed_over_window_warns_and_sends():
    h, t = conv([400])
    c = case(h, t, system_prompt="s" * 28_000)
    view, rep = await budget.prepare(**c)
    assert rep.fixed_over and rep.fixed_tokens == 7000
    assert view[0].role == "system" and len(view) == 1 + len(h)


@pytest.mark.asyncio
async def test_state_only_grows():
    h, t = conv([8000, "answer"], [8000, "answer"], [8000, "answer"], [])
    c = case(h, t)
    await budget.prepare(**c)
    first = list(c["state"].stubbed)
    h2, t2 = conv([8000, "answer"], [8000, "answer"], [8000, "answer"], [8000, "answer"], [])
    c2 = case(h2, t2, state=c["state"])
    await budget.prepare(**c2)
    assert set(first) <= set(c2["state"].stubbed)


@pytest.mark.asyncio
async def test_prefix_stable_between_calls():
    h, t = conv([8000, "answer"], [8000, "answer"], [8000, "answer"], [])
    c = case(h, t)
    v1, _ = await budget.prepare(**c)              # compacts
    c["history"] = h + [Message.assistant("short answer"), Message.user("and now?")]
    c["turns"] = t + [t[-1], t[-1] + 1]
    c["current_turn"] = t[-1] + 1
    v2, rep = await budget.prepare(**c)            # under trigger: no new cut
    assert rep is None
    assert dump(v2[:len(v1)]) == dump(v1)          # same prefix, byte for byte


@pytest.mark.asyncio
async def test_forced_target_runs_under_trigger():
    h, t = conv([4000, "answer"], [])
    c = case(h, t)
    view, rep = await budget.prepare(**c, target_tokens=200)
    assert rep is not None and rep.stubbed_finished == 1


@pytest.mark.asyncio
async def test_clip_picks_largest_message_as_rendered():
    # An earlier 48k result was already clipped down; a new 48k result arrives.
    h, t = conv([48_000])
    c = case(h, t)
    await budget.prepare(**c)
    first = next(iter(c["state"].clipped))
    h2 = h + [m for m in conv([48_000])[0][1:]]       # second batch, same turn
    t2 = t + [t[-1]] * (len(h2) - len(h))
    c2 = case(h2, t2, state=c["state"])
    view, rep = await budget.prepare(**c2)
    assert rep.clipped == 1
    assert (len(h2) - 1) in c2["state"].clipped       # the new result, not `first`
    assert rep.after_tokens <= 6144                   # one clip per run (spec §06)
