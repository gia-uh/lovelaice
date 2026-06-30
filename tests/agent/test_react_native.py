import pytest
from unittest.mock import AsyncMock
from lingo.llm import Message, ToolCall
from lingo.tools import tool as lingo_tool
from lovelaice.agent.tools import AgentTool, ToolRegistry
from lovelaice.agent.hooks import HookRegistry
from lovelaice.agent.harness import Harness
from lovelaice.agent.session import Session
from lovelaice.agent.loops.react_native import ReActNative
from lovelaice.agent.errors import StopReason


@lingo_tool
async def echo(text: str) -> str:
    """Echo."""
    return text


def _make_harness(mock_llm, tools=None):
    reg = ToolRegistry()
    for t in tools or []:
        reg.register(t)
    return Harness(llm=mock_llm, tools=reg, hooks=HookRegistry(),
                   system_prompt="SYS")


@pytest.mark.asyncio
async def test_react_native_one_tool_then_text(tmp_path):
    """Canary #2 — mocked LLM emits one tool call, then plain text.
    Harness dispatches the tool, second LLM call ends the turn."""
    responses = iter([
        Message(role="assistant", content="", tool_calls=[
            ToolCall(id="c1", name="echo", arguments={"text": "hello"})
        ], stop_reason="tool_calls"),
        Message(role="assistant", content="done.", stop_reason="stop"),
    ])
    mock_llm = AsyncMock()
    mock_llm.chat = AsyncMock(side_effect=lambda *a, **kw: next(responses))

    harness = _make_harness(mock_llm, tools=[AgentTool(inner=echo)])
    sess = Session.create(tmp_path / "s.jsonl", model="x",
                          system_prompt_hash="h", loop="ReActNative",
                          cwd=str(tmp_path))

    loop = ReActNative()
    stop = await loop.run(harness, sess, Message.user("please"))

    assert stop == StopReason.END_TURN
    # Session should have: user, assistant(tool_calls), tool, assistant(stop)
    msgs = sess.messages_for_llm("SYS")
    roles = [m.role for m in msgs]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]


@pytest.mark.asyncio
async def test_react_native_no_tools_returns_immediately(tmp_path):
    """LLM with no tool calls → one turn → end_turn."""
    mock_llm = AsyncMock()
    mock_llm.chat = AsyncMock(return_value=Message.assistant("hello", stop_reason="stop"))

    harness = _make_harness(mock_llm)
    sess = Session.create(tmp_path / "s.jsonl", model="x",
                          system_prompt_hash="h", loop="ReActNative",
                          cwd=str(tmp_path))

    stop = await ReActNative().run(harness, sess, Message.user("hi"))
    assert stop == StopReason.END_TURN

    msgs = sess.messages_for_llm("SYS")
    assert [m.role for m in msgs] == ["system", "user", "assistant"]


@pytest.mark.asyncio
async def test_react_native_emits_turn_start_and_end_events(tmp_path):
    mock_llm = AsyncMock()
    mock_llm.chat = AsyncMock(return_value=Message.assistant("hi", stop_reason="stop"))
    harness = _make_harness(mock_llm)
    events = []
    harness.subscribe(lambda ev: events.append(ev))
    sess = Session.create(tmp_path / "s.jsonl", model="x",
                          system_prompt_hash="h", loop="ReActNative",
                          cwd=str(tmp_path))
    await ReActNative().run(harness, sess, Message.user("hi"))
    types = [type(e).__name__ for e in events]
    assert "TurnStart" in types
    assert "TurnEnd" in types
    assert "AssistantMessageFinalized" in types


@pytest.mark.asyncio
async def test_react_native_aborts_if_signal_set(tmp_path):
    """If harness.abort is set before/during the loop, returns CANCELLED."""
    import asyncio
    mock_llm = AsyncMock()
    mock_llm.chat = AsyncMock(return_value=Message.assistant("hi", stop_reason="stop"))
    harness = _make_harness(mock_llm)
    harness.abort.set()  # set BEFORE the loop starts
    sess = Session.create(tmp_path / "s.jsonl", model="x",
                          system_prompt_hash="h", loop="ReActNative",
                          cwd=str(tmp_path))
    stop = await ReActNative().run(harness, sess, Message.user("hi"))
    assert stop == StopReason.CANCELLED


@pytest.mark.asyncio
async def test_react_native_second_call_message_format(tmp_path):
    """After a tool call, the second LLM call must receive content:null (not "")
    for the assistant message that emitted the tool call.

    Strict providers (Qwen via OpenRouter) reject content:"" on tool-calling
    turns. This test captures the actual messages passed to the second call and
    verifies the wire format. Regression for the 'agent cuts after first tool
    call with small models' bug."""
    captured: list[list] = []

    async def fake_chat(messages, **kw):
        captured.append(messages)
        if len(captured) == 1:
            return Message(role="assistant", content="", tool_calls=[
                ToolCall(id="c1", name="echo", arguments={"text": "x"})
            ], stop_reason="tool_calls")
        return Message.assistant("done.", stop_reason="stop")

    mock_llm = AsyncMock()
    mock_llm.chat = fake_chat

    harness = _make_harness(mock_llm, tools=[AgentTool(inner=echo)])
    sess = Session.create(tmp_path / "s.jsonl", model="x",
                          system_prompt_hash="h", loop="ReActNative",
                          cwd=str(tmp_path))

    stop = await ReActNative().run(harness, sess, Message.user("run it"))
    assert stop == StopReason.END_TURN
    assert len(captured) == 2, "expected exactly two LLM calls"

    # The second call must include the assistant+tool_calls message.
    second_messages = captured[1]
    assistant_msgs = [m for m in second_messages if m.role == "assistant"]
    assert assistant_msgs, "second call must include assistant message"
    # Verify wire format: content must be null, not ""
    wire = assistant_msgs[0].model_dump()
    assert wire.get("content") is None, (
        f"assistant message with tool_calls must have content:null in dump, "
        f"got {wire.get('content')!r} — strict providers reject content:\"\""
    )


@pytest.mark.asyncio
async def test_react_native_maps_stop_reason_length_to_max_tokens(tmp_path):
    mock_llm = AsyncMock()
    mock_llm.chat = AsyncMock(return_value=Message.assistant("hi", stop_reason="length"))
    harness = _make_harness(mock_llm)
    sess = Session.create(tmp_path / "s.jsonl", model="x",
                          system_prompt_hash="h", loop="ReActNative",
                          cwd=str(tmp_path))
    stop = await ReActNative().run(harness, sess, Message.user("hi"))
    assert stop == StopReason.MAX_TOKENS


# --- empty-turn continuation: don't end the turn until there's a real answer ---


@pytest.mark.asyncio
async def test_react_native_empty_turn_nudges_then_real_answer(tmp_path):
    """A thinking model may return empty content + no tool_calls (it reasoned
    but produced no answer). The loop must NOT treat that as done — it nudges
    and continues until a genuine final answer arrives."""
    responses = iter([
        Message(role="assistant", content="", stop_reason="stop"),
        Message(role="assistant", content="You have 1 vault.", stop_reason="stop"),
    ])
    mock_llm = AsyncMock()
    mock_llm.chat = AsyncMock(side_effect=lambda *a, **kw: next(responses))
    harness = _make_harness(mock_llm)
    sess = Session.create(tmp_path / "s.jsonl", model="x",
                          system_prompt_hash="h", loop="ReActNative",
                          cwd=str(tmp_path))

    stop = await ReActNative().run(harness, sess, Message.user("how many vaults?"))

    assert stop == StopReason.END_TURN
    assert mock_llm.chat.await_count == 2, "loop must continue past the empty turn"
    msgs = sess.messages_for_llm("SYS")
    roles = [m.role for m in msgs]
    assert roles.count("user") == 2, "a nudge user message should be injected"
    assert roles[-1] == "assistant"
    assert "1 vault" in (msgs[-1].content or "")


@pytest.mark.asyncio
async def test_react_native_empty_turn_gives_up_after_three_nudges(tmp_path):
    """If the model never produces content, the loop must not spin forever —
    it gives up after at most 3 nudge continuations (4 LLM calls total)."""
    mock_llm = AsyncMock()
    mock_llm.chat = AsyncMock(
        side_effect=lambda *a, **kw: Message(role="assistant", content="", stop_reason="stop"))
    harness = _make_harness(mock_llm)
    sess = Session.create(tmp_path / "s.jsonl", model="x",
                          system_prompt_hash="h", loop="ReActNative",
                          cwd=str(tmp_path))

    stop = await ReActNative().run(harness, sess, Message.user("hi"))

    assert stop == StopReason.END_TURN
    assert mock_llm.chat.await_count == 4, \
        "initial empty turn + 3 nudge retries, then give up"
