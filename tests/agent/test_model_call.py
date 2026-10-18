"""The budget wired into the loop: compaction before each call, one retry on overflow."""
import math

import pytest
from lingo.llm import Message, ToolCall, Usage
from lingo.tools import tool as lingo_tool

from lovelaice.agent import Agent, AgentConfig
from lovelaice.agent.compaction import check_tool_pairs, fixed_chars, view_chars
from lovelaice.agent.errors import StopReason
from lovelaice.agent.events import ContextCompacted, ContextOverflow, ContextOverheadWarning
from lovelaice.agent.loops.react_native import ReActNative
from lovelaice.agent.tools import AgentTool


@lingo_tool
async def big(n: int) -> str:
    """Return a large page."""
    return "p" * 8000


class ProviderError(Exception):
    def __init__(self, message, status_code):
        super().__init__(message)
        self.status_code = status_code


def context_error(prompt_tokens=9000, n_ctx=8192):
    return ProviderError(
        f"Error code: 400 - {{'error': {{'code': 400, 'message': 'request ({prompt_tokens} tokens) "
        f"exceeds the available context size ({n_ctx} tokens), try increasing it', "
        f"'type': 'exceed_context_size_error'}}}}", 400)


class Call:
    def __init__(self, messages, tools, kwargs):
        self.messages, self.tools, self.kwargs = messages, tools, kwargs


class ScriptedLLM:
    """Plays a script of replies or exceptions; summary calls answer apart."""
    model = "fake"

    def __init__(self, script):
        self.script = list(script)
        self.calls: list[Call] = []
        self.summary_calls = 0
        self._on_token = None

    async def chat(self, messages, tools=None, **kwargs):
        if messages and messages[0].role == "system" and messages[0].content.startswith(
                "You compress"):
            self.summary_calls += 1
            return Message.assistant("SUMMARY")
        self.calls.append(Call(list(messages), tools, kwargs))
        item = self.script.pop(0) if self.script else Message.assistant("ok", stop_reason="stop")
        if isinstance(item, BaseException):
            raise item
        return item


def tool_turn(i):
    return Message(role="assistant", content="", stop_reason="tool_calls",
                   tool_calls=[ToolCall(id=f"t{i}", name="big", arguments={"n": i})])


def make_agent(tmp_path, monkeypatch, llm, window=8192, system_prompt="SYS"):
    monkeypatch.setattr("lovelaice.agent.agent._build_llm", lambda cfg: llm)
    agent = Agent(config=AgentConfig(model="m", system_prompt=system_prompt,
                                     context_window=window, list_tools_in_prompt=False),
                  tools=[AgentTool(inner=big)], loop=ReActNative(),
                  session_path=tmp_path / "s.jsonl")
    events = []
    agent.subscribe(events.append)
    return agent, events


def seed(agent, turns):
    """Finished turns, each with one tool result of `chars` characters."""
    for i, chars in enumerate(turns, start=1):
        s = agent.session
        s.append(Message.user(f"question {i}"), turn=i)
        s.append(Message(role="assistant", content="", tool_calls=[
            ToolCall(id=f"s{i}", name="big", arguments={"n": i})]), turn=i)
        s.append(Message.tool("q" * chars, tool_call_id=f"s{i}"), turn=i)
        s.append(Message.assistant(f"answer {i}"), turn=i)


def est(agent, call):
    chars = fixed_chars(agent.harness.system_prompt, call.tools) + view_chars(call.messages)
    return math.ceil(chars / 4)


def dump(ms):
    return [m.model_dump() for m in ms]


@pytest.mark.asyncio
async def test_no_window_view_identical_to_today(tmp_path, monkeypatch):
    llm = ScriptedLLM([tool_turn(1), tool_turn(2), Message.assistant("done", stop_reason="stop")])
    agent, _ = make_agent(tmp_path, monkeypatch, llm, window=None)
    seed(agent, [40_000, 40_000])
    await agent.prompt("go")
    for call in llm.calls:
        today = agent.session.messages_for_llm(agent.harness.system_prompt)[:len(call.messages)]
        assert dump(call.messages) == dump(today)


@pytest.mark.asyncio
async def test_overflow_retry_once_then_succeeds(tmp_path, monkeypatch):
    llm = ScriptedLLM([context_error(), Message.assistant("ok", stop_reason="stop")])
    agent, events = make_agent(tmp_path, monkeypatch, llm)
    seed(agent, [6000, 6000])                       # ~3,000 tokens: under the trigger
    assert await agent.prompt("go") == StopReason.END_TURN
    assert len(llm.calls) == 2
    assert est(agent, llm.calls[0]) > 0.40 * 6144
    assert est(agent, llm.calls[1]) <= 0.40 * 6144
    assert [e for e in events if isinstance(e, ContextCompacted)]


@pytest.mark.asyncio
async def test_second_refusal_ends_turn_with_readable_error(tmp_path, monkeypatch):
    llm = ScriptedLLM([context_error(), context_error()])
    agent, events = make_agent(tmp_path, monkeypatch, llm)
    seed(agent, [6000])
    assert await agent.prompt("go") == StopReason.REFUSAL
    ov = [e for e in events if isinstance(e, ContextOverflow)]
    assert len(ov) == 1
    assert ov[0].usable == 6144 and ov[0].fixed_tokens > 0
    assert f"{ov[0].fixed_tokens:,}" in ov[0].message and "6,144" in ov[0].message
    assert len(llm.calls) == 2


@pytest.mark.asyncio
async def test_no_window_refusal_uses_provider_window(tmp_path, monkeypatch):
    llm = ScriptedLLM([context_error(prompt_tokens=9000, n_ctx=4096),
                       Message.assistant("ok", stop_reason="stop")])
    agent, _ = make_agent(tmp_path, monkeypatch, llm, window=None)
    seed(agent, [8000, 8000, 8000, 8000])
    assert await agent.prompt("go") == StopReason.END_TURN
    assert est(agent, llm.calls[1]) <= 0.40 * 4096


@pytest.mark.asyncio
async def test_other_provider_errors_propagate_untouched(tmp_path, monkeypatch):
    err = ProviderError("Error code: 500 - upstream failed", 500)
    llm = ScriptedLLM([err])
    agent, _ = make_agent(tmp_path, monkeypatch, llm)
    with pytest.raises(ProviderError):
        await agent.prompt("go")
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_compacted_view_has_no_orphans(tmp_path, monkeypatch):
    script = [tool_turn(i) for i in range(1, 9)] + [Message.assistant("done", stop_reason="stop")]
    llm = ScriptedLLM(script)
    agent, events = make_agent(tmp_path, monkeypatch, llm)
    seed(agent, [8000, 8000, 8000])
    assert await agent.prompt("long research") == StopReason.END_TURN
    assert [e for e in events if isinstance(e, ContextCompacted)]
    for call in llm.calls:
        check_tool_pairs(call.messages)
        assert est(agent, call) <= 0.75 * 6144 + 2100   # at most one new batch over the line


@pytest.mark.asyncio
async def test_overhead_warning_emitted_once_and_call_sent(tmp_path, monkeypatch):
    llm = ScriptedLLM([Message.assistant("ok", stop_reason="stop"),
                       Message.assistant("ok", stop_reason="stop")])
    agent, events = make_agent(tmp_path, monkeypatch, llm, system_prompt="s" * 28_000)
    await agent.prompt("one")
    await agent.prompt("two")
    warnings = [e for e in events if isinstance(e, ContextOverheadWarning)]
    assert len(warnings) == 1 and warnings[0].usable == 6144
    assert len(llm.calls) == 2


@pytest.mark.asyncio
async def test_provider_prompt_tokens_calibrate_the_estimate(tmp_path, monkeypatch):
    reply = Message.assistant("ok", stop_reason="stop")
    reply.usage = Usage(prompt_tokens=1234, completion_tokens=2, total_tokens=1236)
    llm = ScriptedLLM([reply])
    agent, _ = make_agent(tmp_path, monkeypatch, llm)
    await agent.prompt("hi")
    st = agent.session.compaction
    assert st.last_prompt_tokens == 1234
    assert st.last_view_chars == (fixed_chars(agent.harness.system_prompt, llm.calls[0].tools)
                                  + view_chars(llm.calls[0].messages))
