import pytest
from lingo.tools import tool as lingo_tool
from lingo.llm import ToolCall
from lovelaice.agent.tools import AgentTool, ToolRegistry
from lovelaice.agent.hooks import HookRegistry, Block
from lovelaice.agent.harness import Harness


@lingo_tool
async def echo(text: str) -> str:
    """Echo."""
    return text


@lingo_tool
async def boom(text: str) -> str:
    """Always raises."""
    raise RuntimeError("boom")


def _harness(tools_list, hooks=None):
    reg = ToolRegistry()
    for t in tools_list:
        reg.register(t)
    return Harness(llm=None, tools=reg, hooks=hooks or HookRegistry(),
                   system_prompt="x")


@pytest.mark.asyncio
async def test_execute_tool_happy():
    h = _harness([AgentTool(inner=echo, kind="other")])
    call = ToolCall(id="c1", name="echo", arguments={"text": "hi"})
    r = await h.execute_tool(call)
    assert r.is_error is False
    assert r.content[0]["text"] == "hi"


@pytest.mark.asyncio
async def test_execute_tool_exception_becomes_error_result():
    h = _harness([AgentTool(inner=boom)])
    call = ToolCall(id="c1", name="boom", arguments={"text": "x"})
    r = await h.execute_tool(call)
    assert r.is_error is True
    assert "RuntimeError: boom" in r.content[0]["text"]


@pytest.mark.asyncio
async def test_execute_tool_blocked_by_hook():
    hooks = HookRegistry()
    hooks.register("tool_call", lambda call: Block("policy"))
    h = _harness([AgentTool(inner=echo)], hooks=hooks)
    call = ToolCall(id="c1", name="echo", arguments={"text": "x"})
    r = await h.execute_tool(call)
    assert r.is_error is True
    assert "policy" in r.content[0]["text"]


@pytest.mark.asyncio
async def test_execute_tool_unknown_tool_returns_error():
    h = _harness([])
    call = ToolCall(id="c1", name="nonexistent", arguments={})
    r = await h.execute_tool(call)
    assert r.is_error is True
    assert "unknown tool" in r.content[0]["text"].lower()


@pytest.mark.asyncio
async def test_execute_tool_validation_error_missing_required_arg():
    h = _harness([AgentTool(inner=echo)])
    call = ToolCall(id="c1", name="echo", arguments={})  # missing text
    r = await h.execute_tool(call)
    assert r.is_error is True
    assert "validation" in r.content[0]["text"].lower() or "missing" in r.content[0]["text"].lower() or "field required" in r.content[0]["text"].lower()


@pytest.mark.asyncio
async def test_execute_tool_emits_start_and_end_events():
    h = _harness([AgentTool(inner=echo)])
    events = []
    h.subscribe(lambda ev: events.append(ev))
    call = ToolCall(id="c1", name="echo", arguments={"text": "hi"})
    await h.execute_tool(call)
    types = [type(e).__name__ for e in events]
    assert "ToolExecutionStart" in types
    assert "ToolExecutionEnd" in types


@pytest.mark.asyncio
async def test_execute_tools_batch_parallel_preserves_order():
    h = _harness([AgentTool(inner=echo)])
    calls = [
        ToolCall(id=f"c{i}", name="echo", arguments={"text": str(i)})
        for i in range(3)
    ]
    rs = await h.execute_tools_batch(calls)
    assert len(rs) == 3
    assert [r.content[0]["text"] for r in rs] == ["0", "1", "2"]


@pytest.mark.asyncio
async def test_execute_tools_batch_sequential_when_any_sequential():
    """If any tool in the batch has sequential=True, the entire batch runs serially.
    Output order matches source order regardless."""
    seq_tool = AgentTool(inner=echo, sequential=True)
    h = _harness([seq_tool])
    calls = [ToolCall(id=f"c{i}", name="echo", arguments={"text": str(i)})
             for i in range(3)]
    rs = await h.execute_tools_batch(calls)
    assert [r.content[0]["text"] for r in rs] == ["0", "1", "2"]


@pytest.mark.asyncio
async def test_execute_tools_batch_empty_returns_empty():
    h = _harness([])
    assert await h.execute_tools_batch([]) == []
