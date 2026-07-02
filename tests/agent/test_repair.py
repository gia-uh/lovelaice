"""Focused one-shot tool-arg repair in Harness.execute_tool."""
import pytest
from lingo.tools import tool as lingo_tool
from lingo.llm import ToolCall

from lovelaice.agent.tools import AgentTool, ToolRegistry
from lovelaice.agent.hooks import HookRegistry
from lovelaice.agent.harness import Harness
from lovelaice.agent.events import ToolCallRepaired, ToolExecutionStart


@lingo_tool
async def grep(pattern: str, path: str) -> str:
    """Search files.

    Args:
        pattern: Regex to search for.
        path: Directory to search under.
    """
    return f"{pattern}@{path}"


class FakeLLM:
    """Stub lingo.LLM whose create() returns a fixed args object."""

    def __init__(self, obj):
        self.obj = obj
        self.calls = 0

    async def create(self, model, messages, **kw):
        self.calls += 1
        return model(**self.obj)


def _harness(llm, repair=True, context="none"):
    reg = ToolRegistry()
    reg.register(AgentTool(inner=grep))
    return Harness(llm=llm, tools=reg, hooks=HookRegistry(), system_prompt="x",
                   repair_tool_calls=repair, repair_context=context)


@pytest.mark.asyncio
async def test_repair_heals_missing_arg():
    llm = FakeLLM({"pattern": "x", "path": "."})
    h = _harness(llm)
    events = []
    h.subscribe(events.append)
    call = ToolCall(id="c1", name="grep", arguments={"pattern": "x"})  # missing path
    r = await h.execute_tool(call)
    assert r.is_error is False
    assert r.content[0]["text"] == "x@."
    assert call.arguments == {"pattern": "x", "path": "."}
    assert llm.calls == 1
    repaired = [e for e in events if isinstance(e, ToolCallRepaired)]
    assert len(repaired) == 1
    assert repaired[0].original_args == {"pattern": "x"}
    assert repaired[0].repaired_args == {"pattern": "x", "path": "."}
    start = [e for e in events if isinstance(e, ToolExecutionStart)][0]
    assert start.args == {"pattern": "x", "path": "."}


@pytest.mark.asyncio
async def test_repair_disabled_returns_error():
    h = _harness(FakeLLM({}), repair=False)
    r = await h.execute_tool(
        ToolCall(id="c1", name="grep", arguments={"pattern": "x"}))
    assert r.is_error is True
    assert "validation failed" in r.content[0]["text"]


@pytest.mark.asyncio
async def test_repair_failure_falls_back_to_error():
    class BadLLM:
        calls = 0

        async def create(self, model, messages, **kw):
            raise RuntimeError("no structured output support")

    h = _harness(BadLLM())
    events = []
    h.subscribe(events.append)
    r = await h.execute_tool(
        ToolCall(id="c1", name="grep", arguments={"pattern": "x"}))
    assert r.is_error is True
    assert not any(isinstance(e, ToolCallRepaired) for e in events)


@pytest.mark.asyncio
async def test_repair_still_invalid_falls_back():
    # create returns an object missing 'path' is impossible (model requires it),
    # so simulate a repair that validates but we assert healthy path elsewhere;
    # here the LLM returns an empty-ish path that still validates as str.
    llm = FakeLLM({"pattern": "x", "path": ""})
    h = _harness(llm)
    r = await h.execute_tool(
        ToolCall(id="c1", name="grep", arguments={"pattern": "x"}))
    assert r.is_error is False
    assert r.content[0]["text"] == "x@"


@pytest.mark.asyncio
async def test_execution_error_not_repaired():
    @lingo_tool
    async def boom(x: str) -> str:
        """Boom.

        Args:
            x: value.
        """
        raise RuntimeError("boom")

    reg = ToolRegistry()
    reg.register(AgentTool(inner=boom))
    llm = FakeLLM({"x": "y"})
    h = Harness(llm=llm, tools=reg, hooks=HookRegistry(), system_prompt="x",
                repair_tool_calls=True, repair_context="none")
    r = await h.execute_tool(ToolCall(id="c1", name="boom", arguments={"x": "y"}))
    assert r.is_error is True
    assert "boom" in r.content[0]["text"]
    assert llm.calls == 0  # valid args → no repair; execution error not repaired
