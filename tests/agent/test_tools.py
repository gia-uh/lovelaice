import pytest
from lingo.tools import tool as lingo_tool
from lovelaice.agent.tools import AgentTool, ToolRegistry, ToolResult


@lingo_tool
async def echo(text: str) -> str:
    """Echo the input back."""
    return text


@lingo_tool
async def write(path: str, content: str) -> str:
    """Write content to path."""
    return "ok"


def test_agent_tool_wraps_lingo_tool():
    at = AgentTool(inner=echo, sequential=False, kind="other")
    assert at.name == "echo"
    assert at.kind == "other"
    assert at.sequential is False


def test_agent_tool_default_kind_and_sequential():
    at = AgentTool(inner=echo)
    assert at.kind == "other"
    assert at.sequential is False


def test_agent_tool_title_for_with_template():
    at = AgentTool(inner=echo, title_template="Echoing {text}")
    assert at.title_for({"text": "hello"}) == "Echoing hello"


def test_agent_tool_title_for_without_template_uses_name():
    at = AgentTool(inner=echo)
    assert at.title_for({"text": "hello"}) == "echo"


def test_agent_tool_title_for_missing_arg_falls_back_to_name():
    at = AgentTool(inner=echo, title_template="Reading {path}")
    # 'path' not in args → KeyError caught → fallback to name
    assert at.title_for({"text": "hello"}) == "echo"


def test_tool_registry_register_and_lookup():
    reg = ToolRegistry()
    at = AgentTool(inner=echo, kind="read")
    reg.register(at)
    assert reg.get("echo") is at
    assert reg.get("nonexistent") is None


def test_tool_registry_all():
    reg = ToolRegistry()
    reg.register(AgentTool(inner=echo))
    reg.register(AgentTool(inner=write))
    names = {t.name for t in reg.all()}
    assert names == {"echo", "write"}


def test_tool_registry_lingo_tools_returns_underlying():
    reg = ToolRegistry()
    reg.register(AgentTool(inner=echo))
    lst = reg.lingo_tools()
    assert lst == [echo]


def test_tool_registry_any_sequential_false():
    reg = ToolRegistry()
    reg.register(AgentTool(inner=echo, sequential=False))
    reg.register(AgentTool(inner=write, sequential=False))
    assert reg.any_sequential(["echo", "write"]) is False


def test_tool_registry_any_sequential_true_when_any_sequential():
    reg = ToolRegistry()
    reg.register(AgentTool(inner=echo, sequential=False))
    reg.register(AgentTool(inner=write, sequential=True))
    assert reg.any_sequential(["echo", "write"]) is True


def test_tool_registry_any_sequential_unknown_tool_ignored():
    reg = ToolRegistry()
    reg.register(AgentTool(inner=echo, sequential=False))
    # unknown tool name in batch — should not crash, just not contribute
    assert reg.any_sequential(["echo", "unknown_tool"]) is False


def test_tool_result_from_value_str():
    r = ToolResult.from_value("hello")
    assert r.content == [{"type": "text", "text": "hello"}]
    assert r.is_error is False
    assert r.terminate is False
    assert r.locations is None


def test_tool_result_from_value_non_str():
    r = ToolResult.from_value({"foo": "bar"})
    assert r.content[0]["type"] == "text"
    assert "foo" in r.content[0]["text"]  # str() of the dict
    assert r.raw_output == {"foo": "bar"}


def test_tool_result_passthrough_when_already_toolresult():
    inner = ToolResult(content=[{"type": "text", "text": "x"}], is_error=True)
    r = ToolResult.from_value(inner)
    assert r is inner


def test_tool_result_from_exception():
    r = ToolResult.from_exception(RuntimeError("boom"))
    assert r.is_error is True
    assert r.content[0]["text"] == "RuntimeError: boom"
