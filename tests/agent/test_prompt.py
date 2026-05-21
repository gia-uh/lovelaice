from lingo.tools import tool as lingo_tool
from lovelaice.agent.tools import AgentTool, ToolRegistry
from lovelaice.agent.prompt import assemble_system_prompt


@lingo_tool
async def read(path: str) -> str:
    """Read a file."""
    return ""


@lingo_tool
async def bash(command: str) -> str:
    """Run a shell command."""
    return ""


def test_assemble_minimal_no_tools():
    reg = ToolRegistry()
    out = assemble_system_prompt(base="You are helpful.", tools=reg, cwd="/x", today="2026-01-01")
    assert "You are helpful." in out
    assert "/x" in out
    assert "2026-01-01" in out
    assert "Available tools" not in out


def test_assemble_with_tools():
    reg = ToolRegistry()
    reg.register(AgentTool(inner=read, kind="read"))
    reg.register(AgentTool(inner=bash, kind="execute"))
    out = assemble_system_prompt(base="You are helpful.", tools=reg, cwd="/x", today="2026-01-01")
    assert "Available tools" in out
    assert "read" in out
    assert "Read a file." in out
    assert "bash" in out
    assert "Run a shell command." in out


def test_assemble_is_deterministic():
    reg = ToolRegistry()
    reg.register(AgentTool(inner=read))
    a = assemble_system_prompt(base="b", tools=reg, cwd="/x", today="2026-01-01")
    b = assemble_system_prompt(base="b", tools=reg, cwd="/x", today="2026-01-01")
    assert a == b


def test_assemble_default_today_uses_iso_date():
    """When today is None, the default is today's ISO date (YYYY-MM-DD)."""
    from datetime import date
    reg = ToolRegistry()
    out = assemble_system_prompt(base="b", tools=reg, cwd="/x")
    assert date.today().isoformat() in out


def test_assemble_tool_signature_shows_param_name_and_type():
    reg = ToolRegistry()
    reg.register(AgentTool(inner=read))
    out = assemble_system_prompt(base="b", tools=reg, cwd="/x", today="2026-01-01")
    # Tool signature should include the parameter name with type annotation
    assert "path" in out
    assert "str" in out
