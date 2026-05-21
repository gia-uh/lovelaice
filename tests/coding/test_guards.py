import pytest
from lingo.llm import ToolCall
from lovelaice.coding.hooks import path_guard, bash_prefix_guard
from lovelaice.agent.hooks import Block


def test_path_guard_blocks_outside_cwd(tmp_path):
    # Use an absolute path that does NOT live under tmp_path.
    call = ToolCall(id="c1", name="read", arguments={"path": "/etc/passwd"})
    result = path_guard(call, cwd=str(tmp_path))
    assert isinstance(result, Block)


def test_path_guard_allows_within_cwd(tmp_path):
    inside = tmp_path / "x.py"
    inside.write_text("# noop")
    call = ToolCall(id="c1", name="read", arguments={"path": str(inside)})
    result = path_guard(call, cwd=str(tmp_path))
    assert result is None


def test_path_guard_ignores_non_path_tools(tmp_path):
    """Only read/write/edit are guarded; other tools pass through."""
    call = ToolCall(id="c1", name="bash", arguments={"command": "ls"})
    result = path_guard(call, cwd=str(tmp_path))
    assert result is None


def test_bash_prefix_guard_blocks_rm_rf():
    call = ToolCall(id="c1", name="bash", arguments={"command": "rm -rf /"})
    result = bash_prefix_guard(call)
    assert isinstance(result, Block)


def test_bash_prefix_guard_blocks_sudo():
    call = ToolCall(id="c1", name="bash", arguments={"command": "sudo apt install x"})
    result = bash_prefix_guard(call)
    assert isinstance(result, Block)


def test_bash_prefix_guard_allows_safe_command():
    call = ToolCall(id="c1", name="bash", arguments={"command": "ls -la"})
    result = bash_prefix_guard(call)
    assert result is None


def test_bash_prefix_guard_ignores_non_bash_tools():
    call = ToolCall(id="c1", name="read", arguments={"path": "/etc/passwd"})
    result = bash_prefix_guard(call)
    assert result is None
