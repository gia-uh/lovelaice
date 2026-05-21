"""Coding host: tool_call guard hooks (path + bash prefix)."""
from pathlib import Path

from lovelaice.agent.hooks import Block


BAD_BASH_PREFIXES = (
    "rm -rf /",
    "rm -rf ~",
    "sudo ",
)


def path_guard(call, *, cwd: str):
    """Block read/write/edit calls whose `path` argument is outside cwd."""
    if call.name not in ("read", "write", "edit"):
        return None
    path = call.arguments.get("path")
    if path is None:
        return None
    try:
        resolved = Path(path).resolve()
        cwd_resolved = Path(cwd).resolve()
        resolved.relative_to(cwd_resolved)
    except (ValueError, OSError):
        return Block(reason=f"path {path} is outside cwd {cwd}")
    return None


def bash_prefix_guard(call):
    """Block bash commands with dangerous prefixes."""
    if call.name != "bash":
        return None
    cmd = call.arguments.get("command", "").strip()
    for bad in BAD_BASH_PREFIXES:
        if cmd.startswith(bad):
            return Block(reason=f"bash prefix '{bad}' blocked by host policy")
    return None
