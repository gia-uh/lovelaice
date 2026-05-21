"""Coding host: bash tool with timeout + output truncation."""
import asyncio

from lingo.tools import tool


MAX_OUTPUT_LINES = 5000
DEFAULT_TIMEOUT = 60


@tool
async def bash(command: str) -> str:
    """Run a shell command. Returns merged stdout+stderr and exit code."""
    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(),
                                        timeout=DEFAULT_TIMEOUT)
    except asyncio.TimeoutError:
        proc.kill()
        return f"[timeout after {DEFAULT_TIMEOUT}s]"
    text = out.decode("utf-8", errors="replace")
    lines = text.splitlines()
    if len(lines) > MAX_OUTPUT_LINES:
        lines = lines[:MAX_OUTPUT_LINES] + [
            f"[truncated: kept {MAX_OUTPUT_LINES} lines]"
        ]
        text = "\n".join(lines)
    return f"{text}\nexit code: {proc.returncode}"
