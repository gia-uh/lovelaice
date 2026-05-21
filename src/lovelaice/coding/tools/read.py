"""Coding host: read tool with line/byte truncation."""
from pathlib import Path

from lingo.tools import tool


MAX_LINES = 2000
MAX_BYTES = 100_000


@tool
async def read(path: str) -> str:
    """Read a file from disk. Truncates long files."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"no such file: {path}")
    if not p.is_file():
        raise IsADirectoryError(f"not a regular file: {path}")
    data = p.read_bytes()
    truncated_bytes = False
    if len(data) > MAX_BYTES:
        data = data[:MAX_BYTES]
        truncated_bytes = True
    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()
    truncated_lines = False
    if len(lines) > MAX_LINES:
        lines = lines[:MAX_LINES]
        truncated_lines = True
    out = "\n".join(lines)
    if truncated_bytes or truncated_lines:
        out += f"\n[truncated: kept {len(lines)} lines / {len(data)} bytes]"
    return out
