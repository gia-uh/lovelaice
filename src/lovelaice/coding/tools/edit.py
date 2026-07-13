"""Coding host: edit tool (delegates to lovelaice.tools.files.edit)."""
from lingo.tools import tool

from lovelaice.tools import files


@tool
async def edit(path: str, old: str, new: str) -> str:
    """Replace the first exact occurrence of `old` with `new` in the file at
    `path`. Both are matched literally (no regex).

    Fails if `old` does not appear in the file, or if it appears more than
    once — in that case, include more surrounding context in `old` so the
    match is unique.
    """
    return await files.edit(path, old, new)
