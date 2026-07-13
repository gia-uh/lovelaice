"""Coding host: glob tool (delegates to lovelaice.tools.search.glob)."""
from lingo.tools import tool

from lovelaice.tools import search


@tool
async def glob(pattern: str) -> list[str]:
    """Return paths matching `pattern` (e.g. "src/**/*.py") as forward-slash
    strings relative to cwd. Patterns follow Python's pathlib glob syntax.

    Honors `.gitignore` at the workspace root and always excludes `.git/`.
    """
    return await search.glob(pattern)
