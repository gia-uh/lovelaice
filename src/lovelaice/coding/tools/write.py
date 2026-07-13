"""Coding host: write tool (delegates to lovelaice.tools.files.write)."""
from lingo.tools import tool

from lovelaice.tools import files


@tool
async def write(path: str, content: str) -> str:
    """Write content to a file, overwriting it if it already exists.
    Creates parent directories as needed.

    Use this for full-file rewrites or for creating new files. For surgical
    changes inside an existing file, prefer `edit` so the rest of the file
    is preserved verbatim.
    """
    return await files.write(path, content)
