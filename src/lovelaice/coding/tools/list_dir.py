"""Coding host: list_dir tool (delegates to lovelaice.tools.files.list_)."""
from lingo.tools import tool

from lovelaice.tools import files


@tool
async def list_dir(path: str = ".") -> list[str]:
    """List the entries in a directory. Returns a flat, sorted list of names
    (not recursive). Use `bash("find ...")` or `glob` for recursive listings.
    """
    return await files.list_(path)
