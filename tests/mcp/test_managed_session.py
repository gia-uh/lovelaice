import sys

import pytest

from lovelaice.mcp import start_managed_session

SERVER = '''
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("echo")

@mcp.tool()
def ping(msg: str) -> str:
    return f"pong:{msg}"

mcp.run(transport="stdio")
'''


@pytest.mark.asyncio
async def test_managed_stdio_session_lists_and_calls_and_closes(tmp_path):
    script = tmp_path / "echo_server.py"
    script.write_text(SERVER)
    sess = start_managed_session(
        {"name": "echo", "command": sys.executable, "args": [str(script)]})
    try:
        assert any(t.name == "ping" for t in sess.tools)
        result = await sess.call_tool("ping", {"msg": "hi"})
        text = "".join(
            getattr(p, "text", "") for p in (getattr(result, "content", None) or []))
        assert "pong:hi" in text
    finally:
        await sess.aclose()
