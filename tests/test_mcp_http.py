"""HTTP MCP transport — connects to a FastMCP server over streamable HTTP."""

import asyncio
import pytest
import pytest_asyncio
from mcp.server.fastmcp import FastMCP
from lovelaice.mcp import connect


@pytest_asyncio.fixture
async def http_mcp_url():
    """Spin up a FastMCP server on a random port; yield the URL."""
    import uvicorn

    server = FastMCP("test")

    @server.tool()
    def echo(text: str) -> str:
        """Echo the input."""
        return f"echo: {text}"

    @server.tool()
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    config = uvicorn.Config(
        server.streamable_http_app(),
        host="127.0.0.1",
        port=0,
        log_level="error",
    )
    uv_server = uvicorn.Server(config)
    task = asyncio.create_task(uv_server.serve())
    while not uv_server.started:
        await asyncio.sleep(0.01)
    port = uv_server.servers[0].sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}/mcp"
    uv_server.should_exit = True
    await task


@pytest.mark.asyncio
async def test_http_mcp_connect_and_list_tools(http_mcp_url):
    async with connect({"name": "test", "url": http_mcp_url}) as session:
        tools = await session.list_tools()
        tool_names = {t.name for t in tools.tools}
        assert "echo" in tool_names
        assert "add" in tool_names


@pytest.mark.asyncio
async def test_http_mcp_call_tool(http_mcp_url):
    async with connect({"name": "test", "url": http_mcp_url}) as session:
        result = await session.call_tool("echo", {"text": "hello"})
        assert "echo: hello" in str(result.content)


@pytest.mark.asyncio
async def test_http_mcp_with_bearer_auth(http_mcp_url):
    async with connect(
        {
            "name": "test",
            "url": http_mcp_url,
            "auth": {"bearer": "test-token-123"},
        }
    ) as session:
        tools = await session.list_tools()
        assert len(tools.tools) >= 2


@pytest.mark.asyncio
async def test_http_mcp_forwards_extra_headers(monkeypatch):
    """`config["headers"]` extends the per-request header dict that the
    streamable_http client sees — used by warden to pass scoping keys like
    X-Peacock-Conversation through to the downstream MCP server."""
    captured: dict[str, dict[str, str]] = {}

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _fake_streamablehttp_client(url, headers=None):
        captured["url"] = url
        captured["headers"] = dict(headers or {})
        raise RuntimeError("stop-before-real-connect")
        yield  # pragma: no cover — required for asynccontextmanager typing

    from lovelaice import mcp as _mcp_mod

    monkeypatch.setattr(_mcp_mod, "streamablehttp_client", _fake_streamablehttp_client)

    with pytest.raises(_mcp_mod.MCPTransportError):
        async with _mcp_mod.connect(
            {
                "name": "peacock",
                "url": "http://peacock.ainbox.local:8001/mcp",
                "auth": {"bearer": "svc-token"},
                "headers": {"X-Peacock-Conversation": "conv-abc"},
            }
        ) as _:
            pass

    assert captured["url"] == "http://peacock.ainbox.local:8001/mcp"
    assert captured["headers"]["Authorization"] == "Bearer svc-token"
    assert captured["headers"]["X-Peacock-Conversation"] == "conv-abc"
