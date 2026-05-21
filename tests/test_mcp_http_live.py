"""Live integration: an Agent uses HTTP MCP tools end-to-end."""

import asyncio
import os
import pytest

from mcp.server.fastmcp import FastMCP

from lovelaice.mcp import connect, _MCPTool, _params_from_input_schema


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="requires OPENROUTER_API_KEY for live LLM",
)
async def test_agent_calls_http_mcp_tool_end_to_end(tmp_path):
    import uvicorn

    server = FastMCP("test")
    called_with: list[str] = []

    @server.tool()
    def get_secret_code(name: str) -> str:
        """Return the secret code for a given person."""
        called_with.append(name)
        return f"The secret code for {name} is 4242."

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
    url = f"http://127.0.0.1:{port}/mcp"

    try:
        async with connect({"name": "test", "url": url}) as mcp_session:
            tools_resp = await mcp_session.list_tools()
            # Use a sanitized display name (no ':') since OpenAI-compatible
            # tool-call schemas restrict names to [a-zA-Z0-9_-].
            mcp_lingo_tools = [
                _MCPTool(
                    display_name=t.name,
                    description=(t.description or "MCP tool"),
                    params=_params_from_input_schema(
                        getattr(t, "inputSchema", None) or {}
                    ),
                    session=mcp_session,
                    tool_name=t.name,
                )
                for t in tools_resp.tools
            ]

            from lovelaice.agent import Agent, AgentConfig, AgentTool
            from lovelaice.agent.loops.react_native import ReActNative

            agent_tools = [
                AgentTool(inner=lt, kind="other") for lt in mcp_lingo_tools
            ]
            agent = Agent(
                config=AgentConfig(
                    model="anthropic/claude-haiku-4-5",
                    api_key=os.environ["OPENROUTER_API_KEY"],
                    base_url="https://openrouter.ai/api/v1",
                    system_prompt=(
                        "You are a helpful assistant. When the user asks for "
                        "someone's secret code, you MUST call the available "
                        "tool to retrieve it. Do not make up codes."
                    ),
                    cwd=str(tmp_path),
                ),
                tools=agent_tools,
                loop=ReActNative(),
                session_path=tmp_path / "session.jsonl",
            )
            await agent.prompt(
                "Use your tools to look up Alice's secret code and tell me what it is."
            )
            assert called_with == ["Alice"], (
                f"expected the tool to be called with 'Alice', got {called_with!r}"
            )
    finally:
        uv_server.should_exit = True
        await task
