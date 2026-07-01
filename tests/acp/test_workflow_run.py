from unittest.mock import AsyncMock

import pytest
from lingo.llm import Message

from lovelaice.acp.server import AcpServer
from lovelaice.acp.protocol import JsonRpcRequest, JsonRpcResponse
from lovelaice.agent import Agent, AgentConfig
from lovelaice.agent.loops.react_native import ReActNative


def _factory(monkeypatch, tmp_path):
    def make(*, conversation=None) -> Agent:
        fake = AsyncMock()
        fake.chat = AsyncMock(
            return_value=Message.assistant(
                '{"title":"T","generated_at":"2026-07-01","widgets":[]}',
                stop_reason="stop",
            )
        )
        monkeypatch.setattr("lovelaice.agent.agent._build_llm", lambda cfg: fake)
        return Agent(
            config=AgentConfig(model="m"),
            tools=[],
            loop=ReActNative(),
            session_path=tmp_path / "s.jsonl",
        )

    return make


@pytest.mark.asyncio
async def test_handle_workflow_run_returns_json(monkeypatch, tmp_path):
    server = AcpServer(agent_factory=_factory(monkeypatch, tmp_path))
    result = await server._handle_workflow_run(
        {
            "spec": {
                "name": "w",
                "root": {
                    "kind": "agent",
                    "prompt": "emit",
                    "output_schema": {"type": "object"},
                },
            },
            "inputs": None,
        }
    )
    assert result == {"result": {"title": "T", "generated_at": "2026-07-01", "widgets": []}}


@pytest.mark.asyncio
async def test_workflow_run_dispatch_via_handle_request(monkeypatch, tmp_path):
    server = AcpServer(agent_factory=_factory(monkeypatch, tmp_path))
    # initialize gate
    await server.handle_request(JsonRpcRequest(id=1, method="initialize", params={}))
    resp = await server.handle_request(
        JsonRpcRequest(
            id=2,
            method="workflow/run",
            params={"spec": {"name": "w", "root": {"kind": "agent", "prompt": "go", "output_schema": {}}}},
        )
    )
    assert isinstance(resp, JsonRpcResponse)
    assert resp.result == {"result": {"title": "T", "generated_at": "2026-07-01", "widgets": []}}


@pytest.mark.asyncio
async def test_workflow_tool_node_bridges_agent_tool(monkeypatch, tmp_path):
    from lingo.tools import tool as lingo_tool
    from lovelaice.agent import AgentTool

    written = {}

    @lingo_tool
    async def write_note(path: str, payload: dict) -> str:
        """Write a note."""
        written["path"] = path
        written["payload"] = payload
        return "ok:" + path

    def make(*, conversation=None) -> Agent:
        fake = AsyncMock()
        fake.chat = AsyncMock(
            return_value=Message.assistant('{"widgets": [1]}', stop_reason="stop")
        )
        monkeypatch.setattr("lovelaice.agent.agent._build_llm", lambda cfg: fake)
        return Agent(
            config=AgentConfig(model="m"),
            tools=[AgentTool(inner=write_note)],
            loop=ReActNative(),
            session_path=tmp_path / "s.jsonl",
        )

    server = AcpServer(agent_factory=make)
    out = await server._handle_workflow_run(
        {
            "spec": {
                "name": "w",
                "root": {
                    "kind": "sequence",
                    "children": [
                        {"kind": "agent", "prompt": "emit", "output_schema": {}, "name": "data"},
                        {"kind": "tool", "tool": "write_note",
                         "args": {"path": "kpis.md", "payload": "{data}"}},
                    ],
                },
            }
        }
    )
    assert written == {"path": "kpis.md", "payload": {"widgets": [1]}}
    assert out == {"result": {"tool": "write_note", "result": "ok:kpis.md"}}
