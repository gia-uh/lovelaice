import pytest
from unittest.mock import AsyncMock
from lingo.llm import Message
from lovelaice.agent import Agent, AgentConfig, AgentTool
from lovelaice.agent.loops.react_native import ReActNative
from lovelaice.acp.server import AcpServer
from lovelaice.acp.protocol import (
    JsonRpcRequest, JsonRpcResponse, JsonRpcNotification,
)


def _agent_with_canned_llm(monkeypatch, tmp_path, *, responses, idx=[0]):
    """Build an Agent whose LLM returns each of `responses` in turn.

    `idx` is a default-list-as-counter so the closure can mutate it; each
    test invocation gets a fresh closure via the `idx=[0]` default."""
    counter = {"n": 0}

    def _next_response(*a, **kw):
        i = counter["n"]
        counter["n"] = i + 1
        return responses[i]

    fake = AsyncMock()
    fake.chat = AsyncMock(side_effect=_next_response)
    monkeypatch.setattr("lovelaice.agent.agent._build_llm", lambda cfg: fake)
    return Agent(
        config=AgentConfig(model="m", system_prompt="SYS"),
        tools=[],
        loop=ReActNative(),
        session_path=tmp_path / f"s_{counter['n']}.jsonl",
    )


@pytest.mark.asyncio
async def test_initialize_returns_capabilities(monkeypatch, tmp_path):
    agent = _agent_with_canned_llm(
        monkeypatch, tmp_path,
        responses=[Message.assistant("ok", stop_reason="stop")])
    server = AcpServer(agent_factory=lambda: agent)
    resp = await server.handle_request(
        JsonRpcRequest(id=1, method="initialize", params={}))
    assert isinstance(resp, JsonRpcResponse)
    assert resp.result["protocolVersion"] == "0.1"
    assert "agentCapabilities" in resp.result
    assert resp.result["agentCapabilities"]["loadSession"] is False


@pytest.mark.asyncio
async def test_request_before_initialize_returns_error(monkeypatch, tmp_path):
    agent = _agent_with_canned_llm(
        monkeypatch, tmp_path,
        responses=[Message.assistant("ok", stop_reason="stop")])
    server = AcpServer(agent_factory=lambda: agent)
    resp = await server.handle_request(
        JsonRpcRequest(id=1, method="session/new", params={"cwd": "/x"}))
    assert resp.error is not None


@pytest.mark.asyncio
async def test_session_new_returns_id(monkeypatch, tmp_path):
    agent = _agent_with_canned_llm(
        monkeypatch, tmp_path,
        responses=[Message.assistant("ok", stop_reason="stop")])
    server = AcpServer(agent_factory=lambda: agent)
    await server.handle_request(JsonRpcRequest(id=1, method="initialize", params={}))
    resp = await server.handle_request(
        JsonRpcRequest(id=2, method="session/new",
                       params={"cwd": str(tmp_path)}))
    assert "sessionId" in resp.result


@pytest.mark.asyncio
async def test_session_prompt_streams_updates_and_returns_stop_reason(
        monkeypatch, tmp_path):
    agent = _agent_with_canned_llm(
        monkeypatch, tmp_path,
        responses=[Message.assistant("hello world", stop_reason="stop")])

    server = AcpServer(agent_factory=lambda: agent)
    await server.handle_request(JsonRpcRequest(id=1, method="initialize", params={}))
    new_resp = await server.handle_request(
        JsonRpcRequest(id=2, method="session/new", params={"cwd": str(tmp_path)}))
    sid = new_resp.result["sessionId"]

    notifications: list = []
    server.on_notification(lambda n: notifications.append(n))

    prompt_resp = await server.handle_request(JsonRpcRequest(
        id=3, method="session/prompt",
        params={"sessionId": sid,
                "prompt": [{"type": "text", "text": "hi"}]}))
    assert prompt_resp.result["stopReason"] == "end_turn"
    # At least one session/update notification with agent_message_chunk fired.
    kinds = [n.params.get("sessionUpdate") for n in notifications
             if n.method == "session/update"]
    assert "agent_message_chunk" in kinds


@pytest.mark.asyncio
async def test_session_prompt_unknown_session_returns_error(monkeypatch, tmp_path):
    agent = _agent_with_canned_llm(
        monkeypatch, tmp_path,
        responses=[Message.assistant("hi", stop_reason="stop")])
    server = AcpServer(agent_factory=lambda: agent)
    await server.handle_request(JsonRpcRequest(id=1, method="initialize", params={}))
    resp = await server.handle_request(JsonRpcRequest(
        id=2, method="session/prompt",
        params={"sessionId": "nonexistent", "prompt": [{"type": "text", "text": "hi"}]}))
    assert resp.error is not None


@pytest.mark.asyncio
async def test_unknown_method_returns_error(monkeypatch, tmp_path):
    agent = _agent_with_canned_llm(
        monkeypatch, tmp_path,
        responses=[Message.assistant("ok", stop_reason="stop")])
    server = AcpServer(agent_factory=lambda: agent)
    await server.handle_request(JsonRpcRequest(id=1, method="initialize", params={}))
    resp = await server.handle_request(
        JsonRpcRequest(id=2, method="some/unknown", params={}))
    assert resp.error is not None
    assert "method not found" in resp.error["message"].lower()
