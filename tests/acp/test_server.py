import pytest
from unittest.mock import AsyncMock
from lingo.llm import Message, Usage
from lovelaice.agent import Agent, AgentConfig, AgentTool
from lovelaice.agent.events import AssistantMessageFinalized
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


# ---------------- Conversation persistence (Task 4) ----------------

import pytest_asyncio
from pathlib import Path
from beaver import AsyncBeaverDB
from lovelaice.agent.conversation import ConversationStore


def _stub_agent_factory(*args, **kwargs):
    class _StubAgent:
        async def prompt(self, text):
            from lovelaice.agent.errors import StopReason
            return StopReason.END_TURN
        def subscribe(self, fn):
            pass
    return _StubAgent()


@pytest_asyncio.fixture
async def conv_store(tmp_path: Path):
    db = AsyncBeaverDB(str(tmp_path / "lovelaice.db"))
    await db.connect()
    try:
        yield ConversationStore(db)
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_session_new_without_conversation_id_mints_one(conv_store):
    server = AcpServer(agent_factory=_stub_agent_factory, conversation_store=conv_store)
    await server.handle_request(
        JsonRpcRequest(id=1, method="initialize", params={}))
    resp = await server.handle_request(
        JsonRpcRequest(id=2, method="session/new", params={}))
    body = resp.result
    assert "sessionId" in body
    assert "conversationId" in body
    assert isinstance(body["conversationId"], str) and len(body["conversationId"]) > 0
    assert body.get("messages") == []


@pytest.mark.asyncio
async def test_session_new_with_known_conversation_id_returns_messages(conv_store):
    conv = await conv_store.create(model="m", system_prompt_hash="h")
    await conv_store.append(conv.id, Message.user("hi from yesterday"))
    server = AcpServer(agent_factory=_stub_agent_factory, conversation_store=conv_store)
    await server.handle_request(
        JsonRpcRequest(id=1, method="initialize", params={}))
    resp = await server.handle_request(
        JsonRpcRequest(id=2, method="session/new",
                       params={"conversationId": conv.id}))
    body = resp.result
    assert body["conversationId"] == conv.id
    assert len(body["messages"]) >= 1
    assert body["messages"][0]["role"] == "user"


@pytest.mark.asyncio
async def test_session_new_with_unknown_conversation_id_errors(conv_store):
    server = AcpServer(agent_factory=_stub_agent_factory, conversation_store=conv_store)
    await server.handle_request(
        JsonRpcRequest(id=1, method="initialize", params={}))
    resp = await server.handle_request(
        JsonRpcRequest(id=2, method="session/new",
                       params={"conversationId": "no-such-id"}))
    assert resp.error is not None
    assert "unknown conversation" in resp.error["message"].lower()


@pytest.mark.asyncio
async def test_conversation_archive_flips_flag(conv_store):
    conv = await conv_store.create(model="m", system_prompt_hash="h")
    server = AcpServer(agent_factory=_stub_agent_factory, conversation_store=conv_store)
    await server.handle_request(
        JsonRpcRequest(id=1, method="initialize", params={}))
    await server.handle_notification(JsonRpcNotification(
        method="conversation/archive",
        params={"conversationId": conv.id}))
    fresh = await conv_store.get(conv.id)
    assert fresh is not None
    assert fresh.archived is True


def _capture(server) -> list:
    """Collect outbound notifications from `server` into a list."""
    sink: list = []
    server.on_notification(sink.append)
    return sink


def test_finalized_message_with_usage_emits_a_usage_update():
    """lovelaice records usage per message; ACP consumers could not see it."""
    server = AcpServer(agent_factory=_stub_agent_factory)
    sink = _capture(server)
    msg = Message.assistant("hello", stop_reason="stop")
    msg.usage = Usage(prompt_tokens=10, completion_tokens=4, total_tokens=14)

    server._agent_event_to_notification("s1", AssistantMessageFinalized(message=msg))

    kinds = [n.params["sessionUpdate"] for n in sink]
    assert "usage" in kinds
    params = next(n.params for n in sink if n.params["sessionUpdate"] == "usage")
    assert params["sessionId"] == "s1"
    assert params["usage"] == {"prompt_tokens": 10, "completion_tokens": 4,
                               "total_tokens": 14}


def test_finalized_message_without_usage_emits_no_usage_update():
    server = AcpServer(agent_factory=_stub_agent_factory)
    sink = _capture(server)
    msg = Message.assistant("hello", stop_reason="stop")
    assert msg.usage is None

    server._agent_event_to_notification("s1", AssistantMessageFinalized(message=msg))

    assert "usage" not in [n.params["sessionUpdate"] for n in sink]


def test_usage_is_emitted_for_a_textless_turn():
    """A turn that ends in tool calls has no assistant text but still costs
    tokens — emitting usage inside the `if text:` guard would report it free."""
    server = AcpServer(agent_factory=_stub_agent_factory)
    sink = _capture(server)
    msg = Message.assistant("", stop_reason="tool_calls")
    msg.usage = Usage(prompt_tokens=7, completion_tokens=0, total_tokens=7)

    server._agent_event_to_notification("s1", AssistantMessageFinalized(message=msg))

    kinds = [n.params["sessionUpdate"] for n in sink]
    assert kinds == ["usage"]
    assert sink[0].params["usage"]["total_tokens"] == 7
