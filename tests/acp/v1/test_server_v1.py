import asyncio
import os
from pathlib import Path

import pytest
import acp

from lovelaice.acp.v1.server import AcpServerV1


def _factory(**kw):
    raise AssertionError("factory not needed for initialize")


@pytest.mark.asyncio
async def test_initialize_advertises_protocol_v1():
    server = AcpServerV1(agent_factory=_factory)
    resp = await server.initialize(protocol_version=1)
    assert isinstance(resp, acp.InitializeResponse)
    assert resp.protocol_version == 1
    # load_session is a VS4 capability — advertised False for now.
    assert resp.agent_capabilities.load_session is False


class _FakeAgent:
    def __init__(self):
        self.subscribers = []

    def subscribe(self, fn):
        self.subscribers.append(fn)


@pytest.mark.asyncio
async def test_new_session_registers_agent_and_subscribes():
    made = _FakeAgent()
    server = AcpServerV1(agent_factory=lambda **kw: made)
    resp = await server.new_session(cwd="/tmp")
    assert isinstance(resp, acp.NewSessionResponse)
    sid = resp.session_id
    assert sid and server._sessions[sid] is made
    assert made.subscribers, "agent should have an event subscriber wired"


class _FakeConn:
    def __init__(self):
        self.updates = []

    async def session_update(self, session_id, update, **kw):
        self.updates.append((session_id, update))


class _Msg:
    content = "hello from agent"


@pytest.mark.asyncio
async def test_emit_translates_message_and_tool_events():
    from lovelaice.agent.events import (
        AssistantMessageFinalized, ToolExecutionStart, ToolExecutionEnd,
    )
    conn = _FakeConn()
    server = AcpServerV1(agent_factory=lambda **kw: _FakeAgent())
    server.on_connect(conn)
    server._loop = asyncio.get_running_loop()

    server._emit("s1", AssistantMessageFinalized(message=_Msg()))
    server._emit("s1", ToolExecutionStart(call_id="c1", name="read", args={"path": "x"}))
    server._emit("s1", ToolExecutionEnd(call_id="c1", result=None, is_error=False))
    await asyncio.sleep(0.05)  # let scheduled coros run

    kinds = [type(u).__name__ for _sid, u in conn.updates]
    assert "AgentMessageChunk" in kinds
    assert "ToolCallStart" in kinds
    assert "ToolCallProgress" in kinds


def _real_agent_factory(tmp_path):
    os.environ["LOVELAICE_FAKE_LLM"] = "1"
    from unittest.mock import AsyncMock
    from lingo.llm import Message
    import lovelaice.agent.agent as agent_mod
    fake = AsyncMock()
    fake.chat = AsyncMock(
        return_value=Message.assistant("done", stop_reason="stop"))
    agent_mod._build_llm = lambda cfg: fake

    from lovelaice.agent import Agent, AgentConfig
    from lovelaice.agent.loops.react_native import ReActNative

    def factory(**kw):
        cfg = AgentConfig(model="fake/model", cwd=str(tmp_path))
        return Agent(config=cfg, tools=[], loop=ReActNative(),
                     session_path=tmp_path / "s.jsonl")
    return factory


def test_prompt_text_extraction_handles_dicts_and_content_blocks():
    # Over the real ACP wire the SDK delivers typed TextContentBlock objects,
    # not dicts. Both must yield the text (regression: the object path once
    # returned "" and the agent saw an empty prompt).
    block = acp.text_block("from object")
    assert AcpServerV1._prompt_text([{"type": "text", "text": "from dict"}]) == "from dict"
    assert AcpServerV1._prompt_text([block]) == "from object"
    assert AcpServerV1._prompt_text([block, {"type": "text", "text": "!"}]) == "from object!"


@pytest.mark.asyncio
async def test_prompt_returns_stop_reason(tmp_path):
    server = AcpServerV1(agent_factory=_real_agent_factory(tmp_path))
    server.on_connect(_FakeConn())
    new = await server.new_session(cwd=str(tmp_path))
    resp = await server.prompt(
        prompt=[{"type": "text", "text": "hi"}], session_id=new.session_id)
    assert isinstance(resp, acp.PromptResponse)
    assert resp.stop_reason == "end_turn"
