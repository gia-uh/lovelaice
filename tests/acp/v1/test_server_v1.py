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


def test_mcp_specs_from_acp_maps_http_and_stdio():
    class Hdr:
        def __init__(self, name, value):
            self.name, self.value = name, value

    class H:  # HttpMcpServer-shaped
        name, url = "aegis", "http://x/mcp"
        headers = [Hdr("Authorization", "Bearer z")]

    class S:  # McpServerStdio-shaped
        name, command, args, env = "local", "mytool", ["--x"], None

    specs = AcpServerV1._mcp_specs_from_acp([H(), S()])
    assert specs[0] == {"name": "aegis", "url": "http://x/mcp",
                        "headers": {"Authorization": "Bearer z"}}
    assert specs[1] == {"name": "local", "command": "mytool",
                        "args": ["--x"], "env": None}


def test_prompt_text_extraction_handles_dicts_and_content_blocks():
    # Over the real ACP wire the SDK delivers typed TextContentBlock objects,
    # not dicts. Both must yield the text (regression: the object path once
    # returned "" and the agent saw an empty prompt).
    block = acp.text_block("from object")
    assert AcpServerV1._prompt_text([{"type": "text", "text": "from dict"}]) == "from dict"
    assert AcpServerV1._prompt_text([block]) == "from object"
    assert AcpServerV1._prompt_text([block, {"type": "text", "text": "!"}]) == "from object!"


_MCP_ECHO = '''
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("echo")

@mcp.tool()
def ping(msg: str) -> str:
    return f"pong:{msg}"

mcp.run(transport="stdio")
'''


@pytest.mark.asyncio
async def test_new_session_attaches_mcp_tools_and_close_tears_down(tmp_path):
    import sys as _sys
    script = tmp_path / "echo_server.py"
    script.write_text(_MCP_ECHO)

    captured = {}

    def factory(*, mcp_tools=None, **kw):
        captured["mcp_tools"] = mcp_tools or []
        agent = _FakeAgent()
        agent.harness_tools = {t.name for t in (mcp_tools or [])}
        return agent

    server = AcpServerV1(agent_factory=factory)
    # ACP-shaped stdio server object.
    class Stdio:
        name, command, args, env = "echo", _sys.executable, [str(script)], None

    resp = await server.new_session(cwd=str(tmp_path), mcp_servers=[Stdio()])
    sid = resp.session_id
    assert any(t.name == "mcp_echo_ping" for t in captured["mcp_tools"])
    assert server._mcp_sessions[sid], "managed session retained for teardown"

    await server.close_session(sid)
    assert sid not in server._mcp_sessions
    assert sid not in server._sessions


@pytest.mark.asyncio
async def test_deltas_stream_and_finalized_not_duplicated():
    from lovelaice.agent.events import (
        AssistantMessageDelta, AssistantMessageFinalized)
    conn = _FakeConn()
    server = AcpServerV1(agent_factory=lambda **kw: _FakeAgent())
    server.on_connect(conn)
    server._loop = asyncio.get_running_loop()
    server._streamed_any = False
    server._emit("s", AssistantMessageDelta(text="he"))
    server._emit("s", AssistantMessageDelta(text="llo"))
    server._emit("s", AssistantMessageFinalized(message=_Msg()))
    await asyncio.sleep(0.05)
    texts = [u.content.text for _s, u in conn.updates
             if type(u).__name__ == "AgentMessageChunk"]
    assert texts == ["he", "llo"]  # finalized did NOT re-emit content


@pytest.mark.asyncio
async def test_finalized_emits_content_when_no_streaming():
    from lovelaice.agent.events import AssistantMessageFinalized
    conn = _FakeConn()
    server = AcpServerV1(agent_factory=lambda **kw: _FakeAgent())
    server.on_connect(conn)
    server._loop = asyncio.get_running_loop()
    server._streamed_any = False
    server._emit("s", AssistantMessageFinalized(message=_Msg()))
    await asyncio.sleep(0.05)
    texts = [u.content.text for _s, u in conn.updates
             if type(u).__name__ == "AgentMessageChunk"]
    assert texts == ["hello from agent"]  # fallback emit


@pytest.mark.asyncio
async def test_prompt_surfaces_token_usage(tmp_path):
    from lovelaice.agent.events import AssistantMessageFinalized

    class _Usage:
        prompt_tokens, completion_tokens, total_tokens = 100, 20, 120

    class _UsageMsg:
        content = "answer"
        usage = _Usage()

    class _Ag:
        def __init__(self):
            self._subs = []

        def subscribe(self, fn):
            self._subs.append(fn)

        async def prompt(self, text):
            for fn in self._subs:
                fn(AssistantMessageFinalized(message=_UsageMsg()))
            from lovelaice.agent.errors import StopReason
            return StopReason.END_TURN

    server = AcpServerV1(agent_factory=lambda **kw: _Ag())
    server.on_connect(_FakeConn())
    new = await server.new_session(cwd=str(tmp_path))
    resp = await server.prompt(
        prompt=[{"type": "text", "text": "hi"}], session_id=new.session_id)
    assert resp.usage is not None
    assert resp.usage.input_tokens == 100
    assert resp.usage.output_tokens == 20
    assert resp.usage.total_tokens == 120


@pytest.mark.asyncio
async def test_prompt_returns_stop_reason(tmp_path):
    server = AcpServerV1(agent_factory=_real_agent_factory(tmp_path))
    server.on_connect(_FakeConn())
    new = await server.new_session(cwd=str(tmp_path))
    resp = await server.prompt(
        prompt=[{"type": "text", "text": "hi"}], session_id=new.session_id)
    assert isinstance(resp, acp.PromptResponse)
    assert resp.stop_reason == "end_turn"
