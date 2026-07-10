"""ACP v1 server on the official agent-client-protocol SDK.

Clean-room replacement for the legacy hand-rolled ``lovelaice.acp.server``
("0.1" flat dialect), which stays frozen for warden. Implements the
``acp.Agent`` interface; served over stdio via ``acp.run_agent`` (see
``__main__``). Symmetric with aegis's ``acp.Client`` on the driving side.
"""
from __future__ import annotations

import asyncio
import uuid
from typing import Any, Callable

import acp
from acp.schema import AgentCapabilities, PromptCapabilities

from lovelaice.agent.events import (
    AssistantMessageFinalized,
    ToolExecutionEnd,
    ToolExecutionStart,
)


class AcpServerV1(acp.Agent):
    """ACP-v1 agent. ``agent_factory(conversation=None)`` builds a lovelaice
    Agent per session — same constructor contract as the legacy AcpServer so
    hosts that wire their own tools (warden) migrate by import + dialect only.
    """

    def __init__(self, *, agent_factory: Callable[..., Any],
                 conversation_store: Any = None) -> None:
        self._agent_factory = agent_factory
        self._store = conversation_store
        self._conn: acp.Client | None = None
        self._sessions: dict[str, Any] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._inflight: asyncio.Task | None = None

    def on_connect(self, conn: acp.Client) -> None:
        self._conn = conn

    async def initialize(self, protocol_version: int,
                         client_capabilities=None, client_info=None,
                         **kw: Any) -> acp.InitializeResponse:
        return acp.InitializeResponse(
            protocol_version=acp.PROTOCOL_VERSION,
            agent_capabilities=AgentCapabilities(
                load_session=False,
                prompt_capabilities=PromptCapabilities(
                    image=False, audio=False, embedded_context=False,
                ),
            ),
        )

    async def new_session(self, cwd: str, additional_directories=None,
                          mcp_servers=None, **kw: Any) -> acp.NewSessionResponse:
        agent = self._agent_factory()
        sid = uuid.uuid4().hex[:16]
        agent.subscribe(lambda ev, _sid=sid: self._emit(_sid, ev))
        self._sessions[sid] = agent
        # mcp_servers accepted; per-session attach is a later slice (VS2).
        return acp.NewSessionResponse(session_id=sid)

    async def prompt(self, prompt, session_id: str, message_id=None,
                     **kw: Any) -> acp.PromptResponse:
        agent = self._sessions.get(session_id)
        if agent is None:
            raise acp.RequestError(
                code=-32602, message=f"unknown sessionId: {session_id}")
        self._loop = asyncio.get_running_loop()
        text = "".join(b.get("text", "") for b in prompt
                       if isinstance(b, dict) and b.get("type") == "text")
        task = asyncio.ensure_future(agent.prompt(text))
        self._inflight = task
        try:
            stop = await task
        except asyncio.CancelledError:
            return acp.PromptResponse(stop_reason="cancelled")
        finally:
            self._inflight = None
        value = getattr(stop, "value", None) or str(stop)
        return acp.PromptResponse(stop_reason=value)

    async def cancel(self, session_id: str, **kw: Any) -> None:
        task = self._inflight
        if task is not None and not task.done():
            task.cancel()

    # -- event translation --------------------------------------------------

    def _emit(self, session_id: str, ev: Any) -> None:
        update = self._translate(ev)
        if update is None or self._conn is None:
            return
        loop = self._loop or asyncio.get_event_loop()
        conn = self._conn
        loop.call_soon_threadsafe(
            lambda: asyncio.ensure_future(
                conn.session_update(session_id=session_id, update=update),
                loop=loop,
            )
        )

    def _translate(self, ev: Any):
        if isinstance(ev, AssistantMessageFinalized):
            text = ev.message.content if isinstance(ev.message.content, str) else ""
            return acp.update_agent_message_text(text) if text else None
        if isinstance(ev, ToolExecutionStart):
            return acp.start_tool_call(
                tool_call_id=ev.call_id, title=ev.name,
                kind="other", raw_input=ev.args)
        if isinstance(ev, ToolExecutionEnd):
            status = "failed" if ev.is_error else "completed"
            text = ""
            content = getattr(ev.result, "content", None) or []
            if content and isinstance(content[0], dict):
                text = content[0].get("text", "")
            return acp.update_tool_call(
                tool_call_id=ev.call_id, status=status,
                content=[acp.tool_content(acp.text_block(text))] if text else None)
        return None
