"""Minimal ACP server.

Per spec §5.2 VS1 row — implements: initialize, session/new, session/prompt,
session/cancel. Notifications: session/update with agent_message_chunk,
tool_call, tool_call_update. Stop reasons: end_turn, cancelled.

This module owns the translation from in-process AgentEvent dataclasses
to ACP session/update notifications. The `run_stdio` method wires the
server to stdin/stdout for subprocess use.
"""
import asyncio
import json
import sys
import uuid
from typing import Any, Callable

from lovelaice.acp.protocol import (
    JsonRpcRequest, JsonRpcResponse, JsonRpcNotification,
    parse_message, encode_message,
)
from lovelaice.agent.events import (
    AssistantMessageFinalized, ToolExecutionStart, ToolExecutionEnd,
)


ACP_PROTOCOL_VERSION = "0.1"


def _render_display_messages(message_entries) -> list[dict]:
    """Render persisted MessageEntry rows into the wire-friendly DisplayMessage
    shape — collapses tool_calls inline with the assistant message, drops
    thinking, returns plain text per role."""
    out: list[dict] = []
    for m in message_entries:
        text = m.content if isinstance(m.content, str) else str(m.content)
        entry: dict = {"role": m.role, "text": text}
        if m.tool_calls:
            entry["tool_calls"] = m.tool_calls
        out.append(entry)
    return out


class AcpServer:
    """ACP-compliant agent server.

    `agent_factory()` returns a fresh `Agent` instance per `session/new`.
    The server owns the mapping from `sessionId` → Agent, the wire-level
    translation of AgentEvents to session/update notifications, and the
    JSON-RPC request/notification routing.
    """

    def __init__(self, *, agent_factory: Callable, conversation_store=None):
        self._agent_factory = agent_factory
        self._store = conversation_store
        self._sessions: dict[str, Any] = {}
        self._session_to_conversation: dict[str, str] = {}
        self._initialized = False
        self._notification_handlers: list[Callable[[JsonRpcNotification], Any]] = []
        self._current_prompt_task: asyncio.Task | None = None

    def on_notification(self, fn: Callable[[JsonRpcNotification], Any]) -> None:
        """Register a handler for outbound notifications."""
        self._notification_handlers.append(fn)

    def _notify(self, method: str, params: dict) -> None:
        n = JsonRpcNotification(method=method, params=params)
        for h in self._notification_handlers:
            h(n)

    def _agent_event_to_notification(self, sid: str, ev) -> None:
        """Translate an in-process AgentEvent → ACP session/update notification."""
        if isinstance(ev, AssistantMessageFinalized):
            text = ev.message.content if isinstance(ev.message.content, str) else ""
            if text:
                self._notify("session/update", {
                    "sessionId": sid,
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": text},
                })
        elif isinstance(ev, ToolExecutionStart):
            self._notify("session/update", {
                "sessionId": sid,
                "sessionUpdate": "tool_call",
                "toolCallId": ev.call_id,
                "title": ev.name,
                "kind": "other",
                "status": "pending",
                "rawInput": ev.args,
            })
        elif isinstance(ev, ToolExecutionEnd):
            text_content = ""
            if ev.result.content and isinstance(ev.result.content[0], dict):
                text_content = ev.result.content[0].get("text", "")
            self._notify("session/update", {
                "sessionId": sid,
                "sessionUpdate": "tool_call_update",
                "toolCallId": ev.call_id,
                "status": "failed" if ev.is_error else "completed",
                "content": [{"type": "text", "text": text_content}],
            })

    async def handle_request(self, req: JsonRpcRequest) -> JsonRpcResponse:
        """Dispatch one JSON-RPC request to the right handler."""
        try:
            if req.method == "initialize":
                self._initialized = True
                return JsonRpcResponse(id=req.id, result={
                    "protocolVersion": ACP_PROTOCOL_VERSION,
                    "agentCapabilities": {
                        "loadSession": False,
                        "promptCapabilities": {
                            "image": False,
                            "audio": False,
                            "embeddedContext": False,
                        },
                    },
                })

            if not self._initialized:
                return JsonRpcResponse(id=req.id, error={
                    "code": -32002, "message": "agent not initialized"})

            if req.method == "session/new":
                params = req.params or {}
                given_cid = params.get("conversationId")
                conv = None
                messages: list[dict] = []
                if self._store is not None:
                    if given_cid is None:
                        conv = await self._store.create(
                            model="unknown",
                            system_prompt_hash="sha256:unknown",
                        )
                    else:
                        conv = await self._store.get(given_cid)
                        if conv is None:
                            return JsonRpcResponse(id=req.id, error={
                                "code": -32602,
                                "message": f"unknown conversation: {given_cid}",
                            })
                        messages = _render_display_messages(conv.row.messages)
                sid = uuid.uuid4().hex[:8]
                if conv is not None:
                    try:
                        agent = self._agent_factory(conversation=conv)
                    except TypeError:
                        agent = self._agent_factory()
                    self._session_to_conversation[sid] = conv.id
                else:
                    agent = self._agent_factory()
                agent.subscribe(
                    lambda ev, sid=sid: self._agent_event_to_notification(sid, ev))
                self._sessions[sid] = agent
                result: dict[str, Any] = {"sessionId": sid}
                if conv is not None:
                    result["conversationId"] = conv.id
                    result["messages"] = messages
                return JsonRpcResponse(id=req.id, result=result)

            if req.method == "session/prompt":
                params = req.params or {}
                sid = params.get("sessionId")
                agent = self._sessions.get(sid)
                if agent is None:
                    return JsonRpcResponse(id=req.id, error={
                        "code": -32602, "message": f"unknown sessionId: {sid}"})
                # Extract text from prompt content blocks.
                blocks = params.get("prompt", [])
                text = "".join(b.get("text", "") for b in blocks
                               if b.get("type") == "text")
                self._current_prompt_task = asyncio.create_task(agent.prompt(text))
                try:
                    stop = await self._current_prompt_task
                except asyncio.CancelledError:
                    stop = "cancelled"
                self._current_prompt_task = None
                stop_value = stop.value if hasattr(stop, "value") else str(stop)
                return JsonRpcResponse(id=req.id, result={"stopReason": stop_value})

            if req.method == "workflow/run":
                params = req.params or {}
                return JsonRpcResponse(
                    id=req.id, result=await self._handle_workflow_run(params))

            return JsonRpcResponse(id=req.id, error={
                "code": -32601, "message": f"method not found: {req.method}"})
        except Exception as e:
            return JsonRpcResponse(id=req.id, error={
                "code": -32603,
                "message": f"internal error: {type(e).__name__}: {e}",
            })

    async def _handle_workflow_run(self, params: dict) -> dict:
        """Run a native lovelaice workflow spec, returning ``{"result": <dict>}``.

        Each ``agent`` node gets a fresh agent from the same factory the ACP
        server was built with, so tools/MCP wired by the host are available.
        """
        from lovelaice.workflows import WorkflowSpec, run as _run

        spec = WorkflowSpec.model_validate(params["spec"])
        result = await _run(
            spec,
            agent_factory=lambda: self._agent_factory(),
            inputs=params.get("inputs"),
        )
        return {"result": result}

    async def handle_notification(self, n: JsonRpcNotification) -> None:
        """Handle inbound notifications (session/cancel, conversation/archive)."""
        if n.method == "session/cancel":
            if self._current_prompt_task and not self._current_prompt_task.done():
                self._current_prompt_task.cancel()
            return
        if n.method == "conversation/archive":
            if self._store is None:
                return
            cid = (n.params or {}).get("conversationId")
            if cid:
                await self._store.archive(cid)
            return

    async def run_stdio(self) -> None:
        """Read JSON-RPC lines from stdin; write responses + notifications to stdout."""
        def write_notification(n: JsonRpcNotification):
            sys.stdout.write(encode_message(n) + "\n")
            sys.stdout.flush()
        self.on_notification(write_notification)

        loop = asyncio.get_event_loop()
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await loop.connect_read_pipe(lambda: protocol, sys.stdin)

        while True:
            line = await reader.readline()
            if not line:
                break
            try:
                obj = json.loads(line.decode("utf-8"))
                msg = parse_message(obj)
            except (ValueError, json.JSONDecodeError):
                continue
            if isinstance(msg, JsonRpcRequest):
                resp = await self.handle_request(msg)
                sys.stdout.write(encode_message(resp) + "\n")
                sys.stdout.flush()
            elif isinstance(msg, JsonRpcNotification):
                await self.handle_notification(msg)
            # Responses going inbound are ignored — we're the agent, not the client.
