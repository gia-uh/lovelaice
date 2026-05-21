"""In-process ACP client — for standalone CLI use (no subprocess).

Pair with an `AcpServer` instance directly; bypasses JSON-over-the-wire.
Consumes the server's notification stream via direct callback registration.
"""
import asyncio
from typing import Callable

from lovelaice.acp.server import AcpServer
from lovelaice.acp.protocol import (
    JsonRpcRequest, JsonRpcNotification,
)


class InProcessAcpClient:
    """Drives an AcpServer in the same process — no subprocess, no JSON pipes.

    Notifications flow via direct callback registration on the server.
    Requests dispatch to `server.handle_request` directly.
    """

    def __init__(self, server: AcpServer):
        self._server = server
        self._notification_handlers: list[Callable] = []
        self._server.on_notification(self._on_server_notification)
        self._next_id = 1

    def on_notification(self, fn: Callable[[JsonRpcNotification], None]) -> None:
        """Register a callback that receives outbound notifications from the server."""
        self._notification_handlers.append(fn)

    def _on_server_notification(self, n: JsonRpcNotification) -> None:
        for h in self._notification_handlers:
            h(n)

    async def initialize(self) -> dict:
        return await self._call("initialize", {})

    async def session_new(self, cwd: str) -> str:
        result = await self._call("session/new", {"cwd": cwd})
        return result["sessionId"]

    async def session_prompt(self, session_id: str, text: str) -> dict:
        return await self._call("session/prompt", {
            "sessionId": session_id,
            "prompt": [{"type": "text", "text": text}],
        })

    async def session_cancel(self, session_id: str) -> None:
        await self._server.handle_notification(
            JsonRpcNotification(method="session/cancel",
                                params={"sessionId": session_id}))

    async def _call(self, method: str, params: dict) -> dict:
        req_id = self._next_id
        self._next_id += 1
        resp = await self._server.handle_request(
            JsonRpcRequest(id=req_id, method=method, params=params))
        if resp.error:
            raise RuntimeError(f"ACP error: {resp.error}")
        return resp.result
