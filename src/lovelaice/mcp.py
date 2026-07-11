"""MCP support: spawn stdio MCP servers and register their tools on the agent.

This is a thin wrapper around the official `mcp` Python SDK. Each tool
exposed by an MCP server becomes a `lingo.tools.Tool` with display name
`mcp:<server>:<tool>` and a JSON-schema-derived parameter map.

Lifecycle: servers spawn at `Config.build()` time and inherit the parent
process lifetime. v1 does not clean them up on exit — the OS reaps the
subprocesses. A future `agent.close()` could do graceful teardown.
"""
from __future__ import annotations

import asyncio
import json
import threading
from contextlib import asynccontextmanager
from typing import Any

from lingo.tools import Tool

try:
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client
except ImportError:  # pragma: no cover
    ClientSession = None  # type: ignore[assignment]
    stdio_client = None  # type: ignore[assignment]
    StdioServerParameters = None  # type: ignore[assignment]

try:
    from mcp.client.streamable_http import streamablehttp_client
except ImportError:  # pragma: no cover
    streamablehttp_client = None  # type: ignore[assignment]


class MCPTransportError(RuntimeError):
    """Raised when the HTTP transport encounters an unrecoverable error."""


@asynccontextmanager
async def connect(config: dict):
    """Connect to an MCP server. Dispatches on transport:
        - {"url": ...}      -> HTTP MCP (streamable_http)
        - {"command": ...}  -> stdio MCP
    Yields an initialized ClientSession.
    """
    if "url" in config:
        async with _http_session(config) as session:
            yield session
    elif "command" in config:
        async with _stdio_session(config) as session:
            yield session
    else:
        raise ValueError(f"unrecognized MCP config: {config!r}")


@asynccontextmanager
async def _http_session(config: dict):
    url = config["url"]
    headers: dict[str, str] = {}
    if (auth := config.get("auth")) and (bearer := auth.get("bearer")):
        headers["Authorization"] = f"Bearer {bearer}"
    # Per-server extra headers — used by warden to pass scoping keys like
    # X-Peacock-Conversation. Plain dict merge; "auth.bearer" wins for
    # Authorization since it ran first.
    extra = config.get("headers") or {}
    for k, v in extra.items():
        headers.setdefault(k, v)
    if streamablehttp_client is None:
        raise RuntimeError("mcp.client.streamable_http not available")
    try:
        async with streamablehttp_client(url, headers=headers) as (
            read_stream,
            write_stream,
            _get_session_id,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                yield session
    except Exception as exc:
        raise MCPTransportError(
            f"HTTP MCP transport error for {url}: {exc}"
        ) from exc


@asynccontextmanager
async def _stdio_session(config: dict):
    if ClientSession is None or stdio_client is None:
        raise RuntimeError("mcp Python SDK not installed")
    params = StdioServerParameters(
        command=config["command"],
        args=config.get("args", []),
        env=config.get("env"),
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


_PYTHON_TYPE_FROM_JSON: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _mcp_display_name(server: str, tool: str) -> str:
    return f"mcp:{server}:{tool}"


def _params_from_input_schema(schema: dict[str, Any]) -> dict[str, type]:
    """Pluck a `dict[name, python_type]` from a JSON Schema input descriptor."""
    props = (schema or {}).get("properties", {}) or {}
    out: dict[str, type] = {}
    for name, descriptor in props.items():
        json_type = (descriptor or {}).get("type", "string")
        out[name] = _PYTHON_TYPE_FROM_JSON.get(json_type, str)
    return out


class _MCPTool(Tool):
    """A `lingo.Tool` that proxies to an MCP `session.call_tool` invocation."""

    def __init__(
        self,
        *,
        display_name: str,
        description: str,
        params: dict[str, type],
        session: Any,
        tool_name: str,
        json_schema: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(display_name, description)
        self._params = params
        self._session = session
        self._tool_name = tool_name
        # Carry the FastMCP inputSchema verbatim so lingo's schema builder
        # serializes it without flattening. Set explicitly (not via the base
        # class) so this works regardless of the installed lingo version.
        self.json_schema = json_schema

    def parameters(self) -> dict[str, type]:
        return self._params

    async def run(self, **kwargs: Any) -> Any:
        result = await self._session.call_tool(self._tool_name, kwargs)
        content = getattr(result, "content", None) or []
        parts = []
        for part in content:
            text = getattr(part, "text", None)
            if text is not None:
                parts.append(text)
            else:
                dump = getattr(part, "model_dump", None)
                parts.append(json.dumps(dump() if dump else str(part), default=str))
        return "\n".join(parts)


def _wrap_mcp_tool(*, server_name: str, tool: Any, session: Any) -> _MCPTool:
    """Wrap one MCP tool definition into a `lingo.Tool`."""
    input_schema = getattr(tool, "inputSchema", None) or None
    return _MCPTool(
        display_name=_mcp_display_name(server_name, tool.name),
        description=getattr(tool, "description", "") or "MCP tool",
        params=_params_from_input_schema(input_schema or {}),
        session=session,
        tool_name=tool.name,
        json_schema=input_schema,
    )


def register_mcp_tools(agent: Any, specs: list[dict[str, Any]]) -> None:
    """
    For each spec, spawn the MCP server, fetch its tools, and register
    each on `agent.tools`. Failures on individual servers log + skip.
    """
    for spec in specs:
        try:
            session = _start_session_in_background(spec)
            tools = _list_tools_blocking(session)
        except Exception as exc:
            print(f"[mcp] {spec.get('name', '<unnamed>')}: failed to start ({exc!r})")
            continue
        for t in tools:
            wrapped = _wrap_mcp_tool(server_name=spec["name"], tool=t, session=session)
            agent.tools.append(wrapped)


# --- Managed background session (HTTP + stdio, with teardown) ---------------


class ManagedMcpSession:
    """An MCP ClientSession kept alive on a dedicated background loop/thread,
    with explicit teardown. Supports HTTP (``{url}``) and stdio
    (``{command}``) transports. ``call_tool`` marshals onto the background
    loop; ``aclose`` signals the loop to unwind the transport and joins the
    thread."""

    def __init__(self, loop, session, thread, stop_event, tools):
        self._loop = loop
        self._session = session
        self._thread = thread
        self._stop = stop_event
        self.tools = tools

    async def call_tool(self, name: str, kwargs: dict):
        fut = asyncio.run_coroutine_threadsafe(
            self._session.call_tool(name, kwargs), self._loop)
        return await asyncio.wrap_future(fut)

    async def aclose(self) -> None:
        self._loop.call_soon_threadsafe(self._stop.set)
        self._thread.join(timeout=5.0)


def build_agent_tools(specs: list[dict]):
    """Start a managed session per spec and wrap each exposed MCP tool as an
    ``AgentTool``. Returns ``(tools, sessions)``: the tools go onto the agent,
    the sessions are retained by the caller for teardown. A spec that fails to
    start is logged and skipped (no tools, no session)."""
    from lovelaice.agent.tools import AgentTool

    tools: list = []
    sessions: list[ManagedMcpSession] = []
    for spec in specs:
        try:
            sess = start_managed_session(spec)
        except Exception as exc:  # noqa: BLE001
            print(f"[mcp] {spec.get('name', '<unnamed>')}: "
                  f"failed to start ({exc!r})")
            continue
        sessions.append(sess)
        for t in sess.tools:
            wrapped = _wrap_mcp_tool(
                server_name=spec.get("name", "mcp"), tool=t, session=sess)
            tools.append(AgentTool(inner=wrapped, kind="other"))
    return tools, sessions


def start_managed_session(spec: dict) -> "ManagedMcpSession":
    """Start an MCP server (HTTP or stdio) on a background loop/thread, init
    the ClientSession, list its tools, and park until closed. Reuses the
    module's ``_http_session`` / ``_stdio_session`` context managers."""
    if ClientSession is None:
        raise RuntimeError("mcp Python SDK not installed")

    ready = threading.Event()
    holder: dict[str, Any] = {}

    def _runner() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        stop = asyncio.Event()

        async def _go() -> None:
            if "url" in spec:
                cm = _http_session(spec)
            elif "command" in spec:
                cm = _stdio_session(spec)
            else:
                raise ValueError(f"unrecognized MCP config: {spec!r}")
            async with cm as session:
                tools = (await session.list_tools()).tools
                holder.update(loop=loop, session=session, stop=stop,
                              tools=list(tools))
                ready.set()
                await stop.wait()

        try:
            loop.run_until_complete(_go())
        except BaseException as e:  # noqa: BLE001
            holder["error"] = e
            ready.set()
        finally:
            loop.close()

    t = threading.Thread(target=_runner, daemon=True,
                         name=f"mcp-{spec.get('name', '?')}")
    t.start()
    ready.wait(timeout=20.0)
    if "error" in holder:
        raise holder["error"]
    if "session" not in holder:
        raise RuntimeError(
            f"MCP server {spec.get('name')!r} did not initialize within 20s")
    return ManagedMcpSession(holder["loop"], holder["session"], t,
                             holder["stop"], holder["tools"])


# --- Cross-thread session machinery ----------------------------------------


class _BackgroundSession:
    """Holds an MCP ClientSession alive on a dedicated background loop."""

    def __init__(self, loop: asyncio.AbstractEventLoop, session: Any) -> None:
        self.loop = loop
        self.session = session

    async def call_tool(self, name: str, kwargs: dict[str, Any]) -> Any:
        fut = asyncio.run_coroutine_threadsafe(
            self.session.call_tool(name, kwargs),
            self.loop,
        )
        return await asyncio.wrap_future(fut)


def _start_session_in_background(spec: dict[str, Any]) -> _BackgroundSession:
    """Spawn the MCP server, init the session, and keep it alive on a thread."""
    if ClientSession is None or stdio_client is None:
        raise RuntimeError("mcp Python SDK not installed")

    ready = threading.Event()
    holder: dict[str, Any] = {}

    def _runner() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def _init_and_park() -> None:
            params = StdioServerParameters(
                command=spec["command"],
                args=spec.get("args", []),
                env=spec.get("env"),
            )
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    holder["session"] = session
                    holder["loop"] = loop
                    ready.set()
                    await asyncio.Event().wait()

        try:
            loop.run_until_complete(_init_and_park())
        except Exception as e:
            holder["error"] = e
            ready.set()

    threading.Thread(target=_runner, daemon=True).start()
    ready.wait(timeout=15.0)
    if "error" in holder:
        raise holder["error"]
    return _BackgroundSession(loop=holder["loop"], session=holder["session"])


def _list_tools_blocking(bg: _BackgroundSession) -> list[Any]:
    """Synchronously fetch the tool list from a backgrounded session."""
    fut = asyncio.run_coroutine_threadsafe(bg.session.list_tools(), bg.loop)
    return fut.result(timeout=15.0).tools
