"""Harness — phase machine + LLM call seam + tool dispatch + event bus.

Owned by Agent. AgentLoops use only this surface.

VS1 ships:
- llm_call: wraps lingo.LLM.chat, threads streaming-event subscribers,
  runs before_llm_call reducer chain.
- execute_tool, execute_tools_batch: deferred to B8.
- subscribe, emit: event bus for in-process subscribers (AgentEvent dataclasses).
- abort: asyncio.Event honored by llm_call (deferred) and tool execution.
"""
import asyncio
import inspect
from typing import Any, Callable

from lingo.llm import Message
from lovelaice.agent.hooks import HookRegistry
from lovelaice.agent.tools import ToolRegistry


class Harness:
    """The agentic runtime's central seam.

    Subscribers receive in-process AgentEvent instances (sync or async).
    Loops call `llm_call` / `execute_tool` / `execute_tools_batch` to
    orchestrate the turn.
    """

    def __init__(
        self,
        *,
        llm: Any,  # lingo.LLM (typed loosely to avoid Optional/circular issues)
        tools: ToolRegistry,
        hooks: HookRegistry,
        system_prompt: str,
        abort: asyncio.Event | None = None,
    ):
        self.llm = llm
        self.tools = tools
        self.hooks = hooks
        self.system_prompt = system_prompt
        self.abort = abort or asyncio.Event()
        self._subscribers: list[Callable[[Any], Any]] = []

    def subscribe(self, fn: Callable[[Any], Any]) -> None:
        """Register an event subscriber. Sync or async callables both work."""
        self._subscribers.append(fn)

    def emit(self, ev: Any) -> None:
        """Publish an event to all subscribers.

        Sync subscribers run inline; async subscribers are scheduled as
        tasks (fire-and-forget — they must not block the agent's turn).
        """
        for fn in self._subscribers:
            res = fn(ev)
            if inspect.isawaitable(res):
                asyncio.create_task(res)

    async def llm_call(
        self,
        messages: list[Message],
        tools: list | None = None,
    ) -> Message:
        """One LLM call via lingo.

        Runs the `before_llm_call` reducer chain to allow hooks to rewrite
        messages and tools, then delegates to `lingo.LLM.chat`.
        """
        for fn in self.hooks._hooks.get("before_llm_call", []):
            res = fn(messages, tools)
            if inspect.isawaitable(res):
                res = await res
            if res is not None:
                messages, tools = res
        return await self.llm.chat(messages, tools=tools)

    async def execute_tool(self, call) -> "Any":
        """Run one tool call end-to-end.

        Flow: unknown-tool guard → tool_call hook chain (Block → is_error stub)
        → argument validation (failure → is_error stub) → emit ToolExecutionStart
        → run → emit ToolExecutionEnd → observational tool_result hook → return ToolResult.
        """
        # Late imports to avoid circular import on package load.
        from lovelaice.agent.tools import ToolResult, validate_args
        from lovelaice.agent.hooks import Block
        from lovelaice.agent.events import ToolExecutionStart, ToolExecutionEnd

        tool = self.tools.get(call.name)
        if tool is None:
            return ToolResult.from_value(
                f"unknown tool: {call.name}", is_error=True)

        # Permission hook chain.
        decision = await self.hooks.reduce_tool_call(call)
        if isinstance(decision, Block):
            return ToolResult.from_value(decision.reason, is_error=True)

        # Argument validation.
        validated = validate_args(tool, call.arguments or {})
        if isinstance(validated, str):
            return ToolResult.from_value(validated, is_error=True)

        self.emit(ToolExecutionStart(
            call_id=call.id, name=call.name, args=validated))

        try:
            raw = await tool.inner.run(**validated)
            result = ToolResult.from_value(raw, raw_output=raw)
        except BaseException as exc:
            result = ToolResult.from_exception(exc)

        self.emit(ToolExecutionEnd(
            call_id=call.id, result=result, is_error=result.is_error))

        # Observational hook.
        await self.hooks.emit("tool_result", call, result)
        return result

    async def execute_tools_batch(self, calls: list) -> list:
        """Dispatch a batch of tool calls.

        - Empty list → empty list.
        - If any tool in the batch is sequential, runs the whole batch in
          source order serially.
        - Otherwise runs concurrently and returns results in source order.
        """
        if not calls:
            return []
        names = [c.name for c in calls]
        if self.tools.any_sequential(names):
            out = []
            for c in calls:
                out.append(await self.execute_tool(c))
            return out
        # Parallel — gather preserves task-creation order.
        tasks = [asyncio.create_task(self.execute_tool(c)) for c in calls]
        return await asyncio.gather(*tasks)
