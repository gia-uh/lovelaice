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
