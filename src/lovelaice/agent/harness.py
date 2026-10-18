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
        repair_tool_calls: bool = False,
        repair_context: str = "turn",
        budget: Any | None = None,
    ):
        self.llm = llm
        # Stream content tokens as AssistantMessageDelta events. lingo's LLM
        # invokes `_on_token(token)` per content token when set. Guarded for
        # non-lingo / mocked llms in tests.
        try:
            self.llm._on_token = self._emit_token_delta
        except Exception:  # noqa: BLE001
            pass
        self.tools = tools
        self.hooks = hooks
        self.system_prompt = system_prompt
        self.abort = abort or asyncio.Event()
        self._subscribers: list[Callable[[Any], Any]] = []
        # Opt-in focused repair of failed tool-call arguments (see execute_tool).
        self.repair_tool_calls = repair_tool_calls
        self.repair_context = repair_context
        # Set by Agent after the session is built; enables in-session arg-rewrite
        # when a repair succeeds. None on the bare-harness (test) path.
        self.session: Any | None = None
        # The context budget runs before every model call (see model_call).
        # With no window it changes nothing until a provider refuses.
        from lovelaice.agent.budget import BudgetConfig, ContextBudget
        self.budget = budget or ContextBudget(BudgetConfig(window=None))
        self._warned_overhead = False

    def _emit_token_delta(self, token: str) -> None:
        from lovelaice.agent.events import AssistantMessageDelta
        self.emit(AssistantMessageDelta(text=token))

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

    async def model_call(self, session, tools: list | None = None) -> Message:
        """One model call on the session's history, through the context budget.

        Compacts the view when the budget says so, and on a context-length
        refusal compacts harder and retries once. A second refusal raises
        ContextOverflowError; any other provider error propagates unchanged.
        """
        import math
        from lovelaice.agent.compaction import CHARS_PER_TOKEN, fixed_chars, view_chars
        from lovelaice.agent.errors import ContextOverflowError
        from lovelaice.agent.overflow import context_error_tokens, context_error_window

        history = session.messages_for_llm(self.system_prompt)[1:]
        turns = session.turn_numbers()
        current_turn = turns[-1] if turns else 0
        state = session.compaction
        fixed = fixed_chars(self.system_prompt, tools)

        async def prepare(target_tokens=None):
            view, report = await self.budget.prepare(
                system_prompt=self.system_prompt, tools=tools, history=history,
                turns=turns, current_turn=current_turn, state=state,
                summarize=self.summarizer(), target_tokens=target_tokens)
            self._announce(report)
            return view, report

        view, report = await prepare()
        try:
            assistant = await self.llm_call(view, tools=tools)
        except Exception as exc:
            is_context, prompt_tokens = context_error_tokens(exc)
            if not is_context:
                raise
            usable = self.budget.config.usable
            estimate = report.after_tokens if report else math.ceil(
                (fixed + view_chars(view)) / CHARS_PER_TOKEN)
            base = usable or context_error_window(exc) or prompt_tokens or estimate // 2
            view, report = await prepare(target_tokens=int(0.40 * base))
            try:
                assistant = await self.llm_call(view, tools=tools)
            except Exception as exc2:
                again, prompt2 = context_error_tokens(exc2)
                if not again:
                    raise
                fixed_tokens = math.ceil(fixed / CHARS_PER_TOKEN)
                room = usable if usable is not None else context_error_window(exc2)
                raise ContextOverflowError(
                    _overflow_message(fixed_tokens, room), fixed_tokens=fixed_tokens,
                    usable=usable, prompt_tokens=prompt2 or prompt_tokens) from exc2

        usage = getattr(assistant, "usage", None)
        if usage is not None and getattr(usage, "prompt_tokens", 0):
            state.last_prompt_tokens = usage.prompt_tokens
            state.last_view_chars = fixed + view_chars(view)
        session.save_compaction()
        return assistant

    def _announce(self, report) -> None:
        from lovelaice.agent.events import ContextCompacted, ContextOverheadWarning
        if report is None:
            return
        if report.fixed_over and not self._warned_overhead:
            self._warned_overhead = True
            self.emit(ContextOverheadWarning(
                fixed_tokens=report.fixed_tokens, usable=report.usable))
        if report.cut:
            self.emit(ContextCompacted(report=report))

    async def quiet_chat(self, messages: list[Message], **kwargs) -> Message:
        """A model call without tools whose tokens do not stream as deltas.

        The summary call goes through the session's own model; streaming its
        tokens would print the summary in the chat as if it were the answer."""
        saved = {}
        for attr in ("_on_token", "_on_reasoning_token"):
            if hasattr(self.llm, attr):
                saved[attr] = getattr(self.llm, attr)
                setattr(self.llm, attr, None)
        try:
            return await self.llm.chat(messages, tools=None, **kwargs)
        finally:
            for attr, value in saved.items():
                setattr(self.llm, attr, value)

    def summarizer(self):
        """The Summarizer the context budget calls for step 3."""
        from lovelaice.agent.summarize import build_summary_request

        async def summarize(previous, messages, cap):
            reply = await self.quiet_chat(
                build_summary_request(previous, messages, cap), max_tokens=cap)
            text = reply.content if isinstance(reply.content, str) else ""
            return text.strip(), reply.usage

        return summarize

    async def execute_tool(self, call) -> "Any":
        """Run one tool call end-to-end.

        Flow: unknown-tool guard → tool_call hook chain (Block → is_error stub)
        → argument validation (failure → is_error stub) → emit ToolExecutionStart
        → run → emit ToolExecutionEnd → observational tool_result hook → return ToolResult.
        """
        # Late imports to avoid circular import on package load.
        from lovelaice.agent.tools import ToolResult, validate_args
        from lovelaice.agent.hooks import Block
        from lovelaice.agent.events import (
            ToolExecutionStart, ToolExecutionEnd, ToolCallRepaired)

        tool = self.tools.get(call.name)
        if tool is None:
            return ToolResult.from_value(
                f"unknown tool: {call.name}", is_error=True)

        # Permission hook chain.
        decision = await self.hooks.reduce_tool_call(call)
        if isinstance(decision, Block):
            return ToolResult.from_value(decision.reason, is_error=True)

        # Argument validation — with an optional one-shot focused repair.
        validated = validate_args(tool, call.arguments or {})
        if isinstance(validated, str):
            repaired = None
            if self.repair_tool_calls:
                repaired = await self._repair_args(tool, call, validated)
            if repaired is None:
                return ToolResult.from_value(validated, is_error=True)
            # Repair succeeded: rewrite the live call + in-session history so
            # later turns see a well-formed call, and announce the repair.
            original_args = dict(call.arguments or {})
            call.arguments = repaired
            if self.session is not None:
                try:
                    self.session.update_tool_call_args(call.id, repaired)
                except Exception:
                    pass
            self.emit(ToolCallRepaired(
                call_id=call.id, name=call.name, original_args=original_args,
                repaired_args=repaired, error=validated))
            validated = repaired

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

    async def _repair_args(self, tool, call, error_str: str):
        """One focused, forced-JSON shot to repair failed tool-call arguments.

        Builds the tool's pydantic arg model, hands the model a self-contained
        prompt (schema + the failed args + the validation error + a grounding
        slice per ``repair_context``), and forces a JSON object back via
        ``llm.create``. Returns the re-validated args dict, or ``None`` if the
        LLM call raises or the result still fails validation (→ caller falls
        back to the normal is-error path). No loop — a single attempt.
        """
        import json as _json
        from lingo.llm import Message
        from lovelaice.agent.tools import build_arg_model, validate_args

        model_cls = build_arg_model(tool)
        schema = model_cls.model_json_schema()
        instruction = (
            "A tool call you emitted has arguments that failed schema "
            "validation. Return a corrected arguments object that satisfies the "
            "schema. Preserve the caller's evident intent; only fix what is "
            "invalid or missing."
        )
        detail = (
            f"Tool: {tool.name}\n"
            f"Description: {tool.description.strip()}\n\n"
            f"JSON schema of valid arguments:\n{_json.dumps(schema)}\n\n"
            f"Arguments that failed validation:\n"
            f"{_json.dumps(call.arguments or {}, default=str)}\n\n"
            f"Validation error:\n{error_str}"
        )
        messages = [Message.system(instruction)]
        messages.extend(self._repair_grounding())
        messages.append(Message.user(detail))

        try:
            result = await self.llm.create(model_cls, messages)
        except BaseException:
            return None
        validated = validate_args(tool, result.model_dump())
        if isinstance(validated, str):
            return None
        return validated

    def _repair_grounding(self) -> list:
        """Grounding messages for the repair shot, per ``repair_context``.

        Flattened to plain user/assistant text (tool_calls stripped to a note,
        tool/system roles dropped) so the one-shot structured call never carries
        a dangling tool-call/response sequence.
        """
        from lingo.llm import Message

        if self.repair_context == "none" or self.session is None:
            return []
        try:
            msgs = self.session.messages_for_llm(self.system_prompt)
        except Exception:
            return []
        history = msgs[1:] if msgs and msgs[0].role == "system" else list(msgs)
        if self.repair_context == "full":
            picked = history
        else:  # "turn": the emitting assistant message + preceding user message
            picked = history[-2:]

        out: list = []
        for m in picked:
            content = m.content if isinstance(m.content, str) else ""
            if m.role == "user":
                out.append(Message.user(content))
            elif m.role == "assistant":
                if m.tool_calls:
                    names = ", ".join(tc.name for tc in m.tool_calls)
                    content = (content + f" (attempted tool call: {names})").strip()
                out.append(Message.assistant(content))
            # tool/system roles are intentionally skipped
        return out


def _overflow_message(fixed_tokens: int, room: int | None) -> str:
    if room is None:
        return (f"The prompt does not fit this model's context window: the system prompt "
                f"and tools take about {fixed_tokens:,} tokens. Raise the model's context "
                f"size or use fewer tools.")
    return (f"The prompt does not fit this model's context window: the system prompt and "
            f"tools take about {fixed_tokens:,} tokens and the window leaves {room:,} for "
            f"the conversation. Raise the model's context size or use fewer tools.")
