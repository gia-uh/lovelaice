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
        self._provider_window: int | None = None
        self._provider_input_limit: int | None = None
        self._summary_turn: int | None = None
        self._summary_requests = 0

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
        messages, tools = await self._apply_llm_hooks(messages, tools)
        return await self.llm.chat(messages, tools=tools)

    async def _apply_llm_hooks(self, messages, tools):
        for fn in self.hooks._hooks.get("before_llm_call", []):
            res = fn(messages, tools)
            if inspect.isawaitable(res):
                res = await res
            if res is not None:
                messages, tools = res
        return messages, tools

    async def model_call(self, session, tools: list | None = None) -> Message:
        """One model call on the session's history, through the context budget.

        Compacts the view when the budget says so, and on a context-length
        refusal compacts harder and retries once. A second refusal raises
        ContextOverflowError; any other provider error propagates unchanged.
        """
        import math
        from lovelaice.agent.compaction import (
            CHARS_PER_TOKEN, CompactionState, estimate_tokens, fixed_chars, message_chars,
        )
        from lovelaice.agent.events import ContextCompacting, ContextMeasured
        from lovelaice.agent.errors import ContextOverflowError
        from lovelaice.agent.overflow import context_error_tokens

        history = session.messages_for_llm(self.system_prompt)[1:]
        turns = session.turn_numbers()
        current_turn = turns[-1] if turns else 0
        config = self.budget.config
        saved = session.compaction
        policy_key = config.policy_key()
        if saved.policy_key == policy_key and saved.history_length <= len(history):
            state = saved.model_copy(deep=True)
        else:
            state = CompactionState(policy_key=policy_key)
        fixed_tokens = math.ceil(fixed_chars(self.system_prompt, tools) / CHARS_PER_TOKEN)
        summarize = self.summarizer(current_turn)
        attempted = []

        async def prepare(target_tokens=None, retry_limit=None):
            window = self._context_window()
            input_limit = self._provider_input_limit
            if retry_limit is not None:
                input_limit = retry_limit if input_limit is None else min(input_limit, retry_limit)
            view, report = await self.budget.prepare(
                system_prompt=self.system_prompt, tools=tools, history=history,
                turns=turns, current_turn=current_turn, state=state, summarize=summarize,
                window=window, input_limit=input_limit,
                target_tokens=target_tokens, on_compacting=lambda tokens: self.emit(
                    ContextCompacting(prompt_tokens=tokens, window=window)))
            if report is not None:
                attempted.extend(report.attempted)
            # Reducers can change the serialized request. Measure and compare
            # the actual requests, so a reducer cannot accidentally resend one.
            if self.hooks._hooks.get("before_llm_call"):
                view = [message.model_copy(deep=True) for message in view]
            messages, call_tools = await self._apply_llm_hooks(view, tools)
            chars = fixed_chars("", call_tools) + sum(message_chars(m) for m in messages)
            tokens = estimate_tokens(chars, state)
            return messages, call_tools, chars, tokens, report

        def overflow(prompt_tokens, *, estimated):
            window = self._context_window()
            usable = config.usable_for(window, self._provider_input_limit)
            return ContextOverflowError(
                _overflow_message(fixed_tokens, usable), fixed_tokens=fixed_tokens,
                usable=usable, prompt_tokens=prompt_tokens, window=window, estimated=estimated,
                attempted=tuple(dict.fromkeys(attempted)))

        messages, call_tools, chars, tokens, report = await prepare()
        self._announce(report)
        self.emit(ContextMeasured(prompt_tokens=tokens, window=self._context_window(),
                                  estimated=True))
        try:
            assistant = await self.llm.chat(messages, tools=call_tools)
        except Exception as exc:
            is_context, prompt_tokens = context_error_tokens(exc)
            if not is_context:
                raise
            reported_limit = self._observe_context_error(exc)
            if prompt_tokens is not None:
                state.last_prompt_tokens = prompt_tokens
                state.last_view_chars = chars
            usable = config.usable_for(self._context_window(), self._provider_input_limit)
            rejected = prompt_tokens if prompt_tokens is not None else tokens
            retry_limit = None
            if reported_limit and usable is not None:
                target = min(config.target_for(fixed_tokens, usable), rejected - 1)
            else:
                retry_limit = rejected * 3 // 4
                target = retry_limit
                if usable is not None:
                    target = min(target, config.target_for(fixed_tokens, usable))
            retry, retry_tools, smaller_chars, smaller_tokens, report = await prepare(
                target, retry_limit)
            smaller = smaller_chars < chars and smaller_tokens < rejected
            self._announce(report, sent=smaller)
            if not smaller:
                raise overflow(prompt_tokens if prompt_tokens is not None else tokens,
                               estimated=prompt_tokens is None) from exc
            chars = smaller_chars
            self.emit(ContextMeasured(prompt_tokens=smaller_tokens, window=self._context_window(),
                                      estimated=True))
            try:
                assistant = await self.llm.chat(retry, tools=retry_tools)
            except Exception as second:
                again, prompt2 = context_error_tokens(second)
                if not again:
                    raise
                self._observe_context_error(second)
                raise overflow(prompt2 if prompt2 is not None else smaller_tokens,
                               estimated=prompt2 is None) from second

        usage = getattr(assistant, "usage", None)
        if usage is not None and getattr(usage, "prompt_tokens", 0):
            state.last_prompt_tokens = usage.prompt_tokens
            state.last_view_chars = chars
            self.emit(ContextMeasured(prompt_tokens=usage.prompt_tokens,
                                      window=self._context_window(), estimated=False))
        state.history_length = len(history)
        session.compaction = state
        persistence = session.save_compaction()
        if inspect.isawaitable(persistence):
            await persistence
        return assistant

    def _context_window(self) -> int | None:
        configured = self.budget.config.window
        if configured is not None and self._provider_window is not None:
            return min(configured, self._provider_window)
        return configured or self._provider_window

    def _observe_context_error(self, exc: BaseException) -> bool:
        from lovelaice.agent.overflow import context_error_input_limit, context_error_window

        window = context_error_window(exc)
        if window:
            self._provider_window = min(self._provider_window or window, window)
        limit = context_error_input_limit(exc)
        if limit:
            self._provider_input_limit = min(self._provider_input_limit or limit, limit)
        return bool(window or limit)

    def _announce(self, report, *, sent: bool = True) -> None:
        from lovelaice.agent.events import (
            ContextCompacted, ContextOverheadWarning, ContextSummaryUsage,
        )
        if report is None:
            return
        if report.fixed_over and not self._warned_overhead:
            self._warned_overhead = True
            self.emit(ContextOverheadWarning(
                fixed_tokens=report.fixed_tokens, usable=report.usable))
        if report.cut and sent:
            self.emit(ContextCompacted(report=report))
        elif report.summary_usage is not None:
            self.emit(ContextSummaryUsage(usage=report.summary_usage))

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

    def summarizer(self, turn: int | None = None):
        """The quiet summary call, sharing an RPC limit across one user turn."""
        from lovelaice.agent.overflow import context_error_tokens
        from lovelaice.agent.summarize import MAX_SUMMARY_CALLS, build_summary_request

        if turn is None or self._summary_turn != turn:
            self._summary_turn = turn
            self._summary_requests = 0

        async def summarize(previous, messages, cap):
            request = build_summary_request(previous, messages, cap)
            if self._summary_requests >= MAX_SUMMARY_CALLS:
                return "", None
            self._summary_requests += 1
            try:
                reply = await self.quiet_chat(request, max_tokens=cap, extra_body=_NO_REASONING)
            except Exception as exc:
                if (getattr(exc, "status_code", None) != 400 or context_error_tokens(exc)[0]
                        or self._summary_requests >= MAX_SUMMARY_CALLS):
                    raise
                self._summary_requests += 1
                reply = await self.quiet_chat(request, max_tokens=cap)
            text = reply.content if isinstance(reply.content, str) else ""
            if reply.stop_reason in ("length", "max_tokens"):
                text = ""
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


_NO_REASONING = {
    "reasoning": {"enabled": False},
    "chat_template_kwargs": {"enable_thinking": False},
}


def _overflow_message(fixed_tokens: int, usable: int | None) -> str:
    overhead = f"The system prompt and tool schemas take about {fixed_tokens:,} tokens."
    budget = f" The usable prompt budget is {usable:,} tokens." if usable is not None else ""
    return (f"The prompt does not fit the model's context window. {overhead}{budget} "
            "The selected compaction operations could not make enough room for "
            "the protected messages and tool results.")
