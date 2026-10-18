"""The context budget: decides when the view is compacted and how far.

Before each model call the budget estimates the prompt. Over the trigger, it
cuts the view down to the target in four steps, cheapest first:

1. stub tool results from finished turns, oldest first;
2. stub older tool results in the current turn, keeping the latest batch;
3. fold old turns into one rolling summary (needs a model call);
4. clip the single largest message.

Every cut is recorded in the `CompactionState`; the stored history is never
touched.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Awaitable, Callable

from lingo.llm import Message, Usage

from lovelaice.agent.compaction import (
    CHARS_PER_TOKEN, CLIP_MARKER, CompactionState, _content_text, estimate_tokens,
    fixed_chars, render_view, stub_text, view_chars,
)

# (previous summary, messages to fold, cap in tokens) -> (summary text, usage)
Summarizer = Callable[[str | None, list[Message], int], Awaitable[tuple[str, Usage | None]]]

# Step 4 never clips a message down below this many characters.
MIN_CLIP_KEEP_CHARS = 400


@dataclass
class BudgetConfig:
    window: int | None
    max_tokens: int | None = None
    trigger: float = 0.75
    target: float = 0.5

    @property
    def reserve(self) -> int:
        if self.max_tokens:
            return self.max_tokens
        return min(2048, (self.window or 0) // 4)

    @property
    def usable(self) -> int | None:
        if self.window is None:
            return None
        return self.window - self.reserve


@dataclass
class CompactionReport:
    before_tokens: int
    after_tokens: int
    usable: int | None
    fixed_tokens: int
    stubbed_finished: int = 0
    stubbed_current: int = 0
    turns_summarized: int = 0
    clipped: int = 0
    summary: str | None = None
    summary_usage: Usage | None = None
    fixed_over: bool = False

    @property
    def cut(self) -> bool:
        return bool(self.stubbed_finished or self.stubbed_current
                    or self.turns_summarized or self.clipped)


class ContextBudget:
    def __init__(self, config: BudgetConfig):
        self.config = config

    async def prepare(
        self,
        *,
        system_prompt: str,
        tools: list | None,
        history: list[Message],
        turns: list[int],
        current_turn: int,
        state: CompactionState,
        summarize: Summarizer | None,
        target_tokens: int | None = None,
    ) -> tuple[list[Message], CompactionReport | None]:
        usable = self.config.usable
        if usable is None and target_tokens is None:
            return render_view(system_prompt, history, state), None

        fixed = fixed_chars(system_prompt, tools)

        def estimate() -> tuple[list[Message], int]:
            view = render_view(system_prompt, history, state)
            return view, estimate_tokens(fixed + view_chars(view), state)

        view, before = estimate()
        report = CompactionReport(
            before_tokens=before, after_tokens=before, usable=usable,
            fixed_tokens=math.ceil(fixed / CHARS_PER_TOKEN))
        report.fixed_over = usable is not None and report.fixed_tokens > usable

        if target_tokens is None:
            if before <= self.config.trigger * usable:
                return view, (report if report.fixed_over else None)
            target = int(self.config.target * usable)
        else:
            target = target_tokens

        current = before

        def stub(pos: int) -> int:
            """Stub one tool result; return the tokens it saved."""
            m = history[pos]
            call = next((tc for x in history for tc in (x.tool_calls or [])
                         if tc.id == m.tool_call_id), None)
            original = len(_content_text(m))
            if call is None:
                return 0
            stub_len = len(stub_text(call, math.ceil(original / CHARS_PER_TOKEN)))
            if stub_len >= original:
                return 0
            state.stubbed.append(pos)
            state.stubbed.sort()
            return math.ceil((original - stub_len) / CHARS_PER_TOKEN)

        start = state.summary.covers_through if state.summary else 0
        stubbed = set(state.stubbed)
        last_assistant = max(
            (p for p, m in enumerate(history) if m.role == "assistant"), default=-1)

        # Step 1: tool results from finished turns, oldest first.
        for pos in range(start, len(history)):
            if current <= target:
                break
            if (history[pos].role == "tool" and turns[pos] < current_turn
                    and pos not in stubbed and pos not in state.clipped):
                saved = stub(pos)
                if saved:
                    current -= saved
                    report.stubbed_finished += 1

        # Step 2: older tool results in the current turn; the latest batch,
        # which the model has not answered yet, stays whole.
        for pos in range(start, last_assistant):
            if current <= target:
                break
            if (history[pos].role == "tool" and turns[pos] == current_turn
                    and pos not in stubbed and pos not in state.clipped):
                saved = stub(pos)
                if saved:
                    current -= saved
                    report.stubbed_current += 1

        # Step 3 (summary) is wired in by the summary step.
        if current > target and summarize is not None:
            view, current = estimate()
            current = await self._summarize(
                history=history, turns=turns, current_turn=current_turn,
                state=state, summarize=summarize, report=report,
                current=current, estimate=estimate)

        # Step 4: clip the single largest message.
        if current > target:
            view, current = estimate()
            if current > target:
                self._clip_largest(history, state, view, current - target, report)

        view, after = estimate()
        report.after_tokens = after
        return view, report

    async def _summarize(self, **kw) -> int:
        return kw["current"]

    def _clip_largest(self, history, state, view, excess_tokens, report) -> None:
        start = state.summary.covers_through if state.summary else 0
        stubbed = set(state.stubbed)
        best, best_len = None, 0
        for pos in range(start, len(history)):
            m = history[pos]
            if pos in stubbed:
                continue
            n = len(_content_text(m))
            if n > best_len:
                best, best_len = pos, n
        if best is None:
            return
        marker = len(CLIP_MARKER.format(tokens=10**6))
        already = state.clipped.get(best, 0)
        remove = already + excess_tokens + math.ceil(marker / CHARS_PER_TOKEN) + 1
        max_remove = (best_len - MIN_CLIP_KEEP_CHARS) // CHARS_PER_TOKEN
        remove = min(remove, max_remove)
        if remove <= already:
            return
        state.clipped[best] = remove
        report.clipped += 1
