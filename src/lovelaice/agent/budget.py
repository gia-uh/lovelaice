"""Prepare the view sent to the model under an ordered compaction policy."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import groupby
from typing import Awaitable, Callable

from lingo.llm import Message, Usage

from lovelaice.agent.compaction import (
    CHARS_PER_TOKEN, CompactionState, Summary, _content_text, batch_text, clip_text,
    estimate_tokens, fixed_chars, message_chars, render_view, stub_text, tool_batches,
    view_chars,
)
from lovelaice.agent.policy import CompactionPolicy, CompactionStep
from lovelaice.agent.summarize import MAX_SUMMARY_CALLS, summarize_prefix

Summarizer = Callable[[str | None, str, int], Awaitable[tuple[str, Usage | None]]]
MIN_CLIP_KEEP_CHARS = 400


@dataclass
class BudgetConfig:
    window: int | None
    max_tokens: int | None = None
    trigger: float = 0.75
    target: float = 0.5
    model: str = ""
    policy: CompactionPolicy | None = None
    endpoint: str | None = None

    @property
    def settings(self) -> CompactionPolicy:
        return self.policy if self.policy is not None else CompactionPolicy()

    @property
    def trigger_fraction(self) -> float:
        return self.policy.trigger_percent / 100 if self.policy is not None else self.trigger

    @property
    def target_fraction(self) -> float:
        return self.policy.target_percent / 100 if self.policy is not None else self.target

    def reserve_for(self, window: int | None) -> int:
        return self.max_tokens or min(2048, (window or 0) // 4)

    @property
    def reserve(self) -> int:
        return self.reserve_for(self.window)

    def usable_for(self, window: int | None, input_limit: int | None = None) -> int | None:
        if window is None:
            return input_limit
        usable = max(window - self.reserve_for(window), 0)
        return min(usable, input_limit) if input_limit is not None else usable

    @property
    def usable(self) -> int | None:
        return self.usable_for(self.window)

    def should_compact(self, size: int, fixed: int, usable: int) -> bool:
        room = max(usable - fixed, 0)
        if self.policy is not None:
            return (size - fixed) * 100 > room * self.policy.trigger_percent
        return size > fixed + self.trigger * room

    def target_for(self, fixed: int, usable: int) -> int:
        room = max(usable - fixed, 0)
        if self.policy is not None:
            return fixed + room * self.policy.target_percent // 100
        return fixed + math.floor(self.target * room)

    def policy_key(self) -> str:
        data = (self.trigger_fraction, self.target_fraction, self.settings.protected_turns,
                self.settings.steps, self.model, self.endpoint, self.window, self.max_tokens)
        return "1:" + hashlib.sha256(json.dumps(data).encode()).hexdigest()


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
    operations: dict[CompactionStep, int] = field(default_factory=dict)
    attempted: list[CompactionStep] = field(default_factory=list)
    window: int | None = None

    @property
    def cut(self) -> bool:
        return bool(self.operations or self.stubbed_finished or self.stubbed_current
                    or self.turns_summarized or self.clipped)

    def record(self, step: CompactionStep, count: int) -> None:
        if count:
            self.operations[step] = count


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
        window: int | None = None,
        input_limit: int | None = None,
        on_compacting: Callable[[int], None] | None = None,
    ) -> tuple[list[Message], CompactionReport | None]:
        window = self.config.window if window is None else window
        usable = self.config.usable_for(window, input_limit)
        view = render_view(system_prompt, history, state)
        if usable is None and target_tokens is None:
            return view, None

        fixed = fixed_chars(system_prompt, tools)
        chars = fixed + view_chars(view)
        before = estimate_tokens(chars, state)
        fixed_tokens = math.ceil(fixed / CHARS_PER_TOKEN)
        report = CompactionReport(
            before_tokens=before, after_tokens=before, usable=usable, fixed_tokens=fixed_tokens,
            fixed_over=usable is not None and fixed_tokens > usable, window=window)
        policy = self.config.settings
        if target_tokens is None:
            if not self.config.should_compact(before, fixed_tokens, usable):
                return view, report if report.fixed_over else None
            target = self.config.target_for(fixed_tokens, usable)
        else:
            target = target_tokens

        if policy.steps and before > target and on_compacting is not None:
            on_compacting(before)
        last_assistant = max(
            (pos for pos, msg in enumerate(history) if msg.role == "assistant"), default=-1)
        # When the provider provides no limit, a retry still has a forced input
        # target. Never use it to clip history on a proactive, fitting request.
        clip_limit = usable if usable is not None else target
        closed = list(dict.fromkeys(turn for turn in turns if turn < current_turn))
        oldest_protected = len(closed) - policy.protected_turns
        last_old_turn = closed[oldest_protected - 1] if oldest_protected > 0 else -1
        visible_start = state.summary.covers_through if state.summary else 0
        batches = list(tool_batches(history, visible_start))
        complete = {pos for pos, end, _ in batches if turns[pos] == turns[end - 1]}
        first_pending_turn = next((turns[pos] for pos, msg in enumerate(history)
                                   if pos >= visible_start and msg.tool_calls and pos not in complete),
                                  current_turn)
        last_summary_turn = min(last_old_turn, first_pending_turn - 1)
        for step in policy.steps:
            if estimate_tokens(chars, state) <= target:
                break
            report.attempted.append(step)
            start = state.summary.covers_through if state.summary else 0
            if step in (CompactionStep.FINISHED_RESULTS, CompactionStep.READ_RESULTS):
                old = step == CompactionStep.FINISHED_RESULTS

                def eligible(pos, end):
                    if pos not in complete:
                        return False
                    return (turns[pos] < current_turn if old else
                            turns[pos] == current_turn and end <= last_assistant)

                chars, count = self._stub_results(
                    history, batches, state, start, eligible, chars, target)
                if old:
                    report.stubbed_finished = count
                else:
                    report.stubbed_current = count
                report.record(step, count)
            elif step == CompactionStep.FINISHED_TOOL_BATCHES:
                chars, count = self._replace_batches(
                    history, turns, batches, state, start, current_turn, chars, target)
                report.record(step, count)
            elif step == CompactionStep.SUMMARIZE_OLD_TURNS and summarize is not None:
                await self._summarize(system_prompt, history, turns, last_summary_turn,
                                      state, summarize, report, window, target, fixed)
                report.record(step, report.turns_summarized)
            elif step == CompactionStep.TRUNCATE_OLD_TEXT:
                report.clipped = self._clip_old_text(
                    history, turns, state, start, last_old_turn, chars, clip_limit, target)
                report.record(step, report.clipped)

            view = render_view(system_prompt, history, state)
            chars = fixed + view_chars(view)

        report.after_tokens = estimate_tokens(chars, state)
        return view, report

    @staticmethod
    def _stub_results(history, batches, state, start, eligible, chars, target):
        stubbed = set(state.stubbed)
        count = 0
        for pos, end, calls in batches:
            if pos < start or pos in state.batches or not eligible(pos, end):
                continue
            for result in range(pos + 1, end):
                if estimate_tokens(chars, state) <= target:
                    return chars, count
                if result in stubbed:
                    continue
                msg = history[result]
                original = len(_content_text(msg))
                marker = stub_text(calls[msg.tool_call_id],
                                   math.ceil(original / CHARS_PER_TOKEN), result)
                if len(marker) < original:
                    state.stubbed.append(result)
                    chars -= original - len(marker)
                    count += 1
        return chars, count

    @staticmethod
    def _replace_batches(history, turns, batches, state, start, current_turn, chars, target):
        stubbed = set(state.stubbed)
        count = 0
        for pos, end, calls in batches:
            if estimate_tokens(chars, state) <= target:
                break
            if (pos < start or pos in state.batches or turns[pos] >= current_turn
                    or turns[end - 1] != turns[pos]):
                continue
            markers = []
            for result in range(pos + 1, end):
                message = history[result]
                if result in stubbed:
                    tokens = math.ceil(len(_content_text(message)) / CHARS_PER_TOKEN)
                    content = stub_text(calls[message.tool_call_id], tokens, result)
                    message = message.model_copy(update={"content": content})
                markers.append(message_chars(message))
            old = message_chars(history[pos]) + sum(markers)
            new = len(_content_text(history[pos])) + len(batch_text(list(calls.values()), pos, end)) + 2
            if new < old:
                state.batches[pos] = end
                chars -= old - new
                count += 1
        return chars, count

    def _clip_old_text(self, history, turns, state, start, last_old_turn, chars, limit, target):
        if estimate_tokens(chars, state) <= limit:
            return 0
        count = 0
        for pos in range(start, len(history)):
            if estimate_tokens(chars, state) <= limit:
                break
            msg = history[pos]
            if (turns[pos] > last_old_turn or msg.role not in ("user", "assistant")
                    or not isinstance(msg.content, str) or msg.tool_calls):
                continue
            old = state.clipped.get(pos, 0)
            text = msg.content
            existing = clip_text(text, old, pos) if old else text
            excess = estimate_tokens(chars, state) - target
            marker_size = len(clip_text("", old + excess, pos))
            keep_limit = (len(text) - MIN_CLIP_KEEP_CHARS) // CHARS_PER_TOKEN
            remove = min(keep_limit, old + excess + math.ceil(marker_size / CHARS_PER_TOKEN))
            if remove <= old:
                continue
            clipped = clip_text(text, remove, pos)
            if len(clipped) < len(existing):
                state.clipped[pos] = remove
                chars -= len(existing) - len(clipped)
                count += 1
        return count

    async def _summarize(self, system_prompt, history, turns, last_old_turn, state,
                         summarize, report, window, target, fixed):
        previous = state.summary
        start = previous.covers_through if previous else 0
        groups = []
        for turn, positions in groupby(range(start, len(history)), key=turns.__getitem__):
            if turn > last_old_turn:
                break
            positions = list(positions)
            groups.append((positions[0], positions[-1] + 1, turn))
        if not groups:
            return
        usable = report.usable if report.usable is not None else target
        cap = min(1024, usable // 10)
        if cap <= 0:
            return
        original = estimate_tokens(fixed + view_chars(render_view(system_prompt, history, state)), state)
        async for end, last_turn, text, usage in summarize_prefix(
            history, groups, previous.text if previous else None, summarize,
            input_budget=min(window - cap, usable) if window is not None else usable,
            cap=cap, max_calls=MAX_SUMMARY_CALLS,
        ):
            report.summary_usage = usage
            if text is None:
                continue
            summary = Summary(
                text=text, covers_through=end, turns_covered=last_turn,
                model=self.config.model, created_at=datetime.now(timezone.utc),
                runs=(previous.runs + 1) if previous else 1)
            candidate = state.model_copy(update={"summary": summary})
            size = estimate_tokens(
                fixed + view_chars(render_view(system_prompt, history, candidate)), state)
            if size < original:
                state.summary = summary
                state.stubbed = [pos for pos in state.stubbed if pos >= end]
                state.batches = {pos: stop for pos, stop in state.batches.items() if pos >= end}
                state.clipped = {pos: count for pos, count in state.clipped.items() if pos >= end}
                original = size
                report.turns_summarized = len(set(turns[start:end]))
                report.summary = text
            if original <= target:
                break
