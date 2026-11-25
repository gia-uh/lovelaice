"""Compaction state and the view sent to the model.

The stored conversation is never edited. What the model sees is a view: the
system prompt plus the history, with tool results and batches replaced, old text clipped, and old turns
folded into a summary. The view is a
pure function of the history and a `CompactionState`, and the state only grows,
so the start of the prompt changes only when compaction runs.
"""
from __future__ import annotations

import json
import math
from datetime import datetime

from lingo.llm import Message, ToolCall
from pydantic import BaseModel, Field

CHARS_PER_TOKEN = 4

STUB = "[result elided to save context: {call}, {tokens:,} tokens{reference}.]"
CLIP_MARKER = "\n[… {tokens:,} tokens removed to save context …]\n"
SUMMARY_FOOTER = "\n[End of summary]\n\n"
STUB_ARGS_CHARS = 120


class Summary(BaseModel):
    text: str
    covers_through: int  # history positions [0, covers_through) are folded in
    turns_covered: int
    model: str
    created_at: datetime
    runs: int = 1


class CompactionState(BaseModel):
    policy_key: str | None = None
    history_length: int = 0
    stubbed: list[int] = Field(default_factory=list)
    batches: dict[int, int] = Field(default_factory=dict)  # first position -> exclusive end
    clipped: dict[int, int] = Field(default_factory=dict)  # position -> tokens removed
    summary: Summary | None = None
    last_prompt_tokens: int | None = None
    last_view_chars: int | None = None


def _content_text(m: Message) -> str:
    c = m.content
    if isinstance(c, str):
        return c
    if hasattr(c, "model_dump"):
        return json.dumps(c.model_dump(), default=str)
    return "" if c is None else str(c)


def message_chars(m: Message) -> int:
    n = len(_content_text(m))
    for tc in m.tool_calls or []:
        n += len(tc.name) + len(json.dumps(tc.arguments or {}, default=str))
    if m.tool_call_id:
        n += len(m.tool_call_id)
    return n


def fixed_chars(system_prompt: str, tools: list | None) -> int:
    """The part compaction never cuts: system prompt and tool schemas."""
    from lingo.llm import tool_to_openai_schema
    n = len(system_prompt)
    for t in tools or []:
        n += len(json.dumps(tool_to_openai_schema(t), default=str))
    return n


def estimate_tokens(view_chars: int, state: CompactionState) -> int:
    """Provider-calibrated estimate: the last reported prompt size plus the
    character difference since, at CHARS_PER_TOKEN, never below the raw
    CHARS_PER_TOKEN count.

    The floor matters after a large cut. Text that tokenizes at more than four
    characters per token makes the calibrated figure fall too far: on a live
    Qwen 3.5 9B run it said 906 tokens where the provider counted 1,866, and
    the raw count said 1,905."""
    raw = math.ceil(view_chars / CHARS_PER_TOKEN)
    if state.last_prompt_tokens is not None and state.last_view_chars is not None:
        delta = view_chars - state.last_view_chars
        return max(raw, state.last_prompt_tokens + math.ceil(delta / CHARS_PER_TOKEN))
    return raw


def _args_text(arguments: dict) -> str:
    text = ", ".join(
        f"{k}={json.dumps(v, ensure_ascii=False, default=str)}"
        for k, v in (arguments or {}).items())
    if len(text) > STUB_ARGS_CHARS:
        text = text[:STUB_ARGS_CHARS] + "…"
    return text


def history_reference(position: int) -> str:
    """One-based position in the append-only conversation transcript."""
    return f"history message {position + 1}"


def stub_text(call: ToolCall, tokens: int, position: int | None = None) -> str:
    reference = "" if position is None else f"; original: {history_reference(position)}"
    return STUB.format(call=f"{call.name}({_args_text(call.arguments)})",
                       tokens=tokens, reference=reference)


def batch_text(calls: list[ToolCall], start: int, end: int) -> str:
    names = ", ".join(dict.fromkeys(call.name for call in calls))
    return (f"[tool batch elided: {names}; original: "
            f"{history_reference(start)}–{end}.]")


def clip_text(content: str, tokens_removed: int, position: int | None = None) -> str:
    keep = max(0, len(content) - tokens_removed * CHARS_PER_TOKEN)
    head = keep - keep // 2
    tail = keep // 2
    marker = CLIP_MARKER.format(tokens=tokens_removed)
    if position is not None:
        marker += f"[original: {history_reference(position)}]\n"
    return content[:head] + marker + (content[-tail:] if tail else "")


def summary_header(turns_covered: int) -> str:
    return f"[Summary of the earlier conversation, turns 1–{turns_covered}]\n"


def tool_batches(history: list[Message], first: int = 0):
    """Yield complete, contiguous OpenAI tool-call groups in source order."""
    for start in range(first, len(history)):
        message = history[start]
        calls = message.tool_calls
        if message.role != "assistant" or not calls:
            continue
        end = start + len(calls) + 1
        results = history[start + 1:end]
        expected = {call.id for call in calls}
        if (len(expected) == len(results) == len(calls)
                and all(result.role == "tool" for result in results)
                and {result.tool_call_id for result in results} == expected):
            yield start, end, {call.id: call for call in calls}


def render_view(system_prompt: str, history: list[Message],
                state: CompactionState) -> list[Message]:
    """System prompt plus the history as the state says the model sees it."""
    start = state.summary.covers_through if state.summary else 0
    stubbed = set(state.stubbed)
    calls = {
        pos: calls[history[pos].tool_call_id]
        for pos_start, end, calls in tool_batches(history, start)
        for pos in range(pos_start + 1, end)
        if pos in stubbed
    } if stubbed else {}
    out = [Message.system(system_prompt)]
    pending: list[str] = []
    pos = start
    while pos < len(history):
        m = history[pos]
        content = m.content
        if end := state.batches.get(pos):
            pending.extend(filter(None, (
                _content_text(m), batch_text(m.tool_calls or [], pos, end),
            )))
            pos = end
            continue
        if pos in stubbed and m.role == "tool":
            call = calls.get(pos)
            tokens = math.ceil(len(_content_text(m)) / CHARS_PER_TOKEN)
            content = (stub_text(call, tokens, pos) if call is not None
                       else STUB.format(call="tool", tokens=tokens,
                                        reference=f"; original: {history_reference(pos)}"))
        elif pos in state.clipped:
            content = clip_text(_content_text(m), state.clipped[pos], pos)
        if pos == start and state.summary is not None:
            content = (summary_header(state.summary.turns_covered)
                       + state.summary.text + SUMMARY_FOOTER
                       + (content if isinstance(content, str) else _content_text(m)))
        if pending:
            if m.role == "assistant":
                text = content if isinstance(content, str) else _content_text(m)
                content = "\n".join([*pending, text] if text else pending)
            else:
                out.append(Message.assistant("\n".join(pending)))
            pending.clear()
        out.append(m.model_copy(update={"content": content}) if content is not m.content
                   else m)
        pos += 1
    if pending:
        out.append(Message.assistant("\n".join(pending)))
    return out


def view_chars(view: list[Message]) -> int:
    """Characters of a view, system prompt excluded (fixed_chars counts it)."""
    return sum(message_chars(m) for m in view[1:])


def check_tool_pairs(view: list[Message]) -> None:
    """Raise ValueError if a tool call has no result or a result has no call."""
    open_ids: list[str] = []
    seen: set[str] = set()
    for m in view:
        if m.role == "assistant":
            if open_ids:
                raise ValueError(f"tool call without result: {open_ids[0]}")
            open_ids = [tc.id for tc in (m.tool_calls or [])]
            seen = set(open_ids)
        elif m.role == "tool":
            if m.tool_call_id not in seen:
                raise ValueError(f"tool result without call: {m.tool_call_id}")
            if m.tool_call_id in open_ids:
                open_ids.remove(m.tool_call_id)
        elif m.role == "user" and open_ids:
            raise ValueError(f"tool call without result: {open_ids[0]}")
    if open_ids:
        raise ValueError(f"tool call without result: {open_ids[0]}")
