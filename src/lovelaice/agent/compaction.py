"""Compaction state and the view sent to the model.

The stored conversation is never edited. What the model sees is a view: the
system prompt plus the history, with old tool results replaced by stubs, one
oversized message clipped, and old turns folded into a summary. The view is a
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

STUB = ("[result elided to save context: {call}, {tokens:,} tokens. "
        "Call the tool again if you need it.]")
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
    stubbed: list[int] = Field(default_factory=list)
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
    character difference since, at CHARS_PER_TOKEN."""
    if state.last_prompt_tokens is not None and state.last_view_chars is not None:
        delta = view_chars - state.last_view_chars
        return max(0, state.last_prompt_tokens + math.ceil(delta / CHARS_PER_TOKEN))
    return math.ceil(view_chars / CHARS_PER_TOKEN)


def _args_text(arguments: dict) -> str:
    text = ", ".join(
        f"{k}={json.dumps(v, ensure_ascii=False, default=str)}"
        for k, v in (arguments or {}).items())
    if len(text) > STUB_ARGS_CHARS:
        text = text[:STUB_ARGS_CHARS] + "…"
    return text


def stub_text(call: ToolCall, tokens: int) -> str:
    return STUB.format(call=f"{call.name}({_args_text(call.arguments)})", tokens=tokens)


def clip_text(content: str, tokens_removed: int) -> str:
    keep = max(0, len(content) - tokens_removed * CHARS_PER_TOKEN)
    head = keep - keep // 2
    tail = keep // 2
    return (content[:head] + CLIP_MARKER.format(tokens=tokens_removed)
            + (content[len(content) - tail:] if tail else ""))


def summary_header(turns_covered: int) -> str:
    return f"[Summary of the earlier conversation, turns 1–{turns_covered}]\n"


def _calls_by_id(history: list[Message]) -> dict[str, ToolCall]:
    return {tc.id: tc for m in history for tc in (m.tool_calls or [])}


def render_view(system_prompt: str, history: list[Message],
                state: CompactionState) -> list[Message]:
    """System prompt plus the history as the state says the model sees it."""
    calls = _calls_by_id(history) if state.stubbed else {}
    stubbed = set(state.stubbed)
    start = state.summary.covers_through if state.summary else 0
    out = [Message.system(system_prompt)]
    for pos in range(start, len(history)):
        m = history[pos]
        content = m.content
        if pos in stubbed and m.role == "tool":
            call = calls.get(m.tool_call_id or "")
            tokens = math.ceil(len(_content_text(m)) / CHARS_PER_TOKEN)
            content = (stub_text(call, tokens) if call is not None
                       else STUB.format(call="tool", tokens=tokens))
        elif pos in state.clipped:
            content = clip_text(_content_text(m), state.clipped[pos])
        if pos == start and state.summary is not None:
            text = content if isinstance(content, str) else _content_text(m)
            content = (summary_header(state.summary.turns_covered)
                       + state.summary.text + SUMMARY_FOOTER + text)
        out.append(m.model_copy(update={"content": content}) if content is not m.content
                   else m)
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
