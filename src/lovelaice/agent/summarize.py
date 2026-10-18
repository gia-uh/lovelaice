"""Step 3 of compaction: the request that folds old turns into a summary."""
from __future__ import annotations

import json

from lingo.llm import Message

SUMMARY_INSTRUCTIONS = (
    "You compress the earlier part of a conversation between a user and an assistant "
    "that uses tools,\nso the assistant can keep working without the full transcript. "
    "Write a summary that keeps every\nfact, decision, name, number, file or note "
    "reference, and open question or unfinished task.\nDrop greetings, repetition and "
    "tool output that was only read to answer an earlier question.\nWrite in the "
    "language the user writes in. Use at most {cap} tokens. Output only the summary."
)


def render_transcript(messages: list[Message]) -> str:
    lines = []
    for m in messages:
        content = m.content if isinstance(m.content, str) else str(m.content or "")
        if m.role == "assistant" and m.tool_calls:
            calls = ", ".join(
                f"→ {tc.name}({json.dumps(tc.arguments or {}, ensure_ascii=False, default=str)})"
                for tc in m.tool_calls)
            content = (content + " " + calls).strip()
        lines.append(f"{m.role}: {content}")
    return "\n".join(lines)


def build_summary_request(previous: str | None, messages: list[Message],
                          cap_tokens: int) -> list[Message]:
    parts = []
    if previous:
        parts.append(f"Previous summary:\n{previous}")
        parts.append("Write one updated summary that merges the previous summary with "
                     "the conversation below: keep everything from the previous summary "
                     "that still matters, then add what is new.")
    parts.append(f"Conversation to add:\n{render_transcript(messages)}")
    return [Message.system(SUMMARY_INSTRUCTIONS.format(cap=cap_tokens)),
            Message.user("\n\n".join(parts))]
