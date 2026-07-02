"""Append-only JSONL session log.

Linear (no branching). Two entry types: session_header (first line),
message (every subsequent line). Crash-safe: fsync per append; skip
malformed trailing lines on load.
"""
import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from lingo.llm import Message, ToolCall


SCHEMA_VERSION = 1


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _msg_id() -> str:
    return f"msg_{uuid.uuid4().hex[:8]}"


def _serialize_message(msg: Message) -> dict:
    """Convert a lingo.Message into a JSONL entry dict."""
    entry: dict = {
        "type": "message",
        "id": _msg_id(),
        "role": msg.role,
        "timestamp": _now_iso(),
    }
    # Content: lingo's content may be str or a Content subtype.
    if isinstance(msg.content, str):
        entry["content"] = msg.content
    elif hasattr(msg.content, "model_dump"):
        entry["content"] = msg.content.model_dump()
    else:
        entry["content"] = str(msg.content)
    if msg.tool_calls:
        entry["tool_calls"] = [
            {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
            for tc in msg.tool_calls
        ]
    if msg.tool_call_id is not None:
        entry["tool_call_id"] = msg.tool_call_id
    if msg.thinking:
        entry["thinking"] = msg.thinking
    if msg.stop_reason:
        entry["stop_reason"] = msg.stop_reason
    if msg.usage:
        entry["usage"] = msg.usage.model_dump()
    return entry


def _deserialize_message(entry: dict) -> Message:
    """Reverse of _serialize_message — into a lingo.Message."""
    role = entry["role"]
    content = entry.get("content", "")
    tool_calls = None
    if entry.get("tool_calls"):
        tool_calls = [
            ToolCall(id=tc["id"], name=tc["name"], arguments=tc.get("arguments", {}))
            for tc in entry["tool_calls"]
        ]
    return Message(
        role=role,
        content=content,
        tool_calls=tool_calls,
        tool_call_id=entry.get("tool_call_id"),
        thinking=entry.get("thinking"),
        stop_reason=entry.get("stop_reason"),
    )


class Session:
    """Append-only JSONL session log. Linear, no branches."""

    def __init__(self, path: Path, entries: list[dict]):
        self.path = path
        self._entries = entries

    @classmethod
    def create(
        cls,
        path: Path,
        *,
        model: str,
        system_prompt_hash: str,
        loop: str,
        cwd: str,
    ) -> "Session":
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = {
            "type": "session_header",
            "id": uuid.uuid4().hex[:8],
            "schema_version": SCHEMA_VERSION,
            "cwd": cwd,
            "model": model,
            "system_prompt_hash": system_prompt_hash,
            "loop": loop,
            "created_at": _now_iso(),
        }
        with path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(header) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return cls(path, [header])

    @classmethod
    def load(cls, path: Path) -> "Session":
        path = Path(path)
        entries: list[dict] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    # Skip trailing malformed line (crash mid-write).
                    break
        if not entries or entries[0].get("type") != "session_header":
            raise ValueError(f"Session {path} has no header (unloadable)")
        if entries[0].get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"Session {path} schema_version={entries[0].get('schema_version')} "
                f"unsupported (expected {SCHEMA_VERSION})"
            )
        return cls(path, entries)

    def append(self, msg: Message) -> dict:
        entry = _serialize_message(msg)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._entries.append(entry)
        return entry

    def update_tool_call_args(self, call_id: str, new_args: dict) -> None:
        """Rewrite the stored arguments of the assistant tool call `call_id`.

        Walks entries newest-first and updates the first matching tool_call's
        `arguments` in place (in-memory `_entries` only — the append-only JSONL
        keeps the honest emitted record). No-op if the call id is not found.
        Used by the repair path so later turns see well-formed calls in history.
        """
        for entry in reversed(self._entries):
            if entry.get("type") != "message":
                continue
            for tc in entry.get("tool_calls") or []:
                if tc.get("id") == call_id:
                    tc["arguments"] = new_args
                    return

    @property
    def turn_count(self) -> int:
        """Number of user messages (a rough proxy for turns)."""
        return sum(
            1 for e in self._entries
            if e.get("type") == "message" and e.get("role") == "user"
        )

    def messages_for_llm(self, system_prompt: str) -> list[Message]:
        """Reconstruct the lingo.Message list to pass to LLM.chat.

        Prepends a synthesized system message, then walks the JSONL
        entries in order and emits each message-typed entry."""
        out: list[Message] = [Message.system(system_prompt)]
        for entry in self._entries:
            if entry.get("type") != "message":
                continue
            out.append(_deserialize_message(entry))
        return out

    @staticmethod
    def hash_system_prompt(prompt: str) -> str:
        return "sha256:" + hashlib.sha256(prompt.encode("utf-8")).hexdigest()
