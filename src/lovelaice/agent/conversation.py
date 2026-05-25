"""Persistent conversation store — beaverdb-backed, keyed by UUID.

The store wraps one beaverdb dict named 'conversations'. Each row is
a ConversationRow. The Conversation wrapper exposes the same shape as
the existing jsonl Session (append, messages_for_llm) so the agent
can drop it in.

Cross-process safety: aiosqlite + WAL mode handles a handful of writers
on one db file. Warden's parent opens it (for archive); spawned ACP
subprocesses open it (for append). We assume at most one subprocess
writes a given conversation at a time (enforced upstream by warden's
session map).
"""
from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from beaver import AsyncBeaverDB
from lingo.llm import Message, ToolCall

from lovelaice.agent.conversation_models import (
    ConversationMeta,
    ConversationRow,
    MessageEntry,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return uuid.uuid4().hex


def hash_system_prompt(prompt: str) -> str:
    return "sha256:" + hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _serialise(msg: Message) -> MessageEntry:
    content = msg.content
    if hasattr(content, "model_dump"):
        content_payload = content.model_dump()
    elif isinstance(content, str):
        content_payload = content
    else:
        content_payload = str(content)
    tool_calls = [
        {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
        for tc in (msg.tool_calls or [])
    ] or None
    return MessageEntry(
        id=f"msg_{uuid.uuid4().hex[:8]}",
        role=msg.role,
        timestamp=_now(),
        content=content_payload,
        tool_calls=tool_calls,
        tool_call_id=msg.tool_call_id,
        thinking=msg.thinking,
        stop_reason=msg.stop_reason,
        usage=msg.usage.model_dump() if msg.usage else None,
    )


def _deserialise(entry: MessageEntry) -> Message:
    tool_calls = None
    if entry.tool_calls:
        tool_calls = [
            ToolCall(id=tc["id"], name=tc["name"], arguments=tc.get("arguments", {}))
            for tc in entry.tool_calls
        ]
    return Message(
        role=entry.role,
        content=entry.content,
        tool_calls=tool_calls,
        tool_call_id=entry.tool_call_id,
        thinking=entry.thinking,
        stop_reason=entry.stop_reason,
    )


@dataclass
class Conversation:
    """Live handle on a persisted conversation. Append goes through the store."""

    row: ConversationRow
    store: "ConversationStore"

    @property
    def id(self) -> str:
        return self.row.id

    @property
    def archived(self) -> bool:
        return self.row.archived

    def messages_for_llm(self, system_prompt: str) -> list[Message]:
        out = [Message.system(system_prompt)]
        out.extend(_deserialise(m) for m in self.row.messages)
        return out

    async def append(self, msg: Message) -> None:
        await self.store.append(self.row.id, msg)


class ConversationStore:
    def __init__(self, db: AsyncBeaverDB):
        self._dict = db.dict("conversations", model=ConversationRow)

    async def create(
        self, *, model: str, system_prompt_hash: str
    ) -> Conversation:
        cid = _new_id()
        now = _now()
        row = ConversationRow(
            id=cid,
            created_at=now,
            updated_at=now,
            model=model,
            system_prompt_hash=system_prompt_hash,
            archived=False,
            messages=[],
        )
        await self._dict.set(cid, row)
        return Conversation(row=row, store=self)

    async def get(self, conversation_id: str) -> Conversation | None:
        row = await self._dict.fetch(conversation_id, default=None)
        if row is None:
            return None
        return Conversation(row=row, store=self)

    async def append(self, conversation_id: str, message: Message) -> None:
        row = await self._dict.fetch(conversation_id, default=None)
        if row is None:
            raise KeyError(f"unknown conversation: {conversation_id}")
        row.messages.append(_serialise(message))
        row.updated_at = _now()
        await self._dict.set(conversation_id, row)

    async def archive(self, conversation_id: str) -> None:
        row = await self._dict.fetch(conversation_id, default=None)
        if row is None:
            return  # idempotent
        row.archived = True
        row.updated_at = _now()
        await self._dict.set(conversation_id, row)

    async def list(self, *, archived: bool = False) -> list[ConversationMeta]:
        out: list[ConversationMeta] = []
        async for key, row in self._dict.items():
            if row.archived != archived:
                continue
            out.append(ConversationMeta(
                id=row.id,
                created_at=row.created_at,
                updated_at=row.updated_at,
                archived=row.archived,
                turn_count=sum(1 for m in row.messages if m.role == "user"),
            ))
        out.sort(key=lambda m: m.updated_at, reverse=True)
        return out
