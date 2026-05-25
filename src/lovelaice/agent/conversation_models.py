"""Pydantic models for the persistent Conversation concept.

Stored as one row per conversation in a beaverdb dict keyed by UUID.
Messages are embedded; the row carries enough metadata to render a
chat-list view (ConversationMeta) without loading the messages.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class MessageEntry(BaseModel):
    """One serialised lingo.Message. Shape matches the existing jsonl
    Session schema in lovelaice/agent/session.py — only the substrate moves."""

    type: str = "message"
    id: str
    role: str
    timestamp: datetime
    content: Any
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    thinking: str | None = None
    stop_reason: str | None = None
    usage: dict[str, Any] | None = None


class ConversationRow(BaseModel):
    """One persistent conversation. Stored in beaverdb dict 'conversations'."""

    id: str
    created_at: datetime
    updated_at: datetime
    model: str
    system_prompt_hash: str
    archived: bool = False
    messages: list[MessageEntry] = Field(default_factory=list)


class ConversationMeta(BaseModel):
    """Lightweight projection — id + bookkeeping, no messages payload."""

    id: str
    created_at: datetime
    updated_at: datetime
    archived: bool
    turn_count: int
