"""Tests for lovelaice.agent.conversation — persistent conversations."""
from __future__ import annotations

from datetime import datetime, timezone

from lovelaice.agent.conversation_models import ConversationRow, ConversationMeta


def test_conversation_row_round_trips_via_pydantic():
    now = datetime.now(timezone.utc)
    row = ConversationRow(
        id="abc123",
        created_at=now,
        updated_at=now,
        model="anthropic/claude-haiku-4-5",
        system_prompt_hash="sha256:deadbeef",
        archived=False,
        messages=[],
    )
    dumped = row.model_dump()
    restored = ConversationRow.model_validate(dumped)
    assert restored.id == "abc123"
    assert restored.archived is False
    assert restored.messages == []


def test_conversation_meta_omits_messages():
    meta = ConversationMeta(
        id="abc",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        archived=False,
        turn_count=3,
    )
    assert "messages" not in meta.model_dump()
