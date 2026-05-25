"""Tests for lovelaice.agent.conversation — persistent conversations."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
import pytest_asyncio

from beaver import AsyncBeaverDB
from lingo.llm import Message

from lovelaice.agent.conversation import ConversationStore
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


@pytest_asyncio.fixture
async def store(tmp_path: Path):
    db = AsyncBeaverDB(str(tmp_path / "lovelaice.db"))
    await db.connect()
    try:
        yield ConversationStore(db)
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_create_returns_conversation_with_uuid(store: ConversationStore):
    conv = await store.create(
        model="anthropic/claude-haiku-4-5",
        system_prompt_hash="sha256:test",
    )
    assert len(conv.id) > 0
    assert conv.archived is False
    assert conv.row.messages == []


@pytest.mark.asyncio
async def test_get_returns_none_for_unknown(store: ConversationStore):
    assert await store.get("nope") is None


@pytest.mark.asyncio
async def test_append_persists_messages(store: ConversationStore):
    conv = await store.create(model="x", system_prompt_hash="h")
    await store.append(conv.id, Message.user("hi"))
    await store.append(conv.id, Message.assistant("hello"))
    fresh = await store.get(conv.id)
    assert fresh is not None
    assert len(fresh.row.messages) == 2
    assert fresh.row.messages[0].role == "user"
    assert fresh.row.messages[1].role == "assistant"


@pytest.mark.asyncio
async def test_archive_flips_flag(store: ConversationStore):
    conv = await store.create(model="x", system_prompt_hash="h")
    await store.archive(conv.id)
    fresh = await store.get(conv.id)
    assert fresh is not None
    assert fresh.archived is True


@pytest.mark.asyncio
async def test_list_filters_archived(store: ConversationStore):
    a = await store.create(model="x", system_prompt_hash="h")
    b = await store.create(model="x", system_prompt_hash="h")
    await store.archive(a.id)
    live = await store.list(archived=False)
    arch = await store.list(archived=True)
    assert {m.id for m in live} == {b.id}
    assert {m.id for m in arch} == {a.id}
