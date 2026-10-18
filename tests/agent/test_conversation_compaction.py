"""Compaction state saved per conversation, apart from the message row."""
import asyncio
from pathlib import Path

import pytest
import pytest_asyncio
from beaver import AsyncBeaverDB
from lingo.llm import Message

from lovelaice.agent import Agent, AgentConfig
from lovelaice.agent.compaction import CompactionState
from lovelaice.agent.conversation import ConversationStore
from lovelaice.agent.loops.react_native import ReActNative

from tests.agent.test_model_call import ScriptedLLM


@pytest_asyncio.fixture
async def store(tmp_path: Path):
    db = AsyncBeaverDB(str(tmp_path / "lovelaice.db"))
    await db.connect()
    try:
        yield ConversationStore(db)
    finally:
        await db.close()


async def drain():
    for _ in range(5):
        pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        if not pending:
            return
        await asyncio.gather(*pending)


def build_agent(monkeypatch, llm, store, conv, window=8192):
    monkeypatch.setattr("lovelaice.agent.agent._build_llm", lambda cfg: llm)
    return Agent.from_conversation(
        config=AgentConfig(model="m", system_prompt="SYS", context_window=window),
        tools=[], loop=ReActNative(), conversation=conv, store=store)


@pytest.mark.asyncio
async def test_reloaded_conversation_does_not_summarize_again(store, monkeypatch):
    llm = ScriptedLLM([])
    conv = await store.create(model="m", system_prompt_hash="h")
    agent = build_agent(monkeypatch, llm, store, conv)
    for i in range(8):
        await agent.prompt(f"turn {i}: " + "y" * 3000)
        await drain()
    assert llm.summary_calls >= 1
    summaries_before = llm.summary_calls

    agent2 = build_agent(monkeypatch, llm, store, await store.get(conv.id))
    assert agent2.session.compaction.summary is not None
    await agent2.prompt("one more short question")
    await drain()
    assert llm.summary_calls == summaries_before


@pytest.mark.asyncio
async def test_state_saved_under_its_own_key(store):
    conv = await store.create(model="m", system_prompt_hash="h")
    await store.save_compaction(conv.id, CompactionState(stubbed=[3]))
    await store.append(conv.id, Message.user("x"))
    fresh = await store.get(conv.id)
    assert fresh.compaction.stubbed == [3]
    assert "compaction" not in fresh.row.model_dump()
    assert len(fresh.row.messages) == 1


@pytest.mark.asyncio
async def test_new_conversation_has_empty_state(store):
    conv = await store.create(model="m", system_prompt_hash="h")
    assert conv.compaction == CompactionState()
    assert (await store.get(conv.id)).compaction == CompactionState()


@pytest.mark.asyncio
async def test_tool_arg_repair_leaves_state_alone(store, monkeypatch):
    conv = await store.create(model="m", system_prompt_hash="h")
    agent = build_agent(monkeypatch, ScriptedLLM([]), store, conv)
    agent.session.compaction.stubbed.append(2)
    before = agent.session.compaction.model_copy(deep=True)
    agent.session.update_tool_call_args("nope", {"a": 1})
    assert agent.session.compaction == before
