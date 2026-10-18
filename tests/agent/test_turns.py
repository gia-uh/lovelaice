"""Turn numbers: one per prompt() call, recorded on every stored message."""
import pytest
from unittest.mock import AsyncMock
from lingo.llm import Message

from lovelaice.agent.harness import Harness
from lovelaice.agent.hooks import HookRegistry
from lovelaice.agent.loops.react_native import ReActNative
from lovelaice.agent.session import Session
from lovelaice.agent.tools import ToolRegistry
from lovelaice.agent.turns import assign_turns

NUDGE = ReActNative.EMPTY_TURN_NUDGE


def test_nudge_stays_in_its_turn():
    items = [("user", "hi", None), ("assistant", "", None),
             ("user", NUDGE, None), ("assistant", "done", None),
             ("user", "next", None), ("assistant", "ok", None)]
    assert assign_turns(items, NUDGE) == [1, 1, 1, 1, 2, 2]


def test_stored_turn_wins():
    items = [("user", "a", 4), ("assistant", "b", 4), ("user", "c", None)]
    assert assign_turns(items, NUDGE) == [4, 4, 5]


def test_legacy_rows_infer_turns():
    items = [("user", "a", None), ("assistant", "", None), ("tool", "r", None),
             ("user", NUDGE, None), ("assistant", "x", None)]
    assert assign_turns(items, NUDGE) == [1, 1, 1, 1, 1]


def test_messages_before_first_user_are_turn_zero():
    assert assign_turns([("assistant", "hello", None), ("user", "q", None)], NUDGE) == [0, 1]


@pytest.mark.asyncio
async def test_appends_carry_turn_number(tmp_path):
    responses = iter([
        Message.assistant("", stop_reason="stop"),       # empty -> nudge
        Message.assistant("done", stop_reason="stop"),
        Message.assistant("ok", stop_reason="stop"),
    ])
    llm = AsyncMock()
    llm.chat = AsyncMock(side_effect=lambda *a, **kw: next(responses))
    harness = Harness(llm=llm, tools=ToolRegistry(), hooks=HookRegistry(),
                      system_prompt="SYS")
    sess = Session.create(tmp_path / "s.jsonl", model="x", system_prompt_hash="h",
                          loop="ReActNative", cwd=str(tmp_path))
    loop = ReActNative()
    await loop.run(harness, sess, Message.user("first"))
    await loop.run(harness, sess, Message.user("second"))
    assert sess.turn_numbers() == [1, 1, 1, 1, 2, 2]
    reloaded = Session.load(tmp_path / "s.jsonl")
    assert reloaded.turn_numbers() == [1, 1, 1, 1, 2, 2]
