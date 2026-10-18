"""The compaction state, the token estimate and the view sent to the model."""
from datetime import datetime, timezone

import pytest
from lingo.llm import Message, ToolCall

from lovelaice.agent.compaction import (
    CompactionState, Summary, check_tool_pairs, clip_text, estimate_tokens,
    render_view, stub_text, summary_header,
)

NOW = datetime(2026, 10, 17, tzinfo=timezone.utc)


def _call(i, name="web_fetch", **args):
    return Message(role="assistant", content="",
                   tool_calls=[ToolCall(id=f"c{i}", name=name, arguments=args)])


@pytest.fixture
def history():
    return [Message.user("hi"), Message.assistant("hello")]


@pytest.fixture
def history_with_tools():
    return [
        Message.user("fetch it"),
        _call(1, url="https://example.org/a"),
        Message.tool("x" * 4000, tool_call_id="c1"),
        Message.assistant("done"),
    ]


@pytest.fixture
def history_three_turns():
    return [
        Message.user("one"), Message.assistant("a1"),
        Message.user("two"), Message.assistant("a2"),
        Message.user("three"), Message.assistant("a3"),
    ]


def dump(view):
    return [m.model_dump() for m in view]


def test_view_without_state_is_identity(history):
    view = render_view("SYS", history, CompactionState())
    assert dump(view) == dump([Message.system("SYS"), *history])


def test_render_view_does_not_mutate_history(history_with_tools):
    before = dump(history_with_tools)
    render_view("SYS", history_with_tools, CompactionState(stubbed=[2], clipped={0: 1}))
    assert dump(history_with_tools) == before


def test_stub_form_exact():
    call = ToolCall(id="c1", name="web_fetch", arguments={"url": "https://example.org/report"})
    assert stub_text(call, 12400) == (
        '[result elided to save context: web_fetch(url="https://example.org/report"), '
        '12,400 tokens. Call the tool again if you need it.]')


def test_stub_args_cut_to_120():
    call = ToolCall(id="c1", name="read", arguments={"path": "x" * 500})
    inside = stub_text(call, 10).split("read(", 1)[1].split("), ", 1)[0]
    assert len(inside) <= 121 and inside.endswith("…")


def test_stubbed_position_keeps_tool_call_id(history_with_tools):
    view = render_view("SYS", history_with_tools, CompactionState(stubbed=[2]))
    assert view[3].role == "tool"
    assert view[3].tool_call_id == history_with_tools[2].tool_call_id
    assert view[3].content.startswith("[result elided to save context: web_fetch(")
    check_tool_pairs(view)


def test_summary_prepended_to_first_kept_user(history_three_turns):
    st = CompactionState(summary=Summary(text="S", covers_through=4, turns_covered=2,
                                         model="m", created_at=NOW))
    view = render_view("SYS", history_three_turns, st)
    assert view[1].role == "user"
    assert view[1].content == (summary_header(2) + "S" + "\n[End of summary]\n\n"
                               + history_three_turns[4].content)
    assert len(view) == 1 + len(history_three_turns) - 4
    assert summary_header(2) == "[Summary of the earlier conversation, turns 1–2]\n"


def test_clip_marks_tokens_removed():
    out = clip_text("a" * 4000 + "b" * 4000, tokens_removed=1500)
    assert "[… 1,500 tokens removed to save context …]" in out
    assert out.startswith("a") and out.endswith("b")
    assert len(out) < 8000 - 1500 * 4 + 100


def test_check_tool_pairs_rejects_orphan():
    view = [Message.system("S"), Message.user("u"), _call(7)]
    with pytest.raises(ValueError, match="c7"):
        check_tool_pairs(view)


def test_check_tool_pairs_rejects_result_without_call():
    view = [Message.system("S"), Message.user("u"), Message.tool("r", tool_call_id="zz")]
    with pytest.raises(ValueError, match="zz"):
        check_tool_pairs(view)


def test_estimate_uses_last_prompt_tokens():
    st = CompactionState(last_prompt_tokens=5000, last_view_chars=20000)
    assert estimate_tokens(24000, st) == 6000
    assert estimate_tokens(12000, st) == 3000
    assert estimate_tokens(8000, CompactionState()) == 2000
