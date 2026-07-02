import json
import pytest
from lingo.llm import Message, ToolCall
from lovelaice.agent.session import Session


def test_session_writes_header_on_create(tmp_path):
    path = tmp_path / "s.jsonl"
    sess = Session.create(path, model="x", system_prompt_hash="sha256:abc",
                          loop="ReActNative", cwd=str(tmp_path))
    lines = path.read_text().splitlines()
    assert len(lines) == 1
    hdr = json.loads(lines[0])
    assert hdr["type"] == "session_header"
    assert hdr["schema_version"] == 1
    assert hdr["model"] == "x"
    assert hdr["system_prompt_hash"] == "sha256:abc"
    assert hdr["loop"] == "ReActNative"


def test_session_append_user_message(tmp_path):
    path = tmp_path / "s.jsonl"
    sess = Session.create(path, model="x", system_prompt_hash="h",
                          loop="ReActNative", cwd=str(tmp_path))
    sess.append(Message.user("hello"))
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    msg = json.loads(lines[1])
    assert msg["type"] == "message"
    assert msg["role"] == "user"
    assert msg["content"] == "hello"


def test_session_round_trip_5_turns(tmp_path):
    """Canary #3 — write a mixed session, reopen, assert messages_for_llm reconstructs."""
    path = tmp_path / "s.jsonl"
    sess = Session.create(path, model="x", system_prompt_hash="h",
                          loop="ReActNative", cwd=str(tmp_path))

    tc = ToolCall(id="c1", name="read", arguments={"path": "a"})
    sess.append(Message.user("hi"))
    sess.append(Message.assistant("thinking", tool_calls=[tc],
                                  thinking="reason", stop_reason="tool_calls"))
    sess.append(Message.tool("foo"))  # tool-role message; content is the tool output
    sess.append(Message.assistant("done", stop_reason="stop"))
    sess.append(Message.user("ok"))

    # Reopen
    sess2 = Session.load(path)
    msgs = sess2.messages_for_llm(system_prompt="SYS")
    assert msgs[0].role == "system"
    assert msgs[0].content == "SYS"
    assert [m.role for m in msgs[1:]] == ["user", "assistant", "tool",
                                          "assistant", "user"]
    assert msgs[2].tool_calls and msgs[2].tool_calls[0].name == "read"
    assert msgs[2].thinking == "reason"
    assert msgs[2].stop_reason == "tool_calls"


def test_session_skips_malformed_trailing_line(tmp_path):
    path = tmp_path / "s.jsonl"
    sess = Session.create(path, model="x", system_prompt_hash="h",
                          loop="ReActNative", cwd=str(tmp_path))
    sess.append(Message.user("ok"))
    # Simulate crash mid-write: append a partial line.
    with path.open("a") as f:
        f.write('{"type": "message", "rol')  # no newline, no closing brace
    # Loader must skip it without crashing.
    sess2 = Session.load(path)
    msgs = sess2.messages_for_llm(system_prompt="x")
    assert len(msgs) == 2  # system + user


def test_session_load_rejects_missing_header(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"type": "message", "role": "user", "content": "x"}\n')
    with pytest.raises(ValueError, match="header"):
        Session.load(path)


def test_session_load_rejects_unsupported_schema_version(tmp_path):
    path = tmp_path / "v2.jsonl"
    path.write_text(json.dumps({
        "type": "session_header",
        "id": "abc",
        "schema_version": 99,
        "cwd": "/x", "model": "m", "system_prompt_hash": "h",
        "loop": "ReActNative", "created_at": "2026-05-21T00:00:00Z",
    }) + "\n")
    with pytest.raises(ValueError, match="schema_version"):
        Session.load(path)


def test_session_preserves_empty_string_tool_call_id(tmp_path):
    """tool_call_id="" must survive the serialize→deserialize round-trip.

    When a provider doesn't send an ID (lingo falls back to ""), the falsy
    check 'if msg.tool_call_id' dropped the field from the JSONL entry.
    On reload, tool_call_id became None, and model_dump() omitted it from the
    wire message — causing API errors on the second LLM call. Regression for
    the 'agent cuts after first tool call with small models' bug."""
    path = tmp_path / "s.jsonl"
    sess = Session.create(path, model="x", system_prompt_hash="h",
                          loop="ReActNative", cwd=str(tmp_path))

    tc = ToolCall(id="", name="bash", arguments={"cmd": "ls"})
    sess.append(Message.assistant("", tool_calls=[tc], stop_reason="tool_calls"))
    sess.append(Message.tool("total 0", tool_call_id=""))

    sess2 = Session.load(path)
    msgs = sess2.messages_for_llm("SYS")
    tool_msg = msgs[2]
    assert tool_msg.role == "tool"
    assert tool_msg.tool_call_id == "", (
        "empty-string tool_call_id must survive JSONL round-trip; "
        f"got {tool_msg.tool_call_id!r}"
    )
    dump = tool_msg.model_dump()
    assert "tool_call_id" in dump, (
        "model_dump() must include tool_call_id for tool messages even if empty"
    )
    assert dump["tool_call_id"] == ""


def test_session_hash_system_prompt():
    h = Session.hash_system_prompt("hello world")
    assert h.startswith("sha256:")
    assert len(h) == len("sha256:") + 64  # 64 hex chars


def test_session_turn_count(tmp_path):
    path = tmp_path / "s.jsonl"
    sess = Session.create(path, model="x", system_prompt_hash="h",
                          loop="ReActNative", cwd=str(tmp_path))
    assert sess.turn_count == 0
    sess.append(Message.user("hi"))
    assert sess.turn_count == 1
    sess.append(Message.assistant("ok"))
    assert sess.turn_count == 1  # only user messages count
    sess.append(Message.user("again"))
    assert sess.turn_count == 2


def test_update_tool_call_args_rewrites_entry(tmp_path):
    s = Session.create(tmp_path / "s.jsonl", model="m",
                       system_prompt_hash="h", loop="L", cwd=".")
    s.append(Message.assistant("", tool_calls=[
        ToolCall(id="c1", name="grep", arguments={"pattern": "x"})]))
    s.update_tool_call_args("c1", {"pattern": "x", "path": "."})
    msgs = s.messages_for_llm("sys")
    assert msgs[-1].tool_calls[0].arguments == {"pattern": "x", "path": "."}


def test_update_tool_call_args_unknown_id_is_noop(tmp_path):
    s = Session.create(tmp_path / "s.jsonl", model="m",
                       system_prompt_hash="h", loop="L", cwd=".")
    s.append(Message.assistant("", tool_calls=[
        ToolCall(id="c1", name="grep", arguments={"pattern": "x"})]))
    s.update_tool_call_args("nope", {"pattern": "y"})
    assert s.messages_for_llm("sys")[-1].tool_calls[0].arguments == {"pattern": "x"}
