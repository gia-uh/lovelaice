import pytest

from lovelaice.coding.tools.write import write
from lovelaice.coding.tools.edit import edit
from lovelaice.coding.tools.glob import glob
from lovelaice.coding.tools.list_dir import list_dir


@pytest.mark.asyncio
async def test_write_edit_glob_list(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    await write.run(path="sub/a.txt", content="hello world")
    assert (tmp_path / "sub/a.txt").read_text() == "hello world"
    await edit.run(path="sub/a.txt", old="world", new="there")
    assert (tmp_path / "sub/a.txt").read_text() == "hello there"
    names = await list_dir.run(path="sub")
    assert "a.txt" in names
    matches = await glob.run(pattern="sub/*.txt")
    assert "sub/a.txt" in matches
    assert {write.name, edit.name, glob.name, list_dir.name} == {
        "write", "edit", "glob", "list_dir"}


def test_host_wires_full_toolset(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    from lovelaice.coding.host import create_coding_agent
    agent = create_coding_agent(
        model="fake/model", session_path=tmp_path / "s.jsonl", cwd=str(tmp_path))
    names = {t.name for t in agent.harness.tools.all()}
    assert {"read", "bash", "write", "edit", "glob", "list_dir"} <= names


def test_path_guard_blocks_list_dir_outside_cwd():
    from lovelaice.coding.hooks import path_guard

    class Call:
        name = "list_dir"
        arguments = {"path": "/etc"}

    assert path_guard(Call(), cwd="/tmp") is not None
