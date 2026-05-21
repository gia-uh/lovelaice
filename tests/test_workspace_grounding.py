"""Workspace grounding: chdir to .lovelaice.py's directory before agent runs."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from lovelaice.config import find_config_file


def test_find_config_file_walks_upward(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Given .lovelaice.py at the root and cwd in a deep subdir, find_config_file
    returns the root config."""
    (tmp_path / ".lovelaice.py").write_text("# stub\n")
    deep = tmp_path / "a" / "b" / "c"
    deep.mkdir(parents=True)
    monkeypatch.chdir(deep)

    found = find_config_file()
    assert found is not None
    assert found.resolve() == (tmp_path / ".lovelaice.py").resolve()


def test_find_config_file_returns_none_when_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No .lovelaice.py anywhere up the tree → returns None."""
    deep = tmp_path / "a" / "b"
    deep.mkdir(parents=True)
    monkeypatch.chdir(deep)

    assert find_config_file(start_path=deep) is None


@pytest.mark.skip("legacy CLI removed in VS1; will be cleaned up in VS2")
def test_cli_exits_when_no_config(tmp_path: Path) -> None:
    """Running `lovelaice` from a directory with no ancestor .lovelaice.py
    exits 1 with a helpful message."""
    result = subprocess.run(
        [sys.executable, "-m", "lovelaice", "hello"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "No .lovelaice.py" in result.stderr
