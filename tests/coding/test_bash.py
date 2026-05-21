import pytest
from lovelaice.coding.tools.bash import bash


@pytest.mark.asyncio
async def test_bash_runs_command():
    out = await bash.run(command="echo hi")
    assert "hi" in out
    assert "exit code: 0" in out


@pytest.mark.asyncio
async def test_bash_captures_stderr_and_nonzero_exit():
    out = await bash.run(command="echo err 1>&2; exit 2")
    assert "err" in out
    assert "exit code: 2" in out


@pytest.mark.asyncio
async def test_bash_timeout(monkeypatch):
    """Patch the timeout down so this test runs fast."""
    monkeypatch.setattr("lovelaice.coding.tools.bash.DEFAULT_TIMEOUT", 1)
    out = await bash.run(command="sleep 5")
    assert "timeout" in out.lower()
