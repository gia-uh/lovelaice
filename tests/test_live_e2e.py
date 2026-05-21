import os
import sys
import subprocess
from pathlib import Path
import pytest


@pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="requires OPENROUTER_API_KEY for live LLM",
)
def test_lovelaice_list_files_e2e(tmp_path):
    """End-to-end verification per spec §7 VS1 — `lovelaice "list files"`
    routes a real LLM through the new engine + in-process ACP client +
    coding host and produces a non-empty response."""
    repo_root = Path(__file__).parent.parent
    env = os.environ.copy()
    env["LOVELAICE_MODEL"] = "anthropic/claude-haiku-4-5"
    proc = subprocess.run(
        [sys.executable, "-m", "lovelaice.cli",
         "--session-path", str(tmp_path / "s.jsonl"),
         "--cwd", str(repo_root),
         "list the files in src/lovelaice/agent/ using bash"],
        capture_output=True, text=True, timeout=120, env=env,
    )
    assert proc.returncode == 0, (
        f"stdout={proc.stdout}\nstderr={proc.stderr}")
    # Output should mention some agent/ file.
    assert any(
        name in proc.stdout
        for name in ("agent.py", "harness.py", "session.py")
    ), f"got stdout: {proc.stdout!r}"
