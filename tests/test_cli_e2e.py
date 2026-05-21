"""End-to-end CLI test: drive the `lovelaice` binary against a fake
OpenRouter server, verify reasoning fragments and reply content both
reach the user-facing output.

This is the only test in the suite that spawns the CLI as a subprocess
and exercises the full stack: typer entry → oneshot → Lovelaice.chat()
→ Lingo flow → lingo.LLM.chat() → openai SDK streaming → fake
OpenRouter → reasoning passthrough → output renderer.

The custom `.lovelaice.py` registers a single streaming command (no
ReAct, no structured output), so the fake server only needs to handle
streaming `/chat/completions`. That keeps the fake small while still
proving the user-visible reasoning flow works.

Output assertions go through `--json` and `--plain` modes — they're
unambiguously machine-parseable. The Rich panel renderer is exercised
indirectly via the same plumbing; testing its exact box-drawing output
would be brittle.
"""
from __future__ import annotations

import pytest
pytest.skip("legacy CLI removed in VS1; will be cleaned up in VS2", allow_module_level=True)

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

from tests._fake_openrouter import (
    DEFAULT_CONTENT_CHUNKS,
    DEFAULT_REASONING_CHUNKS,
    FakeServer,
)


def _write_config(tmp_path: Path, base_url: str) -> Path:
    """Write a `.lovelaice.py` that points at the fake server and registers
    a single streaming command (no ReAct, no tool use)."""
    cfg = tmp_path / ".lovelaice.py"
    cfg.write_text(
        textwrap.dedent(
            f"""
            from lovelaice import Config

            MODELS = {{
                "test": {{
                    "model": "fake-thinking-model",
                    "api_key": "test-key",
                    "base_url": {base_url!r},
                    "thinking": "high",
                }},
            }}

            config = Config(models=MODELS, prompt="You are a test agent.")

            @config.command
            async def stream_reply(context, engine):
                \"\"\"Single streaming chat — exercises lingo.LLM.chat() directly.\"\"\"
                msg = await engine.reply(context)
                context.append(msg)
                return msg
            """
        ).lstrip()
    )
    return cfg


def _run_lovelaice(cwd: Path, prompt: str, *flags: str) -> subprocess.CompletedProcess:
    """Invoke the CLI as a subprocess via `python -m lovelaice.cli` so we
    don't depend on the `lovelaice` console script being on PATH."""
    args = [sys.executable, "-m", "lovelaice.cli", *flags, prompt]
    return subprocess.run(
        args,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        env={**os.environ, "NO_COLOR": "1"},
        timeout=30,
    )


def test_cli_json_emits_reasoning_then_content_then_done(tmp_path: Path) -> None:
    """`--json` produces a clean NDJSON event stream — the canonical
    machine-parseable mode. Reasoning fragments arrive before content;
    a final `done` event carries the full reply."""
    with FakeServer() as server:
        _write_config(tmp_path, server.base_url)
        proc = _run_lovelaice(tmp_path, "say hi", "--json")

    assert proc.returncode == 0, f"exit {proc.returncode}\nstderr: {proc.stderr}"

    events = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
    types = [e["type"] for e in events]

    # Order matters: every reasoning event arrives before any content event,
    # and `done` is last.
    first_content = types.index("content")
    last_reasoning = max(i for i, t in enumerate(types) if t == "reasoning")
    assert last_reasoning < first_content, f"reasoning/content interleaved: {types}"
    assert types[-1] == "done", f"last event should be done, got {types[-1]}"

    reasoning_deltas = [e["delta"] for e in events if e["type"] == "reasoning"]
    content_deltas = [e["delta"] for e in events if e["type"] == "content"]
    assert reasoning_deltas == DEFAULT_REASONING_CHUNKS
    assert content_deltas == DEFAULT_CONTENT_CHUNKS

    done = events[-1]
    assert done["content"] == "".join(DEFAULT_CONTENT_CHUNKS)


def test_cli_plain_routes_content_to_stdout_and_reasoning_to_stderr(tmp_path: Path) -> None:
    """`--plain` is the pipeline-friendly mode: content tokens stream to
    stdout, reasoning tokens stream to stderr, no Rich markup involved."""
    with FakeServer() as server:
        _write_config(tmp_path, server.base_url)
        proc = _run_lovelaice(tmp_path, "say hi", "--plain")

    assert proc.returncode == 0, f"exit {proc.returncode}\nstderr: {proc.stderr}"

    assert proc.stdout.rstrip("\n") == "".join(DEFAULT_CONTENT_CHUNKS)
    assert proc.stderr.rstrip("\n") == "".join(DEFAULT_REASONING_CHUNKS)


def test_cli_plain_json_mutually_exclusive(tmp_path: Path) -> None:
    """Setting both flags is a config error — fail fast with a non-zero
    exit so misuse doesn't silently degrade to one mode or the other."""
    with FakeServer() as server:
        _write_config(tmp_path, server.base_url)
        proc = _run_lovelaice(tmp_path, "say hi", "--plain", "--json")

    assert proc.returncode != 0
    assert "mutually exclusive" in proc.stderr


def test_cli_quiet_mode_drops_reasoning_to_keep_stdout_clean(tmp_path: Path) -> None:
    """Without any output flag and with stdout piped, reasoning is
    intentionally dropped so stdout carries only the final reply
    (pipe-friendly default behavior)."""
    with FakeServer() as server:
        _write_config(tmp_path, server.base_url)
        proc = _run_lovelaice(tmp_path, "say hi")

    assert proc.returncode == 0, proc.stderr

    full_reply = "".join(DEFAULT_CONTENT_CHUNKS)
    assert full_reply in proc.stdout
    for fragment in DEFAULT_REASONING_CHUNKS:
        assert fragment.strip() not in proc.stdout, (
            f"reasoning leaked into stdout in quiet mode: {fragment!r}\n{proc.stdout}"
        )
