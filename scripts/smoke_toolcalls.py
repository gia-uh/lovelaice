"""Live tool-calling smoke against a real 9B (qwen/qwen3.5-9b via OpenRouter).

NOT part of CI — spends real tokens. Measures, per config:
  - first-try validation rate: fraction of emitted tool calls whose args pass
    pydantic validation on the model's FIRST emission (snapshotted before any
    repair mutates them);
  - repair success rate: of first-try-invalid calls, the fraction that end up
    executing successfully (via the one-shot repair);
  - end-to-end success: the expected tool ran without error.

Run against LOCAL lingo (with the enriched schema builder):
    cd repos/lovelaice
    OPENROUTER_API_KEY=$(cat /home/apiad/Workspace/.claude/openrouter.token) \
      PYTHONPATH=/home/apiad/Workspace/repos/lingo \
      uv run --no-sync python scripts/smoke_toolcalls.py
"""
from __future__ import annotations

import asyncio
import copy
import os
import tempfile
from pathlib import Path
from typing import Literal

import lingo
import lingo.llm as _llm
from lingo.tools import tool as lingo_tool
from pydantic import BaseModel

from lovelaice.agent import Agent, AgentConfig, AgentTool
from lovelaice.agent.loops.react_native import ReActNative
from lovelaice.agent.tools import ToolRegistry, validate_args
from lovelaice.agent.events import (
    AssistantMessageFinalized, ToolCallRepaired, ToolExecutionEnd,
)
from lovelaice.tools import read, write, list_, glob, grep

_REAL_SCHEMA = _llm.tool_to_openai_schema


def _flat_schema(tool_obj) -> dict:
    """Old-main behavior: flat types, no descriptions, everything required."""
    params = tool_obj.parameters()

    def flat(t):
        import typing
        if t is int:
            return {"type": "integer"}
        if t is float:
            return {"type": "number"}
        if t is bool:
            return {"type": "boolean"}
        if t is list or typing.get_origin(t) is list:
            return {"type": "array"}
        if t is dict or typing.get_origin(t) is dict:
            return {"type": "object"}
        return {"type": "string"}

    return {"type": "function", "function": {
        "name": tool_obj.name,
        "description": tool_obj.description.strip(),
        "parameters": {"type": "object",
                       "properties": {n: flat(t) for n, t in params.items()},
                       "required": list(params.keys())}}}

MODEL = "qwen/qwen3.5-9b"
BASE_URL = "https://openrouter.ai/api/v1"

SYSTEM = (
    "You are a coding agent operating in a workspace. Use the provided tools to "
    "answer. Call a tool when the task needs file access; do not guess file "
    "contents. When you have the answer, reply in one short sentence."
)

# (prompt, expected tool name)
FIXTURE = [
    ("Read the file config.txt and tell me the port number.", "read"),
    ("List the files in the notes directory.", "list_"),
    ("Search the whole project for the word TODO.", "grep"),
    ("Find all markdown files in the project.", "glob"),
    ("Read the note at notes/todo.md.", "read"),
    ("Create a file called hello.txt with the content 'hi there'.", "write"),
    ("Search for the word 'port' only inside config.txt.", "grep"),
    ("List everything in the current directory.", "list_"),
]


def _seed(root: Path) -> None:
    (root / "config.txt").write_text("host=localhost\nport=8080\n")
    (root / "README.md").write_text("# Project\nSee notes.\n")
    notes = root / "notes"
    notes.mkdir()
    (notes / "todo.md").write_text("# Todo\n- TODO: wire the API\n- done: init\n")
    (notes / "ideas.md").write_text("random idea\n")


def _tools() -> list[AgentTool]:
    return [AgentTool(inner=lingo_tool(fn))
            for fn in (read, write, list_, glob, grep)]


def _registry(tools: list[AgentTool]) -> ToolRegistry:
    reg = ToolRegistry()
    for t in tools:
        reg.register(t)
    return reg


async def run_config(*, repair: bool, root: Path, api_key: str,
                     flat: bool = False) -> dict:
    _llm.tool_to_openai_schema = _flat_schema if flat else _REAL_SCHEMA
    tools = _tools()
    reg = _registry(tools)
    totals = {"emitted": 0, "first_try_valid": 0, "invalid": 0,
              "repaired_ok": 0, "e2e_success": 0, "correct_tool": 0,
              "no_toolcall": 0}

    sess_dir = Path(tempfile.mkdtemp(prefix="smoke-sess-"))
    for i, (prompt, expected) in enumerate(FIXTURE):
        os.chdir(root)
        cfg = AgentConfig(
            model=MODEL, system_prompt=SYSTEM, api_key=api_key, base_url=BASE_URL,
            cwd=str(root), max_tokens=1024,
            repair_tool_calls=repair, repair_context="turn",
        )
        spath = sess_dir / f"s-{'on' if repair else 'off'}-{i}.jsonl"
        agent = Agent(config=cfg, tools=tools, loop=ReActNative(),
                      session_path=spath)

        snap_valid: list[bool] = []      # first-try validity per emitted call
        emitted_names: list[str] = []
        repaired_ids: set[str] = set()
        exec_ok = False

        def on_event(ev):
            nonlocal exec_ok
            if isinstance(ev, AssistantMessageFinalized):
                msg = ev.message
                for tc in (msg.tool_calls or []):
                    args = copy.deepcopy(tc.arguments or {})
                    emitted_names.append(tc.name)
                    tool = reg.get(tc.name)
                    if tool is None:
                        snap_valid.append(False)
                        continue
                    snap_valid.append(not isinstance(validate_args(tool, args), str))
            elif isinstance(ev, ToolCallRepaired):
                repaired_ids.add(ev.call_id)
            elif isinstance(ev, ToolExecutionEnd):
                if not ev.is_error:
                    exec_ok = True

        agent.subscribe(on_event)
        try:
            await asyncio.wait_for(agent.prompt(prompt), timeout=90)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {expected:6} prompt errored: {type(exc).__name__}: {exc}")

        n = len(snap_valid)
        if n == 0:
            totals["no_toolcall"] += 1
        totals["emitted"] += n
        totals["first_try_valid"] += sum(snap_valid)
        totals["invalid"] += sum(1 for v in snap_valid if not v)
        totals["repaired_ok"] += len(repaired_ids) if exec_ok else 0
        if exec_ok:
            totals["e2e_success"] += 1
        if expected in emitted_names:
            totals["correct_tool"] += 1

        mark = "ok " if exec_ok else "MISS"
        print(f"  [{mark}] want={expected:6} emitted={emitted_names} "
              f"first_try_valid={sum(snap_valid)}/{n} repaired={len(repaired_ids)}")

    return totals


def _pct(a: int, b: int) -> str:
    return f"{(100*a/b):.0f}%" if b else "n/a"


# --- repair provocation: an enum tool a small model tends to violate ---

@lingo_tool
async def set_status(task: str, status: Literal["open", "closed"]) -> str:
    """Set a task's status.

    Args:
        task: The task description to update.
        status: The new status — must be exactly "open" or "closed".
    """
    return f"{task} -> {status}"


async def run_repair_probe(*, api_key: str) -> None:
    """Prompt the model in a way that likely yields an out-of-enum status
    (e.g. 'finished'), so the arg fails validation and repair must heal it."""
    _llm.tool_to_openai_schema = _REAL_SCHEMA
    tool = AgentTool(inner=set_status)
    reg = _registry([tool])
    prompts = [
        "Mark the task 'wire the API' as finished.",
        "The task 'write the docs' is now complete — update its status.",
        "Reopen the task 'wire the API'.",
    ]
    for repair in (False, True):
        print(f"\n== repair probe (enum tool), repair {'ON' if repair else 'OFF'} ==")
        sess_dir = Path(tempfile.mkdtemp(prefix="smoke-probe-"))
        healed = invalid = e2e = 0
        for i, p in enumerate(prompts):
            cfg = AgentConfig(model=MODEL, system_prompt=SYSTEM, api_key=api_key,
                              base_url=BASE_URL, max_tokens=512,
                              repair_tool_calls=repair, repair_context="turn")
            agent = Agent(config=cfg, tools=[tool], loop=ReActNative(),
                          session_path=sess_dir / f"p-{repair}-{i}.jsonl")
            first_invalid = [False]
            repaired = [False]
            ok = [False]

            def on_event(ev, fi=first_invalid, rp=repaired, okr=ok):
                if isinstance(ev, AssistantMessageFinalized):
                    for tc in (ev.message.tool_calls or []):
                        args = copy.deepcopy(tc.arguments or {})
                        t = reg.get(tc.name)
                        if t and isinstance(validate_args(t, args), str):
                            fi[0] = True
                elif isinstance(ev, ToolCallRepaired):
                    rp[0] = True
                elif isinstance(ev, ToolExecutionEnd):
                    if not ev.is_error:
                        okr[0] = True

            agent.subscribe(on_event)
            try:
                await asyncio.wait_for(agent.prompt(p), timeout=90)
            except Exception as exc:  # noqa: BLE001
                print(f"  ! errored: {type(exc).__name__}: {exc}")
            invalid += int(first_invalid[0])
            healed += int(repaired[0] and ok[0])
            e2e += int(ok[0])
            print(f"  [{'ok ' if ok[0] else 'MISS'}] first_try_invalid={first_invalid[0]} "
                  f"repaired={repaired[0]} exec_ok={ok[0]}")
        print(f"  -> first-try invalid: {invalid}/{len(prompts)}  "
              f"healed-by-repair: {healed}  e2e ok: {e2e}/{len(prompts)}")


async def check_structured_output(api_key: str) -> None:
    """Confirm OpenRouter-qwen supports the forced-JSON path repair relies on."""
    class Fix(BaseModel):
        pattern: str
        path: str

    llm = lingo.LLM(model=MODEL, api_key=api_key, base_url=BASE_URL)
    msgs = [_llm.Message.user(
        'Return JSON with pattern="TODO" and path=".".')]
    try:
        out = await llm.create(Fix, msgs)
        print(f"structured-output (llm.create) on {MODEL}: OK -> {out.model_dump()}")
    except Exception as exc:  # noqa: BLE001
        print(f"structured-output on {MODEL}: FAILED -> {type(exc).__name__}: {exc}")


async def main() -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("set OPENROUTER_API_KEY")
    print(f"lingo from: {lingo.__file__}")
    print(f"model: {MODEL}\n")

    root = Path(tempfile.mkdtemp(prefix="smoke-ws-"))
    _seed(root)

    await check_structured_output(api_key)

    print("\n== config A: FLAT schemas (old main), repair OFF ==")
    flat = await run_config(repair=False, root=root, api_key=api_key, flat=True)
    print("\n== config B: enriched schemas, repair OFF ==")
    off = await run_config(repair=False, root=root, api_key=api_key)
    print("\n== config C: enriched schemas, repair ON ==")
    on = await run_config(repair=True, root=root, api_key=api_key)

    # Repair provocation with an enum tool (built-ins are too easy to fumble).
    await run_repair_probe(api_key=api_key)

    for label, t in (("FLAT schemas (baseline)", flat),
                     ("enriched, repair OFF", off),
                     ("enriched, repair ON", on)):
        print(f"\n--- {label} ---")
        print(f"  tool calls emitted:     {t['emitted']}")
        print(f"  first-try valid:        {t['first_try_valid']}/{t['emitted']} "
              f"({_pct(t['first_try_valid'], t['emitted'])})")
        print(f"  first-try invalid:      {t['invalid']}")
        print(f"  repaired & executed ok: {t['repaired_ok']}")
        print(f"  repair success rate:    {_pct(t['repaired_ok'], t['invalid'])}")
        print(f"  correct tool chosen:    {t['correct_tool']}/{len(FIXTURE)}")
        print(f"  end-to-end success:     {t['e2e_success']}/{len(FIXTURE)}")
        print(f"  turns with no toolcall: {t['no_toolcall']}")


if __name__ == "__main__":
    asyncio.run(main())
