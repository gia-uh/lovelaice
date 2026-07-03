"""Hard tool-calling smoke: provoke real validation failures on qwen3.5-9b and
watch the one-shot repair heal them.

The hard cases use nested-model / list-of-model / code-enum params. lingo's wire
schema under-describes these (a BaseModel param serializes as "string"; a
list[Model] as an untyped array), so a 9B emits strings/wrong tokens that FAIL
pydantic validation. The repair shot is handed the FULL expanded arg schema, so
it can reconstruct the object — the scenario where repair earns its keep.

Run against LOCAL lingo:
    cd repos/lovelaice
    OPENROUTER_API_KEY=$(cat /home/apiad/Workspace/.claude/openrouter.token) \
      PYTHONPATH=/home/apiad/Workspace/repos/lingo \
      uv run --no-sync python scripts/smoke_repair_hard.py
"""
from __future__ import annotations

import asyncio
import copy
import json
import os
import tempfile
from pathlib import Path
from typing import Literal

import lingo
from lingo.tools import tool as lingo_tool
from pydantic import BaseModel

from lovelaice.agent import Agent, AgentConfig, AgentTool
from lovelaice.agent.loops.react_native import ReActNative
from lovelaice.agent.tools import ToolRegistry, validate_args
from lovelaice.agent.events import (
    AssistantMessageFinalized, ToolCallRepaired, ToolExecutionEnd,
)

MODEL = "qwen/qwen3.5-9b"
BASE_URL = "https://openrouter.ai/api/v1"
SYSTEM = ("You are an operations agent. Use the provided tools to fulfill each "
          "request. Call exactly one tool with well-formed arguments.")


# --- hard tools -------------------------------------------------------------

class EventSpec(BaseModel):
    title: str
    date: str                      # YYYY-MM-DD
    attendees: list[str]
    priority: Literal["low", "medium", "high"]


@lingo_tool
async def create_event(spec: EventSpec) -> str:
    """Create a calendar event from a structured spec.

    Args:
        spec: The event object: title, date (YYYY-MM-DD), attendees (names),
            and priority (low/medium/high).
    """
    ev = spec if isinstance(spec, EventSpec) else EventSpec(**spec)
    return (f"event '{ev.title}' on {ev.date}, {ev.priority} priority, "
            f"{len(ev.attendees)} attendees")


class LineItem(BaseModel):
    description: str
    quantity: int
    unit_price: float


@lingo_tool
async def create_invoice(customer: str, items: list[LineItem]) -> str:
    """Create an invoice for a customer.

    Args:
        customer: Customer name.
        items: Line items, each with description, quantity, unit_price.
    """
    rows = [i if isinstance(i, LineItem) else LineItem(**i) for i in items]
    total = sum(i.quantity * i.unit_price for i in rows)
    return f"invoice for {customer}: {len(rows)} items, total {total:.2f}"


@lingo_tool
async def triage(summary: str, severity: Literal["SEV1", "SEV2", "SEV3"]) -> str:
    """File an incident at a severity code.

    Args:
        summary: One-line incident summary.
        severity: Severity CODE — exactly SEV1 (highest), SEV2, or SEV3.
    """
    return f"filed [{severity}] {summary}"


CASES = [
    (create_event, "create_event",
     "Schedule an event titled 'Roadmap sync' on 2026-07-10 with Ana and Beto, "
     "high priority."),
    (create_invoice, "create_invoice",
     "Invoice ACME Corp for 3 widgets at $4.50 each and 2 gadgets at $10.00 each."),
    (triage, "triage",
     "The production database is completely down — this is a total outage, "
     "absolutely critical."),
]


async def run(*, repair: bool, api_key: str) -> dict:
    sess = Path(tempfile.mkdtemp(prefix="hard-"))
    totals = {"invalid_first": 0, "repaired": 0, "healed": 0, "e2e": 0}
    for i, (fn, name, prompt) in enumerate(CASES):
        tool = AgentTool(inner=fn)
        reg = ToolRegistry()
        reg.register(tool)
        cfg = AgentConfig(model=MODEL, system_prompt=SYSTEM, api_key=api_key,
                          base_url=BASE_URL, max_tokens=800,
                          repair_tool_calls=repair, repair_context="turn")
        agent = Agent(config=cfg, tools=[tool], loop=ReActNative(),
                      session_path=sess / f"{name}-{repair}-{i}.jsonl")

        st = {"first_args": None, "first_invalid": False,
              "orig": None, "fixed": None, "ok": False}

        def on_event(ev, st=st, reg=reg):
            if isinstance(ev, AssistantMessageFinalized):
                for tc in (ev.message.tool_calls or []):
                    if st["first_args"] is None:
                        st["first_args"] = copy.deepcopy(tc.arguments or {})
                        t = reg.get(tc.name)
                        st["first_invalid"] = (
                            t is None
                            or isinstance(validate_args(t, st["first_args"]), str))
            elif isinstance(ev, ToolCallRepaired):
                st["orig"], st["fixed"] = ev.original_args, ev.repaired_args
            elif isinstance(ev, ToolExecutionEnd):
                st["ok"] = st["ok"] or not ev.is_error

        agent.subscribe(on_event)
        try:
            await asyncio.wait_for(agent.prompt(prompt), timeout=200)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {name} errored: {type(exc).__name__}: {exc}")

        totals["invalid_first"] += int(st["first_invalid"])
        totals["repaired"] += int(st["orig"] is not None)
        totals["healed"] += int(st["orig"] is not None and st["ok"])
        totals["e2e"] += int(st["ok"])

        print(f"  {name}: first-try {'INVALID' if st['first_invalid'] else 'valid'}"
              f" | repaired={st['orig'] is not None} | exec_ok={st['ok']}")
        print(f"      first args : {json.dumps(st['first_args'], default=str)[:160]}")
        if st["orig"] is not None:
            print(f"      repaired-> : {json.dumps(st['fixed'], default=str)[:160]}")
    return totals


async def main() -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("set OPENROUTER_API_KEY")
    print(f"lingo: {lingo.__file__}\nmodel: {MODEL}\n")
    print("== repair OFF ==")
    off = await run(repair=False, api_key=api_key)
    print("\n== repair ON ==")
    on = await run(repair=True, api_key=api_key)
    n = len(CASES)
    print(f"\n--- summary ({n} hard cases) ---")
    print(f"  repair OFF: first-try-invalid {off['invalid_first']}/{n}, "
          f"e2e-ok {off['e2e']}/{n}")
    print(f"  repair ON : first-try-invalid {on['invalid_first']}/{n}, "
          f"repaired {on['repaired']}, healed {on['healed']}, e2e-ok {on['e2e']}/{n}")


if __name__ == "__main__":
    asyncio.run(main())
