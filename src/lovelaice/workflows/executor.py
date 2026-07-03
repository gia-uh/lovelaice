"""Async executor for workflow specs.

Drives lovelaice's ``Agent`` for ``agent`` nodes and threads named outputs
through ``sequence`` nodes. Callers supply an ``agent_factory`` (the same seam
``AcpServer(agent_factory=...)`` uses) so the executor stays decoupled from how
agents are built and what tools/MCP they carry.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Awaitable, Callable

from lovelaice.workflows.models import (
    AgentNode,
    MapNode,
    Node,
    ParallelNode,
    PromptNode,
    SequenceNode,
    ToolNode,
    WorkflowSpec,
)

Handler = Callable[[dict, dict], Awaitable[Any]]

MAX_CONCURRENCY = 4

_RAW_VAR = re.compile(r"^\{(\w+)\}$")


_TOKEN = re.compile(r"\{(\w+)\}")


def _render_template(text: str, vars: dict) -> str:
    """Substitute ``{var}`` tokens for known vars, leaving every other brace
    (e.g. literal JSON in a prompt) untouched. Not ``str.format`` — that treats
    JSON braces as nested format fields and blows the recursion limit."""

    def repl(m: "re.Match") -> str:
        key = m.group(1)
        return str(vars[key]) if key in vars else m.group(0)

    return _TOKEN.sub(repl, text)


def _render_args(args: dict, vars: dict) -> dict:
    out: dict = {}
    for k, v in args.items():
        if isinstance(v, str):
            m = _RAW_VAR.match(v)
            if m and m.group(1) in vars:
                out[k] = vars[m.group(1)]  # preserve raw object (dict/list/...)
            else:
                out[k] = _render_template(v, vars)
        else:
            out[k] = v
    return out


def _final_text(agent: Any) -> str:
    """Return the last assistant message's text after a completed turn."""
    for m in reversed(agent.messages_for_llm()):
        if getattr(m, "role", None) == "assistant":
            return m.content or ""
    return ""


class _NullAsem:
    """Async no-op context — used when no concurrency semaphore is in ctx."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


_NULL_ASEM = _NullAsem()


def _sem(ctx: dict):
    """The concurrency limiter for leaf work; a no-op if none was seeded."""
    return ctx.get("sem") or _NULL_ASEM


async def _run_agent(node: AgentNode, ctx: dict, agent_factory: Callable[[], Any]) -> dict:
    prompt = _render_template(node.prompt, ctx["vars"])
    agent = agent_factory()
    async with _sem(ctx):
        await agent.prompt(prompt)
    text = _final_text(agent)
    if node.output_schema is not None:
        value = json.loads(text)
        if node.name is not None:
            ctx["vars"][node.name] = value  # raw object, for later {name} passthrough
        return value
    if node.name is not None:
        ctx["vars"][node.name] = text
    return {"text": text}


async def _run_tool(node: ToolNode, ctx: dict) -> dict:
    handler = ctx["handlers"].get(node.tool)
    if handler is None:
        raise KeyError(f"no handler registered for tool: {node.tool!r}")
    args = _render_args(node.args, ctx["vars"])
    async with _sem(ctx):
        result = await handler(args, ctx["vars"])
    if node.name is not None:
        ctx["vars"][node.name] = result
    return {"tool": node.tool, "result": result}


async def _run_prompt(node: PromptNode, ctx: dict) -> dict:
    """Run a prompt against the host's live/primary agent via ``prompt_handler``.

    Raises if no handler was supplied — a ``prompt`` node has no meaning without
    a live agent to run it on (e.g. a headless/scheduled run) — or if reached
    inside a fan-out (it would race the single live agent).
    """
    if ctx.get("in_fanout"):
        raise RuntimeError(
            "a 'prompt' node cannot run inside a parallel/map fan-out (it would "
            "race the single live conversation agent) — use an 'agent' node instead"
        )
    handler = ctx.get("prompt_handler")
    if handler is None:
        raise RuntimeError(
            "prompt_handler is required to run a 'prompt' node "
            "(no live/primary agent available — is this a headless run?)"
        )
    prompt = _render_template(node.prompt, ctx["vars"])
    async with _sem(ctx):
        text = await handler(prompt, ctx["vars"])
    if node.name is not None:
        ctx["vars"][node.name] = text
    return {"text": text}


def _child_ctx(ctx: dict, extra_vars: dict | None = None) -> dict:
    """A fan-out branch's ctx: an ISOLATED vars copy (so siblings never race),
    flagged in_fanout (so a prompt node inside raises)."""
    return {**ctx, "vars": {**ctx["vars"], **(extra_vars or {})}, "in_fanout": True}


async def _run_parallel(node: ParallelNode, ctx: dict, agent_factory: Callable[[], Any]) -> dict:
    async def run_child(child):
        return await _run_node(child, _child_ctx(ctx), agent_factory)

    results = list(await asyncio.gather(*[run_child(c) for c in node.children]))
    if node.name is not None:
        ctx["vars"][node.name] = results
    return {"items": results}


async def _run_map(node: MapNode, ctx: dict, agent_factory: Callable[[], Any]) -> dict:
    items = ctx["vars"].get(node.over)
    if not isinstance(items, list):
        raise RuntimeError(
            f"map 'over' must name a list var; vars[{node.over!r}] is "
            f"{type(items).__name__}"
        )

    async def run_item(el):
        return await _run_node(node.node, _child_ctx(ctx, {node.as_: el}), agent_factory)

    results = list(await asyncio.gather(*[run_item(x) for x in items]))
    if node.name is not None:
        ctx["vars"][node.name] = results
    return {"items": results}


async def _run_node(node: Node, ctx: dict, agent_factory: Callable[[], Any]) -> dict:
    if isinstance(node, AgentNode):
        return await _run_agent(node, ctx, agent_factory)
    if isinstance(node, PromptNode):
        return await _run_prompt(node, ctx)
    if isinstance(node, ToolNode):
        return await _run_tool(node, ctx)
    if isinstance(node, ParallelNode):
        return await _run_parallel(node, ctx, agent_factory)
    if isinstance(node, MapNode):
        return await _run_map(node, ctx, agent_factory)
    if isinstance(node, SequenceNode):
        result: dict = {}
        for child in node.children:
            result = await _run_node(child, ctx, agent_factory)
        return result
    raise TypeError(f"unknown node: {node!r}")


async def run(
    spec: WorkflowSpec,
    *,
    agent_factory: Callable[[], Any],
    handlers: dict[str, Handler] | None = None,
    inputs: dict | None = None,
    prompt_handler: "Callable[[str, dict], Awaitable[str]] | None" = None,
) -> dict:
    """Execute a workflow spec and return the root node's result dict.

    ``handlers`` maps a ``tool`` name to ``async (args, vars) -> Any`` — the host
    provides these (e.g. bridged from an agent's MCP tools). The engine itself
    ships none, keeping it decoupled from any concrete tool.

    ``prompt_handler`` is ``async (prompt, vars) -> str`` — the host runs the
    prompt on its live/primary agent (shared context). Required only if the spec
    contains a ``prompt`` node.
    """
    ctx = {
        "vars": dict(inputs or {}),
        "handlers": handlers or {},
        "prompt_handler": prompt_handler,
        "sem": asyncio.Semaphore(MAX_CONCURRENCY),
        "in_fanout": False,
    }
    return await _run_node(spec.root, ctx, agent_factory)
