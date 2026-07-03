"""Async executor for workflow specs.

Drives lovelaice's ``Agent`` for ``agent`` nodes and threads named outputs
through ``sequence`` nodes. Callers supply an ``agent_factory`` (the same seam
``AcpServer(agent_factory=...)`` uses) so the executor stays decoupled from how
agents are built and what tools/MCP they carry.
"""

from __future__ import annotations

import json
import re
from typing import Any, Awaitable, Callable

from lovelaice.workflows.models import (
    AgentNode,
    Node,
    PromptNode,
    SequenceNode,
    ToolNode,
    WorkflowSpec,
)

Handler = Callable[[dict, dict], Awaitable[Any]]

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


async def _run_agent(node: AgentNode, ctx: dict, agent_factory: Callable[[], Any]) -> dict:
    prompt = _render_template(node.prompt, ctx["vars"])
    agent = agent_factory()
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
    result = await handler(args, ctx["vars"])
    if node.name is not None:
        ctx["vars"][node.name] = result
    return {"tool": node.tool, "result": result}


async def _run_prompt(node: PromptNode, ctx: dict) -> dict:
    """Run a prompt against the host's live/primary agent via ``prompt_handler``.

    Raises if no handler was supplied — a ``prompt`` node has no meaning without
    a live agent to run it on (e.g. a headless/scheduled run).
    """
    handler = ctx.get("prompt_handler")
    if handler is None:
        raise RuntimeError(
            "prompt_handler is required to run a 'prompt' node "
            "(no live/primary agent available — is this a headless run?)"
        )
    prompt = _render_template(node.prompt, ctx["vars"])
    text = await handler(prompt, ctx["vars"])
    if node.name is not None:
        ctx["vars"][node.name] = text
    return {"text": text}


async def _run_node(node: Node, ctx: dict, agent_factory: Callable[[], Any]) -> dict:
    if isinstance(node, AgentNode):
        return await _run_agent(node, ctx, agent_factory)
    if isinstance(node, PromptNode):
        return await _run_prompt(node, ctx)
    if isinstance(node, ToolNode):
        return await _run_tool(node, ctx)
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
    }
    return await _run_node(spec.root, ctx, agent_factory)
