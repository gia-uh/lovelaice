"""Async executor for workflow specs.

Drives lovelaice's ``Agent`` for ``agent`` nodes and threads named outputs
through ``sequence`` nodes. Callers supply an ``agent_factory`` (the same seam
``AcpServer(agent_factory=...)`` uses) so the executor stays decoupled from how
agents are built and what tools/MCP they carry.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from lovelaice.workflows.models import AgentNode, Node, SequenceNode, WorkflowSpec


class _SafeDict(dict):
    def __missing__(self, key: str) -> str:  # leave unknown {placeholders} intact
        return "{" + key + "}"


def _final_text(agent: Any) -> str:
    """Return the last assistant message's text after a completed turn."""
    for m in reversed(agent.messages_for_llm()):
        if getattr(m, "role", None) == "assistant":
            return m.content or ""
    return ""


async def _run_agent(node: AgentNode, ctx: dict, agent_factory: Callable[[], Any]) -> dict:
    prompt = node.prompt.format_map(_SafeDict(ctx["vars"]))
    agent = agent_factory()
    await agent.prompt(prompt)
    text = _final_text(agent)
    if node.name is not None:
        ctx["vars"][node.name] = text
    if node.output_schema is not None:
        return json.loads(text)
    return {"text": text}


async def _run_node(node: Node, ctx: dict, agent_factory: Callable[[], Any]) -> dict:
    if isinstance(node, AgentNode):
        return await _run_agent(node, ctx, agent_factory)
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
    inputs: dict | None = None,
) -> dict:
    """Execute a workflow spec and return the root node's result dict."""
    ctx = {"vars": dict(inputs or {})}
    return await _run_node(spec.root, ctx, agent_factory)
