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

import jsonschema

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


class OutputSchemaError(RuntimeError):
    """A node's output still did not match its ``output_schema`` after retries.

    Names the node and carries the last validation error, so a failing workflow
    points at the step and the field instead of a bare ``JSONDecodeError``.
    """


_FENCE = re.compile(r"```[ \t]*(?:[A-Za-z0-9_+.-]+)?[ \t]*\n?(.*?)```", re.DOTALL)


def _extract_json(text: str) -> Any:
    """Return the first JSON value in ``text``.

    Small models wrap their answer in a code fence or in a sentence, so the
    whole text is tried first, then each fenced block, then the first ``{`` or
    ``[`` from which a complete value decodes. Raises ``ValueError`` when the
    text holds no JSON value at all.
    """
    stripped = text.strip()
    if stripped:
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass
        for block in _FENCE.findall(stripped):
            block = block.strip()
            if not block:
                continue
            try:
                return json.loads(block)
            except json.JSONDecodeError:
                continue
        decoder = json.JSONDecoder()
        for i, ch in enumerate(stripped):
            if ch in "{[":
                try:
                    value, _end = decoder.raw_decode(stripped, i)
                except json.JSONDecodeError:
                    continue
                return value
    raise ValueError("the reply contained no JSON value")


def _node_label(name: str | None) -> str:
    return f"node {name!r}" if name else "an unnamed workflow node"


def _schema_checker(
    node_name: str | None, schema: dict
) -> Callable[[str], tuple[Any, str | None]]:
    """Compile ``schema`` into ``text -> (value, error)``.

    An invalid schema is the workflow author's mistake, not the model's, so it
    fails at once instead of burning the retries.
    """
    validator_cls = jsonschema.validators.validator_for(schema)
    try:
        validator_cls.check_schema(schema)
    except jsonschema.exceptions.SchemaError as exc:
        raise OutputSchemaError(
            f"{_node_label(node_name)}: invalid output_schema: {exc.message}"
        ) from exc
    validator = validator_cls(schema)

    def check(text: str) -> tuple[Any, str | None]:
        try:
            value = _extract_json(text)
        except ValueError as exc:
            return None, str(exc)
        errors = sorted(
            validator.iter_errors(value),
            key=lambda e: [str(p) for p in e.absolute_path],
        )
        if not errors:
            return value, None
        first = errors[0]
        where = "/".join(str(p) for p in first.absolute_path)
        return None, f"{where}: {first.message}" if where else first.message

    return check


def _repair_prompt(error: str, schema: dict) -> str:
    """Re-prompt text for the same agent: the error, plus the schema again."""
    return (
        "Your previous reply could not be used as this step's structured output.\n"
        f"Problem: {error}\n\n"
        "Reply again with ONLY one JSON value that matches this JSON Schema — "
        "no prose, no explanation, no code fence:\n\n"
        f"{json.dumps(schema, ensure_ascii=False, indent=2)}"
    )


async def _enforce_schema(
    node: "AgentNode | PromptNode",
    check: Callable[[str], tuple[Any, str | None]],
    text: str,
    reask: Callable[[str], Awaitable[str]],
) -> Any:
    """Validate ``text``; on failure ``reask`` with the error, up to the retries.

    ``check`` comes from ``_schema_checker`` and is built before the first model
    call, so a malformed schema costs nothing. ``reask`` performs one more model
    call with the repair prompt and returns the new reply, so the caller decides
    which agent is asked (the node's own fresh agent, or the host's live one).
    """
    schema = node.output_schema
    assert schema is not None  # callers only ask when a schema is set
    value, error = check(text)
    for _ in range(node.retries):
        if error is None:
            break
        text = await reask(_repair_prompt(error, schema))
        value, error = check(text)
    if error is not None:
        raise _output_schema_error(node.name, node.retries, error)
    return value


def _output_schema_error(node_name: str | None, retries: int, error: str) -> OutputSchemaError:
    attempts = retries + 1
    return OutputSchemaError(
        f"{_node_label(node_name)}: output_schema not satisfied after "
        f"{attempts} attempt{'s' if attempts != 1 else ''}; last error: {error}"
    )


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
    """Run ``node`` on a fresh agent.

    With ``output_schema``, the reply is parsed and validated; a bad reply gets
    ``node.retries`` more chances, each one a re-prompt of the SAME agent with
    the validation error, so its context (the task and its own wrong answer)
    is kept. Exhausted retries raise ``OutputSchemaError``.
    """
    prompt = _render_template(node.prompt, ctx["vars"])
    check = (
        _schema_checker(node.name, node.output_schema)
        if node.output_schema is not None
        else None
    )
    agent = agent_factory()
    async with _sem(ctx):
        await agent.prompt(prompt)
    text = _final_text(agent)
    if check is not None:
        async def reask(repair: str) -> str:
            async with _sem(ctx):
                await agent.prompt(repair)
            return _final_text(agent)

        value = await _enforce_schema(node, check, text, reask)
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
    check = (
        _schema_checker(node.name, node.output_schema)
        if node.output_schema is not None
        else None
    )
    async with _sem(ctx):
        text = await handler(prompt, ctx["vars"])
    if check is not None:
        async def reask(repair: str) -> str:
            async with _sem(ctx):
                return await handler(repair, ctx["vars"])

        value = await _enforce_schema(node, check, text, reask)
        if node.name is not None:
            ctx["vars"][node.name] = value
        return value
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
