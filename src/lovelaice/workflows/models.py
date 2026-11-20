"""Declarative workflow spec models.

A workflow is a tree of typed nodes discriminated by ``kind``. This module is
intentionally generic — it has no lovelaice-agent, ainbox, or magpie coupling,
so the spec format is usable anywhere.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field


class AgentNode(BaseModel):
    """Run a prompt on a fresh agent.

    With ``output_schema`` set, the reply's first JSON value is validated
    against that JSON Schema; a failure re-prompts the same agent with the
    validation error up to ``retries`` times, then raises ``OutputSchemaError``.

    ``agent`` names a factory from ``run(agents={...})``; unset, the run's
    default ``agent_factory`` builds it. A node that is one typed call (classify,
    extract, compare) can point at a tool-less factory, so small models are not
    dragged through a ReAct loop.
    """

    kind: Literal["agent"] = "agent"
    prompt: str
    name: str | None = None
    agent: str | None = None
    output_schema: dict | None = None
    retries: int = Field(default=2, ge=0)


class ToolNode(BaseModel):
    """Deterministically invoke a host-provided tool (no LLM).

    ``tool`` names a handler the host registered (e.g. an MCP tool wired into
    the agent, like magpie's ``write_note``). ``args`` values are templated
    from context: a value that is exactly ``"{var}"`` is replaced by the raw
    context object (dict/list preserved); any other string is ``str.format``-ed.
    """

    kind: Literal["tool"] = "tool"
    tool: str
    args: dict = Field(default_factory=dict)
    name: str | None = None


class PromptNode(BaseModel):
    """Run a prompt against the HOST's live/primary agent (shared context),
    not a fresh one. Same shape as ``AgentNode``; the executor routes it to a
    host-supplied ``prompt_handler`` instead of ``agent_factory``.
    """

    kind: Literal["prompt"] = "prompt"
    prompt: str
    name: str | None = None
    output_schema: dict | None = None
    retries: int = Field(default=2, ge=0)


OnError = Literal["raise", "skip", "keep"]
"""What a fan-out does when one item fails.

``raise`` (default) fails the whole run as before; ``skip`` leaves the failed
items out of the result list; ``keep`` puts ``{"error": "<message>"}`` in their
place, so the rest of the results survive and a consumer can count them.
"""


class ParallelNode(BaseModel):
    """Run children concurrently (each in an isolated vars scope); collect the
    list of their results into ``name``. Children must NOT be ``prompt`` nodes.

    ``on_error`` decides what happens when one child fails (see ``OnError``).
    """

    kind: Literal["parallel"] = "parallel"
    children: list["Node"] = Field(min_length=1)
    name: str | None = None
    on_error: OnError = "raise"


class MapNode(BaseModel):
    """Fan ``node`` over the list at ``vars[over]``, binding each element to
    ``vars[as]`` in an isolated scope; collect results into ``name``.

    ``on_error`` decides what happens when one item fails (see ``OnError``).
    """

    kind: Literal["map"] = "map"
    over: str
    as_: str = Field(alias="as")
    node: "Node"
    name: str | None = None
    on_error: OnError = "raise"

    model_config = {"populate_by_name": True}


class SequenceNode(BaseModel):
    kind: Literal["sequence"] = "sequence"
    children: list["Node"] = Field(min_length=1)


Node = Annotated[
    Union[AgentNode, PromptNode, ToolNode, ParallelNode, MapNode, SequenceNode],
    Field(discriminator="kind"),
]


class WorkflowSpec(BaseModel):
    name: str
    root: Node


SequenceNode.model_rebuild()
ParallelNode.model_rebuild()
MapNode.model_rebuild()
