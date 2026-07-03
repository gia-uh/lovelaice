"""Declarative workflow spec models.

A workflow is a tree of typed nodes discriminated by ``kind``. This module is
intentionally generic — it has no lovelaice-agent, ainbox, or magpie coupling,
so the spec format is usable anywhere.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field


class AgentNode(BaseModel):
    kind: Literal["agent"] = "agent"
    prompt: str
    name: str | None = None
    output_schema: dict | None = None


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


class ParallelNode(BaseModel):
    """Run children concurrently (each in an isolated vars scope); collect the
    list of their results into ``name``. Children must NOT be ``prompt`` nodes."""

    kind: Literal["parallel"] = "parallel"
    children: list["Node"] = Field(min_length=1)
    name: str | None = None


class MapNode(BaseModel):
    """Fan ``node`` over the list at ``vars[over]``, binding each element to
    ``vars[as]`` in an isolated scope; collect results into ``name``."""

    kind: Literal["map"] = "map"
    over: str
    as_: str = Field(alias="as")
    node: "Node"
    name: str | None = None

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
