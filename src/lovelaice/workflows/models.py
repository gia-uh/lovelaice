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


class SequenceNode(BaseModel):
    kind: Literal["sequence"] = "sequence"
    children: list["Node"] = Field(min_length=1)


Node = Annotated[Union[AgentNode, SequenceNode], Field(discriminator="kind")]


class WorkflowSpec(BaseModel):
    name: str
    root: Node


SequenceNode.model_rebuild()
