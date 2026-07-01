"""Native lovelaice workflow engine — spec models + async executor."""

from typing import Any, Callable

from lovelaice.workflows.models import AgentNode, Node, SequenceNode, WorkflowSpec
from lovelaice.workflows.executor import run, _final_text

__all__ = [
    "AgentNode",
    "SequenceNode",
    "WorkflowSpec",
    "Node",
    "run",
    "_final_text",
    "workflow",
]


class _Workflow:
    def __init__(self, spec: WorkflowSpec):
        self._spec = spec

    async def run(
        self, *, agent_factory: Callable[[], Any], inputs: dict | None = None
    ) -> dict:
        return await run(self._spec, agent_factory=agent_factory, inputs=inputs)


def workflow(spec: "dict | WorkflowSpec") -> _Workflow:
    """Native entrypoint: ``lovelaice.workflow(spec).run(agent_factory=...)``."""
    if isinstance(spec, dict):
        spec = WorkflowSpec.model_validate(spec)
    return _Workflow(spec)
