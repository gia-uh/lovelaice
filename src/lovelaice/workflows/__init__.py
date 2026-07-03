"""Native lovelaice workflow engine — spec models + async executor."""

from typing import Any, Callable

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
from lovelaice.workflows.executor import Handler, run, _final_text

__all__ = [
    "AgentNode",
    "PromptNode",
    "ToolNode",
    "ParallelNode",
    "MapNode",
    "SequenceNode",
    "WorkflowSpec",
    "Node",
    "Handler",
    "run",
    "_final_text",
    "workflow",
]


class _Workflow:
    def __init__(self, spec: WorkflowSpec):
        self._spec = spec

    async def run(
        self,
        *,
        agent_factory: Callable[[], Any],
        handlers: "dict[str, Handler] | None" = None,
        inputs: dict | None = None,
        prompt_handler: "Callable[[str, dict], Any] | None" = None,
    ) -> dict:
        return await run(
            self._spec,
            agent_factory=agent_factory,
            handlers=handlers,
            inputs=inputs,
            prompt_handler=prompt_handler,
        )


def workflow(spec: "dict | WorkflowSpec") -> _Workflow:
    """Native entrypoint: ``lovelaice.workflow(spec).run(agent_factory=...)``."""
    if isinstance(spec, dict):
        spec = WorkflowSpec.model_validate(spec)
    return _Workflow(spec)
