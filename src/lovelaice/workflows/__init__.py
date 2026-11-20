"""Native lovelaice workflow engine — spec models + async executor."""

from typing import Any, Callable

from lovelaice.workflows.models import (
    AgentNode,
    MapNode,
    Node,
    OnError,
    ParallelNode,
    PromptNode,
    SequenceNode,
    ToolNode,
    WorkflowSpec,
)
from lovelaice.workflows.executor import (
    Handler,
    OutputSchemaError,
    UnknownAgentError,
    _extract_json,
    _final_text,
    run,
)
from lovelaice.workflows.events import (
    ItemEnd,
    NodeEnd,
    NodeError,
    NodeStart,
    WorkflowEvent,
)

__all__ = [
    "OutputSchemaError",
    "UnknownAgentError",
    "_extract_json",
    "AgentNode",
    "PromptNode",
    "ToolNode",
    "ParallelNode",
    "MapNode",
    "SequenceNode",
    "WorkflowSpec",
    "Node",
    "OnError",
    "Handler",
    "run",
    "_final_text",
    "workflow",
    "NodeStart",
    "NodeEnd",
    "NodeError",
    "ItemEnd",
    "WorkflowEvent",
]


class _Workflow:
    def __init__(self, spec: WorkflowSpec):
        self._spec = spec

    async def run(
        self,
        *,
        agent_factory: Callable[[], Any],
        agents: "dict[str, Callable[[], Any]] | None" = None,
        handlers: "dict[str, Handler] | None" = None,
        inputs: dict | None = None,
        prompt_handler: "Callable[[str, dict], Any] | None" = None,
        on_event: "Callable[[WorkflowEvent], Any] | None" = None,
    ) -> dict:
        return await run(
            self._spec,
            agent_factory=agent_factory,
            agents=agents,
            handlers=handlers,
            inputs=inputs,
            prompt_handler=prompt_handler,
            on_event=on_event,
        )


def workflow(spec: "dict | WorkflowSpec") -> _Workflow:
    """Native entrypoint: ``lovelaice.workflow(spec).run(agent_factory=...)``."""
    if isinstance(spec, dict):
        spec = WorkflowSpec.model_validate(spec)
    return _Workflow(spec)
