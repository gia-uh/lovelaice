"""Per-node agent factories: ``AgentNode.agent`` + ``run(agents=...)``."""

from lingo.llm import Message
import pytest

from lovelaice import workflow
from lovelaice.workflows import (
    AgentNode,
    UnknownAgentError,
    WorkflowSpec,
    run,
)


class StubAgent:
    """The executor's whole agent surface: ``prompt`` + ``messages_for_llm``."""

    def __init__(self, label: str, reply: str):
        self.label = label
        self.reply = reply
        self.prompts: list[str] = []

    async def prompt(self, text: str) -> None:
        self.prompts.append(text)

    def messages_for_llm(self):
        return [Message.assistant(self.reply, stop_reason="stop")]


def _named_factory(label: str, reply: str, built: list[str]):
    def make() -> StubAgent:
        built.append(label)
        return StubAgent(label, reply)

    return make


@pytest.mark.asyncio
async def test_each_agent_node_uses_the_factory_it_names():
    built: list[str] = []
    spec = WorkflowSpec.model_validate(
        {
            "name": "mixed",
            "root": {
                "kind": "sequence",
                "children": [
                    {"kind": "agent", "agent": "plain", "prompt": "classify"},
                    {"kind": "agent", "prompt": "research"},
                ],
            },
        }
    )

    out = await run(
        spec,
        agent_factory=_named_factory("full", "tool answer", built),
        agents={"plain": _named_factory("plain", "one call", built)},
    )

    assert out == {"text": "tool answer"}
    assert built == ["plain", "full"]


@pytest.mark.asyncio
async def test_named_agent_reaches_a_map_item():
    built: list[str] = []
    spec = WorkflowSpec.model_validate(
        {
            "name": "fan",
            "root": {
                "kind": "map",
                "over": "items",
                "as": "item",
                "name": "out",
                "node": {
                    "kind": "agent",
                    "agent": "plain",
                    "prompt": "classify {item}",
                },
            },
        }
    )

    out = await run(
        spec,
        agent_factory=_named_factory("full", "tool answer", built),
        agents={"plain": _named_factory("plain", "typed answer", built)},
        inputs={"items": ["a", "b"]},
    )

    assert built == ["plain", "plain"]
    assert out == {"items": [{"text": "typed answer"}, {"text": "typed answer"}]}


@pytest.mark.asyncio
async def test_unknown_agent_name_raises_a_clear_error():
    spec = WorkflowSpec.model_validate(
        {
            "name": "typo",
            "root": {
                "kind": "agent",
                "name": "ficha",
                "agent": "plian",
                "prompt": "classify",
            },
        }
    )

    with pytest.raises(UnknownAgentError) as excinfo:
        await run(
            spec,
            agent_factory=_named_factory("full", "x", []),
            agents={"plain": _named_factory("plain", "y", [])},
        )

    message = str(excinfo.value)
    assert "node 'ficha'" in message
    assert "'plian'" in message
    assert "'plain'" in message


@pytest.mark.asyncio
async def test_unknown_agent_name_without_any_registered_agents():
    spec = WorkflowSpec.model_validate(
        {"name": "none", "root": {"kind": "agent", "agent": "plain", "prompt": "hi"}}
    )

    with pytest.raises(UnknownAgentError, match="no named agents are registered"):
        await run(spec, agent_factory=_named_factory("full", "x", []))


def test_agent_field_parses_and_defaults_to_none():
    named, default = (
        WorkflowSpec.model_validate(
            {"name": "w", "root": {"kind": "agent", "prompt": "hi", "agent": "plain"}}
        ).root,
        WorkflowSpec.model_validate(
            {"name": "w", "root": {"kind": "agent", "prompt": "hi"}}
        ).root,
    )
    assert isinstance(named, AgentNode) and named.agent == "plain"
    assert isinstance(default, AgentNode) and default.agent is None


@pytest.mark.asyncio
async def test_entrypoint_passes_named_factories_through():
    built: list[str] = []
    spec = {
        "name": "entry",
        "root": {"kind": "agent", "agent": "plain", "prompt": "hi"},
    }

    out = await workflow(spec).run(
        agent_factory=_named_factory("full", "tool answer", built),
        agents={"plain": _named_factory("plain", "typed answer", built)},
    )

    assert out == {"text": "typed answer"}
    assert built == ["plain"]
