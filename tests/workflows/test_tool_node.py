from unittest.mock import AsyncMock

import pytest
from lingo.llm import Message

from lovelaice.agent import Agent, AgentConfig
from lovelaice.agent.loops.react_native import ReActNative
from lovelaice.workflows import WorkflowSpec, run


def _agent_factory(monkeypatch, tmp_path, response):
    def make() -> Agent:
        fake = AsyncMock()
        fake.chat = AsyncMock(return_value=Message.assistant(response, stop_reason="stop"))
        monkeypatch.setattr("lovelaice.agent.agent._build_llm", lambda cfg: fake)
        return Agent(
            config=AgentConfig(model="m"),
            tools=[],
            loop=ReActNative(),
            session_path=tmp_path / "s.jsonl",
        )

    return make


@pytest.mark.asyncio
async def test_tool_node_dispatches_to_handler_with_raw_object(monkeypatch, tmp_path):
    seen = {}

    async def write_note(args, vars):
        seen["args"] = args
        return "written:" + args["path"]

    spec = WorkflowSpec.model_validate(
        {
            "name": "kpis",
            "root": {
                "kind": "sequence",
                "children": [
                    {
                        "kind": "agent",
                        "prompt": "emit",
                        "output_schema": {"type": "object"},
                        "name": "data",
                    },
                    {
                        "kind": "tool",
                        "tool": "write_note",
                        "args": {"path": "kpis.md", "payload": "{data}"},
                    },
                ],
            },
        }
    )
    out = await run(
        spec,
        agent_factory=_agent_factory(monkeypatch, tmp_path, '{"widgets": [1, 2]}'),
        handlers={"write_note": write_note},
    )
    # payload arrived as the raw dict, not a stringified one:
    assert seen["args"] == {"path": "kpis.md", "payload": {"widgets": [1, 2]}}
    assert out == {"tool": "write_note", "result": "written:kpis.md"}


@pytest.mark.asyncio
async def test_tool_node_unknown_handler_raises(monkeypatch, tmp_path):
    spec = WorkflowSpec.model_validate(
        {"name": "w", "root": {"kind": "tool", "tool": "nope", "args": {}}}
    )
    with pytest.raises(KeyError):
        await run(
            spec,
            agent_factory=_agent_factory(monkeypatch, tmp_path, "x"),
            handlers={},
        )
