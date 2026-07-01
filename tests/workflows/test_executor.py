from unittest.mock import AsyncMock

import pytest
from lingo.llm import Message

from lovelaice.agent import Agent, AgentConfig
from lovelaice.agent.loops.react_native import ReActNative
from lovelaice.workflows import WorkflowSpec, run


def _factory(monkeypatch, tmp_path, responses):
    """agent_factory whose successive Agents each return the next canned reply."""
    counter = {"n": 0}

    def make() -> Agent:
        i = counter["n"]
        counter["n"] = i + 1
        fake = AsyncMock()
        fake.chat = AsyncMock(
            return_value=Message.assistant(responses[i], stop_reason="stop")
        )
        monkeypatch.setattr(
            "lovelaice.agent.agent._build_llm", lambda cfg, _f=fake: _f
        )
        return Agent(
            config=AgentConfig(model="m", system_prompt="SYS"),
            tools=[],
            loop=ReActNative(),
            session_path=tmp_path / f"s_{i}.jsonl",
        )

    return make


@pytest.mark.asyncio
async def test_single_agent_node_returns_text(monkeypatch, tmp_path):
    spec = WorkflowSpec.model_validate(
        {"name": "one", "root": {"kind": "agent", "prompt": "hi"}}
    )
    out = await run(spec, agent_factory=_factory(monkeypatch, tmp_path, ["hello world"]))
    assert out == {"text": "hello world"}


@pytest.mark.asyncio
async def test_sequence_templates_prior_output(monkeypatch, tmp_path):
    factory = _factory(monkeypatch, tmp_path, ["42", "84"])
    spec = WorkflowSpec.model_validate(
        {
            "name": "seq",
            "root": {
                "kind": "sequence",
                "children": [
                    {"kind": "agent", "prompt": "give a number", "name": "n"},
                    {"kind": "agent", "prompt": "double {n}"},
                ],
            },
        }
    )
    out = await run(spec, agent_factory=factory)
    assert out == {"text": "84"}


@pytest.mark.asyncio
async def test_agent_node_with_schema_parses_json(monkeypatch, tmp_path):
    spec = WorkflowSpec.model_validate(
        {
            "name": "json",
            "root": {
                "kind": "agent",
                "prompt": "emit widgets",
                "output_schema": {"type": "object"},
            },
        }
    )
    factory = _factory(
        monkeypatch, tmp_path, ['{"title": "T", "generated_at": "2026-07-01", "widgets": []}']
    )
    out = await run(spec, agent_factory=factory)
    assert out == {"title": "T", "generated_at": "2026-07-01", "widgets": []}
