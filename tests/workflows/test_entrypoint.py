from unittest.mock import AsyncMock

import pytest
from lingo.llm import Message

from lovelaice import workflow
from lovelaice.agent import Agent, AgentConfig
from lovelaice.agent.loops.react_native import ReActNative


@pytest.mark.asyncio
async def test_workflow_entrypoint_runs(monkeypatch, tmp_path):
    def make() -> Agent:
        fake = AsyncMock()
        fake.chat = AsyncMock(return_value=Message.assistant("done", stop_reason="stop"))
        monkeypatch.setattr("lovelaice.agent.agent._build_llm", lambda cfg: fake)
        return Agent(
            config=AgentConfig(model="m"),
            tools=[],
            loop=ReActNative(),
            session_path=tmp_path / "s.jsonl",
        )

    out = await workflow({"name": "w", "root": {"kind": "agent", "prompt": "go"}}).run(
        agent_factory=make
    )
    assert out == {"text": "done"}
