import pytest
from unittest.mock import AsyncMock
from lingo.llm import Message
from lingo.tools import tool as lingo_tool
from lovelaice.agent import Agent, AgentConfig, AgentTool
from lovelaice.agent.loops.react_native import ReActNative
from lovelaice.agent.errors import StopReason


@lingo_tool
async def echo(text: str) -> str:
    """Echo."""
    return text


@pytest.mark.asyncio
async def test_agent_prompt_returns_stop_reason(tmp_path, monkeypatch):
    fake = AsyncMock()
    fake.chat = AsyncMock(return_value=Message.assistant("hello", stop_reason="stop"))
    monkeypatch.setattr("lovelaice.agent.agent._build_llm", lambda cfg: fake)

    agent = Agent(
        config=AgentConfig(model="m", system_prompt="SYS"),
        tools=[AgentTool(inner=echo)],
        loop=ReActNative(),
        session_path=tmp_path / "s.jsonl",
    )

    stop = await agent.prompt("hi")
    assert stop == StopReason.END_TURN


@pytest.mark.asyncio
async def test_agent_on_decorator_registers_hook(tmp_path, monkeypatch):
    fake = AsyncMock()
    fake.chat = AsyncMock(return_value=Message.assistant("hi", stop_reason="stop"))
    monkeypatch.setattr("lovelaice.agent.agent._build_llm", lambda cfg: fake)

    agent = Agent(
        config=AgentConfig(model="m", system_prompt="SYS"),
        tools=[],
        loop=ReActNative(),
        session_path=tmp_path / "s.jsonl",
    )

    seen = []

    @agent.hook("turn_start")
    def on_start(ev):
        seen.append(ev)

    await agent.prompt("hi")
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_agent_subscribe_receives_events(tmp_path, monkeypatch):
    fake = AsyncMock()
    fake.chat = AsyncMock(return_value=Message.assistant("hi", stop_reason="stop"))
    monkeypatch.setattr("lovelaice.agent.agent._build_llm", lambda cfg: fake)

    agent = Agent(
        config=AgentConfig(model="m", system_prompt="SYS"),
        tools=[],
        loop=ReActNative(),
        session_path=tmp_path / "s.jsonl",
    )
    events = []
    agent.subscribe(lambda ev: events.append(ev))
    await agent.prompt("hi")
    types = [type(e).__name__ for e in events]
    assert "TurnStart" in types
    assert "TurnEnd" in types


@pytest.mark.asyncio
async def test_agent_resumes_existing_session(tmp_path, monkeypatch):
    """If session_path exists, Agent loads it instead of creating fresh."""
    fake = AsyncMock()
    fake.chat = AsyncMock(return_value=Message.assistant("hi", stop_reason="stop"))
    monkeypatch.setattr("lovelaice.agent.agent._build_llm", lambda cfg: fake)

    # First agent creates the session.
    a1 = Agent(
        config=AgentConfig(model="m", system_prompt="SYS"),
        tools=[],
        loop=ReActNative(),
        session_path=tmp_path / "s.jsonl",
    )
    await a1.prompt("first")

    # Second agent loads the same path.
    a2 = Agent(
        config=AgentConfig(model="m", system_prompt="SYS"),
        tools=[],
        loop=ReActNative(),
        session_path=tmp_path / "s.jsonl",
    )
    # Session should contain the first turn's messages.
    msgs = a2.session.messages_for_llm("SYS")
    roles = [m.role for m in msgs]
    assert "user" in roles
    assert roles.count("user") >= 1


def test_agent_abort_sets_signal(tmp_path, monkeypatch):
    fake = AsyncMock()
    monkeypatch.setattr("lovelaice.agent.agent._build_llm", lambda cfg: fake)

    agent = Agent(
        config=AgentConfig(model="m", system_prompt="SYS"),
        tools=[],
        loop=ReActNative(),
        session_path=tmp_path / "s.jsonl",
    )
    assert not agent.harness.abort.is_set()
    agent.abort()
    assert agent.harness.abort.is_set()
