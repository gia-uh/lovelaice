import pytest
from unittest.mock import AsyncMock
from lingo.llm import Message
from lovelaice.cli import run_one_shot
from lovelaice.agent.errors import StopReason


@pytest.mark.asyncio
async def test_cli_one_shot_prints_final_text(monkeypatch, tmp_path, capsys):
    """Stub the coding host with a no-tools Agent + mock LLM."""
    fake = AsyncMock()
    fake.chat = AsyncMock(
        return_value=Message.assistant("done!", stop_reason="stop"))
    monkeypatch.setattr("lovelaice.agent.agent._build_llm", lambda cfg: fake)

    def stub_create_coding_agent(*, model, session_path, cwd, **kwargs):
        from lovelaice.agent import Agent, AgentConfig
        from lovelaice.agent.loops.react_native import ReActNative
        return Agent(
            config=AgentConfig(model=model, system_prompt="SYS"),
            tools=[],
            loop=ReActNative(),
            session_path=session_path,
        )

    # Stub the coding host. Use monkeypatch.setitem so sys.modules is
    # RESTORED after the test — a raw assignment leaks the stub module and
    # breaks any later test that imports the real create_coding_agent.
    import sys, types
    fake_module = types.ModuleType("lovelaice.coding.host")
    fake_module.create_coding_agent = stub_create_coding_agent
    monkeypatch.setitem(sys.modules, "lovelaice.coding.host", fake_module)

    stop = await run_one_shot(
        prompt="hi",
        model="m",
        session_path=tmp_path / "s.jsonl",
        cwd=str(tmp_path),
    )
    assert stop == StopReason.END_TURN
    out = capsys.readouterr().out
    assert "done!" in out
