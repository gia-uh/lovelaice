import os

from lingo.tools import tool

from lovelaice.agent.tools import AgentTool
from lovelaice.coding.host import create_coding_agent


def test_create_coding_agent_registers_extra_tools(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")

    @tool
    async def widget(x: str) -> str:
        """A widget tool."""
        return x

    extra = AgentTool(inner=widget, kind="other")
    agent = create_coding_agent(
        model="fake/model", session_path=tmp_path / "s.jsonl",
        cwd=str(tmp_path), extra_tools=[extra])
    # Built-ins still present, plus the extra tool.
    assert agent.harness.tools.get("read") is not None
    assert agent.harness.tools.get("widget") is not None
