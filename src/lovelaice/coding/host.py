"""lovelaice.coding host module — assembles an Agent for the coding-agent use case."""
import os
from pathlib import Path

from lovelaice.agent import Agent, AgentConfig, AgentTool
from lovelaice.agent.loops.react_native import ReActNative
from lovelaice.coding.tools.read import read as read_tool
from lovelaice.coding.tools.bash import bash as bash_tool
from lovelaice.coding.hooks import path_guard, bash_prefix_guard


CODING_PREAMBLE = (
    "You are a coding agent. You can read files and run bash commands.\n"
    "Prefer reading before writing; explain your steps."
)


def create_coding_agent(
    *,
    model: str,
    session_path: Path,
    cwd: str,
    base_url: str | None = None,
    api_key: str | None = None,
    extra_tools: list[AgentTool] | None = None,
) -> Agent:
    """Build a coding-host Agent.

    Wires the read+bash tools, the path-pattern + bash-prefix guard hooks,
    and the coding preamble onto a ReActNative agent. ``extra_tools`` (e.g.
    per-session MCP tools) are added to the registry at construction so the
    system prompt advertises them.
    """
    cfg = AgentConfig(
        model=model,
        system_prompt=CODING_PREAMBLE,
        cwd=cwd,
        api_key=api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"),
        base_url=base_url or "https://openrouter.ai/api/v1",
    )
    tools = [
        AgentTool(inner=read_tool, kind="read", title_template="Reading {path}"),
        AgentTool(inner=bash_tool, sequential=True, kind="execute",
                  title_template="Running shell"),
        *(extra_tools or []),
    ]
    agent = Agent(config=cfg, tools=tools, loop=ReActNative(),
                  session_path=session_path)
    # Wire host guard hooks via the agent's hooks registry.
    agent.harness.hooks.register("tool_call",
                                 lambda call: path_guard(call, cwd=cwd))
    agent.harness.hooks.register("tool_call", bash_prefix_guard)
    return agent
