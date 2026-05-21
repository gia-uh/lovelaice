"""Entrypoint: `python -m lovelaice.acp` runs the ACP stdio server.

Used both by external ACP clients (aegis's LovelaiceDriver, Zed) and
by the canary test which spawns it as a subprocess.

Config sources (priority): env vars → built-in defaults.
"""
import asyncio
import os
from pathlib import Path

from lovelaice.acp.server import AcpServer
from lovelaice.agent import Agent, AgentConfig
from lovelaice.agent.loops.react_native import ReActNative


def _agent_factory():
    """Default agent factory for `python -m lovelaice.acp`.

    Reads LOVELAICE_MODEL, LOVELAICE_SYSTEM_PROMPT, LOVELAICE_CWD,
    LOVELAICE_BASE_URL, OPENROUTER_API_KEY/OPENAI_API_KEY,
    LOVELAICE_SESSION_PATH from the environment.

    Set LOVELAICE_FAKE_LLM=1 to use a canned mock LLM (for tests/dev).
    """
    cfg = AgentConfig(
        model=os.getenv("LOVELAICE_MODEL", "anthropic/claude-haiku-4-5"),
        system_prompt=os.getenv("LOVELAICE_SYSTEM_PROMPT", "You are a helpful agent."),
        cwd=os.getenv("LOVELAICE_CWD", os.getcwd()),
        api_key=os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("LOVELAICE_BASE_URL", "https://openrouter.ai/api/v1"),
    )
    session_path = Path(os.getenv(
        "LOVELAICE_SESSION_PATH",
        str(Path.home() / ".lovelaice" / "sessions" / "ad-hoc.jsonl"),
    ))

    if os.getenv("LOVELAICE_FAKE_LLM"):
        # Test mode: monkey-patch the LLM with a canned echo.
        from unittest.mock import AsyncMock
        from lingo.llm import Message
        fake = AsyncMock()
        fake.chat = AsyncMock(
            return_value=Message.assistant("hello from fake", stop_reason="stop"))
        import lovelaice.agent.agent as agent_mod
        agent_mod._build_llm = lambda cfg: fake

    return Agent(config=cfg, tools=[], loop=ReActNative(),
                 session_path=session_path)


def main():
    server = AcpServer(agent_factory=_agent_factory)
    asyncio.run(server.run_stdio())


if __name__ == "__main__":
    main()
