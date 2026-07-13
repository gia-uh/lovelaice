"""``python -m lovelaice.acp.v1`` (and the ``lovelaice-acp`` script) run the
ACP v1 stdio server.

Config from env: LOVELAICE_MODEL / LOVELAICE_BASE_URL /
OPENROUTER_API_KEY|OPENAI_API_KEY / LOVELAICE_CWD / LOVELAICE_SESSION_PATH.
Set LOVELAICE_FAKE_LLM=1 to swap a canned LLM for tests/dev.
"""
import asyncio
import os
from pathlib import Path

import acp

from lovelaice.acp.v1.server import AcpServerV1
from lovelaice.coding.host import create_coding_agent


def _default_factory(*, mcp_tools=None, session_path=None, **_kw):
    if os.getenv("LOVELAICE_FAKE_LLM"):
        from unittest.mock import AsyncMock
        from lingo.llm import Message
        import lovelaice.agent.agent as agent_mod
        fake = AsyncMock()
        fake.chat = AsyncMock(
            return_value=Message.assistant("ok", stop_reason="stop"))
        agent_mod._build_llm = lambda cfg: fake
    # Per-session path from the server (enables load_session resume); fall
    # back to the env default for standalone / one-shot use.
    session_path = Path(session_path or os.getenv(
        "LOVELAICE_SESSION_PATH",
        str(Path.home() / ".lovelaice" / "sessions" / "ad-hoc.jsonl")))
    session_path.parent.mkdir(parents=True, exist_ok=True)
    return create_coding_agent(
        model=os.getenv("LOVELAICE_MODEL", "anthropic/claude-haiku-4-5"),
        session_path=session_path,
        cwd=os.getenv("LOVELAICE_CWD", os.getcwd()),
        base_url=os.getenv("LOVELAICE_BASE_URL"),
        api_key=os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"),
        extra_tools=mcp_tools,
    )


def main() -> None:
    server = AcpServerV1(agent_factory=_default_factory)
    asyncio.run(acp.run_agent(server))


if __name__ == "__main__":
    main()
