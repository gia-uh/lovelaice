"""Agent — top-level user-facing API.

Wires AgentConfig + tools + loop + session. Hosts touch this class;
everything below is harness + loop + lingo.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from lingo.llm import LLM, Message
from lovelaice.agent.events import AgentEvent
from lovelaice.agent.harness import Harness
from lovelaice.agent.hooks import HookRegistry
from lovelaice.agent.prompt import assemble_system_prompt
from lovelaice.agent.session import Session
from lovelaice.agent.tools import AgentTool, ToolRegistry


@dataclass
class AgentConfig:
    """User-facing agent configuration."""

    model: str
    system_prompt: str = "You are a helpful agent."
    cwd: str = "."
    api_key: str | None = None
    base_url: str | None = None


def _build_llm(cfg: AgentConfig) -> LLM:
    """Construct the lingo.LLM client from an AgentConfig.

    Exposed as a module-level function so tests can monkey-patch it."""
    return LLM(model=cfg.model, api_key=cfg.api_key, base_url=cfg.base_url)


class Agent:
    """Top-level user-facing API."""

    def __init__(
        self,
        *,
        config: AgentConfig,
        tools: list[AgentTool],
        loop: Any,  # AgentLoop
        session_path: Path,
    ):
        self.config = config
        self.loop = loop

        # Build the harness.
        registry = ToolRegistry()
        for t in tools:
            registry.register(t)
        hooks = HookRegistry()
        sys_prompt = assemble_system_prompt(
            base=config.system_prompt, tools=registry, cwd=config.cwd)
        llm = _build_llm(config)
        self.harness = Harness(
            llm=llm, tools=registry, hooks=hooks, system_prompt=sys_prompt)

        # Build / load the session.
        path = Path(session_path)
        if path.exists():
            self.session = Session.load(path)
        else:
            self.session = Session.create(
                path,
                model=config.model,
                system_prompt_hash=Session.hash_system_prompt(sys_prompt),
                loop=type(loop).__name__,
                cwd=config.cwd,
            )

    def on(self, event_name: str, fn: Callable) -> None:
        """Register an observational hook on the event bus.

        event_name matches the event class name as snake_case
        (e.g. "turn_start" matches TurnStart).
        """
        import re
        # Convert snake_case event_name to PascalCase for class-name comparison.
        pascal = re.sub(r"_([a-z])", lambda m: m.group(1).upper(),
                        event_name[0].upper() + event_name[1:])

        def _filtered(ev):
            if type(ev).__name__ == pascal:
                fn(ev)

        self.harness.subscribe(_filtered)

    def hook(self, event_name: str):
        """Decorator form of `on()`."""
        def deco(fn):
            self.on(event_name, fn)
            return fn
        return deco

    def subscribe(self, fn: Callable[[AgentEvent], Any]) -> None:
        """Subscribe to the AgentEvent stream emitted by the harness."""
        self.harness.subscribe(fn)

    async def prompt(self, text: str):
        """Run one full turn-cluster. Returns the StopReason."""
        return await self.loop.run(
            self.harness, self.session, Message.user(text))

    def abort(self) -> None:
        """Signal the agent to abort its current turn at the next checkpoint."""
        self.harness.abort.set()
