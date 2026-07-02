"""Agent — top-level user-facing API.

Wires AgentConfig + tools + loop + session. Hosts touch this class;
everything below is harness + loop + lingo.
"""
import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING

from lingo.llm import LLM, Message
from lovelaice.agent.events import AgentEvent
from lovelaice.agent.harness import Harness
from lovelaice.agent.hooks import HookRegistry
from lovelaice.agent.prompt import assemble_system_prompt
from lovelaice.agent.session import Session
from lovelaice.agent.tools import AgentTool, ToolRegistry

if TYPE_CHECKING:
    from lovelaice.agent.conversation import Conversation, ConversationStore


@dataclass
class AgentConfig:
    """User-facing agent configuration."""

    model: str
    system_prompt: str = "You are a helpful agent."
    cwd: str = "."
    api_key: str | None = None
    base_url: str | None = None
    # Cap per-turn output. Some Anthropic-via-OpenRouter models default to
    # a very large `max_tokens` (~64K) which can exceed credit-balance caps
    # even when the actual response would be small. None → SDK default.
    max_tokens: int | None = None
    # Opt-in: on pydantic arg-validation failure, attempt one focused forced-JSON
    # repair of the tool-call arguments before falling back to the loop. Off by
    # default (zero hot-path change); ainbox enables it for small models.
    repair_tool_calls: bool = False
    # Grounding context passed to the repair shot: "none" | "turn" | "full".
    repair_context: str = "turn"


def _build_llm(cfg: AgentConfig) -> LLM:
    """Construct the lingo.LLM client from an AgentConfig.

    Exposed as a module-level function so tests can monkey-patch it."""
    kwargs: dict = {}
    if cfg.max_tokens is not None:
        kwargs["max_tokens"] = cfg.max_tokens
    return LLM(
        model=cfg.model, api_key=cfg.api_key, base_url=cfg.base_url, **kwargs,
    )


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
            llm=llm, tools=registry, hooks=hooks, system_prompt=sys_prompt,
            repair_tool_calls=config.repair_tool_calls,
            repair_context=config.repair_context)

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
        # Let the harness rewrite in-session tool-call args when repair succeeds.
        self.harness.session = self.session

    @classmethod
    def from_conversation(
        cls,
        *,
        config: AgentConfig,
        tools: list[AgentTool],
        loop: Any,
        conversation: "Conversation",
        store: "ConversationStore",
    ) -> "Agent":
        """Build an Agent whose state mirrors a persistent Conversation.

        The agent's message list is seeded from the conversation; every
        new message the loop appends is written back to the store as a
        side effect (fire-and-forget asyncio task — durability is best-
        effort but ordered, since each append is scheduled in turn order).
        """
        agent = cls.__new__(cls)
        agent.config = config
        agent.loop = loop

        registry = ToolRegistry()
        for t in tools:
            registry.register(t)
        hooks = HookRegistry()
        sys_prompt = assemble_system_prompt(
            base=config.system_prompt, tools=registry, cwd=config.cwd)
        llm = _build_llm(config)
        agent.harness = Harness(
            llm=llm, tools=registry, hooks=hooks, system_prompt=sys_prompt,
            repair_tool_calls=config.repair_tool_calls,
            repair_context=config.repair_context)

        agent.session = _ConversationSessionAdapter(conversation, store)
        agent.harness.session = agent.session
        return agent

    def messages_for_llm(self) -> list[Message]:
        """Return the message list the next LLM call would see — system
        prompt plus the session/conversation history."""
        return self.session.messages_for_llm(self.harness.system_prompt)

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


class _ConversationSessionAdapter:
    """Session-shaped adapter over a Conversation + ConversationStore.

    The agent loop calls `session.append(msg)` synchronously; this adapter
    appends to an in-memory list (the source of truth for the loop's next
    `messages_for_llm` call) and schedules a background persist into the
    beaverdb store. Persistence is fire-and-forget — durability is best-
    effort. See spec §11 for the failure model.
    """

    def __init__(self, conversation, store) -> None:
        from lovelaice.agent.conversation import _deserialise
        self._conversation = conversation
        self._store = store
        self._messages: list[Message] = [
            _deserialise(m) for m in conversation.row.messages
        ]

    @property
    def turn_count(self) -> int:
        return sum(1 for m in self._messages if m.role == "user")

    def messages_for_llm(self, system_prompt: str) -> list[Message]:
        return [Message.system(system_prompt), *self._messages]

    def append(self, msg: Message) -> dict:
        self._messages.append(msg)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No event loop — adapter is being used outside async context
            # (test, REPL). Skip persistence; in-memory append is enough.
            return {}
        asyncio.create_task(self._store.append(self._conversation.id, msg))
        return {}

    def update_tool_call_args(self, call_id: str, new_args: dict) -> None:
        """Rewrite the args of the assistant tool call `call_id` in the live
        message list (newest-first). No-op if not found. Mirrors Session so the
        repair path can rewrite history uniformly across both session types."""
        for msg in reversed(self._messages):
            for tc in msg.tool_calls or []:
                if tc.id == call_id:
                    tc.arguments = new_args
                    return
