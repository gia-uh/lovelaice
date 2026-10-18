"""Agent — top-level user-facing API.

Wires AgentConfig + tools + loop + session. Hosts touch this class;
everything below is harness + loop + lingo.
"""
import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING

from lingo.llm import LLM, Message
from lovelaice.agent.budget import BudgetConfig, ContextBudget
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
    # The `Available tools:` prompt block duplicates the API `tools=` array for
    # models with native function-calling. On by default (no behaviour change);
    # ainbox turns it off because its tool descriptions are already one-liners.
    list_tools_in_prompt: bool = True
    # The model's context window in tokens. None (the default) keeps today's
    # behaviour: nothing is compacted unless the provider refuses a prompt.
    context_window: int | None = None
    # Compaction runs above `trigger` of the usable window and cuts to `target`.
    compaction_trigger: float = 0.75
    compaction_target: float = 0.5


def _build_budget(cfg: AgentConfig) -> ContextBudget:
    return ContextBudget(BudgetConfig(
        window=cfg.context_window, max_tokens=cfg.max_tokens,
        trigger=cfg.compaction_trigger, target=cfg.compaction_target, model=cfg.model))


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
            base=config.system_prompt, tools=registry, cwd=config.cwd,
            list_tools=config.list_tools_in_prompt)
        llm = _build_llm(config)
        self.harness = Harness(
            llm=llm, tools=registry, hooks=hooks, system_prompt=sys_prompt,
            repair_tool_calls=config.repair_tool_calls,
            repair_context=config.repair_context, budget=_build_budget(config))

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
            base=config.system_prompt, tools=registry, cwd=config.cwd,
            list_tools=config.list_tools_in_prompt)
        llm = _build_llm(config)
        agent.harness = Harness(
            llm=llm, tools=registry, hooks=hooks, system_prompt=sys_prompt,
            repair_tool_calls=config.repair_tool_calls,
            repair_context=config.repair_context, budget=_build_budget(config))

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
        self._turns: list[int | None] = [m.turn for m in conversation.row.messages]
        from lovelaice.agent.compaction import CompactionState
        self.compaction = getattr(conversation, "compaction", None) or CompactionState()

    def save_compaction(self) -> None:
        """Persisted per conversation in the store's own dict (task 6)."""

    @property
    def turn_count(self) -> int:
        return sum(1 for m in self._messages if m.role == "user")

    def messages_for_llm(self, system_prompt: str) -> list[Message]:
        return [Message.system(system_prompt), *self._messages]

    def turn_numbers(self) -> list[int]:
        """Turn number of each history message, in messages_for_llm order."""
        from lovelaice.agent.loops.react_native import ReActNative
        from lovelaice.agent.turns import assign_turns
        items = [
            (m.role, m.content if isinstance(m.content, str) else "", t)
            for m, t in zip(self._messages, self._turns)
        ]
        return assign_turns(items, ReActNative.EMPTY_TURN_NUDGE)

    def append(self, msg: Message, *, turn: int | None = None) -> dict:
        self._messages.append(msg)
        self._turns.append(turn)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No event loop — adapter is being used outside async context
            # (test, REPL). Skip persistence; in-memory append is enough.
            return {}
        asyncio.create_task(
            self._store.append(self._conversation.id, msg, turn=turn))
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
