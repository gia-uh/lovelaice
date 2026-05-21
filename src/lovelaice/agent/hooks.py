"""Hook registry — observational fan-out + reducer chains.

VS1 supports a minimum set of hooks; the chain dispatch supports any
event name registered. Reducer-style for `tool_call` (Allow/Block/AskUser
return type); observational fan-out for everything else.

`AskUser` is a placeholder in VS1 — when a tool_call hook returns it,
the harness has no permission-flow wired yet (that's VS2's
session/request_permission). VS1 treats AskUser as Block to fail-safe.
"""
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Allow:
    """Hook decision: tool may run."""

    allowed: bool = True


@dataclass
class Block:
    """Hook decision: tool blocked. Synthesized as an is_error result."""

    reason: str
    allowed: bool = False


@dataclass
class AskUser:
    """Hook decision: defer to the ACP client's session/request_permission.

    In VS1 the harness has no permission flow wired, so AskUser is
    treated as Block. VS2 will wire it to ACP."""

    options: list[dict] = field(default_factory=list)
    allowed: bool = False


HookFn = Callable[..., Any]


class HookRegistry:
    """Observational fan-out + reducer chains, keyed by event name."""

    def __init__(self):
        self._hooks: dict[str, list[HookFn]] = {}

    def register(self, name: str, fn: HookFn) -> None:
        self._hooks.setdefault(name, []).append(fn)

    async def emit(self, name: str, *args, **kwargs) -> None:
        """Observational fan-out; return values ignored.

        Sync and async handlers both supported (async are awaited)."""
        for fn in self._hooks.get(name, []):
            res = fn(*args, **kwargs)
            if inspect.isawaitable(res):
                await res

    async def reduce_tool_call(self, call) -> Allow | Block:
        """Run the tool_call reducer chain.

        First Block wins (short-circuits). An AskUser return is converted
        to Block in VS1 (with a 'permission flow not yet implemented' reason).
        Empty chain or all-Allow returns Allow. None returns are treated
        as Allow (pass to next handler)."""
        for fn in self._hooks.get("tool_call", []):
            res = fn(call)
            if inspect.isawaitable(res):
                res = await res
            if res is None:
                continue
            if isinstance(res, Block):
                return res
            if isinstance(res, AskUser):
                # VS1: no permission flow; fail-safe to Block.
                return Block(
                    reason="user permission flow not yet implemented (VS2)",
                )
            if isinstance(res, Allow):
                continue
        return Allow()
