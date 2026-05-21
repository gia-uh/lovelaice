"""AgentTool wrapper + ToolRegistry + ToolResult.

AgentTool wraps a lingo.Tool with agentic metadata (sequential flag,
ACP tool_call.kind, title template). ToolRegistry owns the collection
and exposes the underlying lingo.Tool list for LLM.chat(tools=...).
ToolResult is lovelaice-level (distinct from lingo.ToolResult, which
is only produced on the structured-output path).
"""
from dataclasses import dataclass, field
from typing import Any, Literal


ToolKind = Literal[
    "read", "edit", "delete", "move", "search",
    "execute", "think", "fetch", "other",
]


@dataclass
class ToolCallLocation:
    """ACP tool_call.locations entry — for tools that touch a specific file:line."""

    path: str
    line: int | None = None


@dataclass
class ToolResult:
    """Lovelaice-level tool execution result. Distinct from lingo.ToolResult."""

    content: list[dict] = field(default_factory=list)
    locations: list[ToolCallLocation] | None = None
    raw_output: Any | None = None
    is_error: bool = False
    terminate: bool = False

    @classmethod
    def from_value(
        cls,
        value: Any,
        *,
        is_error: bool = False,
        raw_output: Any | None = None,
    ) -> "ToolResult":
        """Wrap a tool's bare return value into a ToolResult.

        If the value is already a ToolResult, return it unchanged.
        If it's a string, wrap as a single text content block.
        Otherwise stringify for the LLM, keep the original on raw_output.
        """
        if isinstance(value, ToolResult):
            return value
        if isinstance(value, str):
            return cls(
                content=[{"type": "text", "text": value}],
                is_error=is_error,
                raw_output=raw_output,
            )
        return cls(
            content=[{"type": "text", "text": str(value)}],
            is_error=is_error,
            raw_output=raw_output if raw_output is not None else value,
        )

    @classmethod
    def from_exception(cls, exc: BaseException) -> "ToolResult":
        """Convert an uncaught tool exception into an is_error=True result."""
        text = f"{type(exc).__name__}: {exc}"
        return cls(
            content=[{"type": "text", "text": text}],
            is_error=True,
        )


@dataclass
class AgentTool:
    """Wrapper around a lingo.Tool with agentic metadata."""

    inner: Any  # lingo.Tool
    sequential: bool = False
    kind: ToolKind = "other"
    title_template: str | None = None

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def description(self) -> str:
        return self.inner.description

    def title_for(self, args: dict) -> str:
        """Render the human-readable title for an ACP tool_call payload.

        If title_template is set and all referenced keys are present in args,
        format with them. Otherwise fall back to the bare tool name.
        """
        if self.title_template:
            try:
                return self.title_template.format(**args)
            except KeyError:
                pass
        return self.inner.name


class ToolRegistry:
    """Collection of AgentTools keyed by name."""

    def __init__(self):
        self._by_name: dict[str, AgentTool] = {}

    def register(self, tool: AgentTool) -> None:
        self._by_name[tool.name] = tool

    def get(self, name: str) -> AgentTool | None:
        return self._by_name.get(name)

    def all(self) -> list[AgentTool]:
        return list(self._by_name.values())

    def lingo_tools(self) -> list:
        """Underlying lingo.Tool objects for passing to LLM.chat(tools=...)."""
        return [t.inner for t in self._by_name.values()]

    def any_sequential(self, call_names: list[str]) -> bool:
        """True if any tool referenced by name in call_names has sequential=True.
        Unknown tool names are silently skipped (treated as not sequential)."""
        for n in call_names:
            t = self.get(n)
            if t is not None and t.sequential:
                return True
        return False
