"""Typed AgentEvent dataclasses emitted by the harness.

Subscribers receive these in-process; the ACP server translates the relevant
ones to ACP session/update notifications.
"""
from dataclasses import dataclass
from typing import Any

from lovelaice.agent.errors import StopReason


@dataclass
class AgentEvent:
    """Base class for all agent events. Subclasses are simple dataclasses."""


@dataclass
class TurnStart(AgentEvent):
    turn_no: int
    model: str


@dataclass
class TurnEnd(AgentEvent):
    stop_reason: StopReason
    soft_terminate: bool


@dataclass
class AssistantMessageDelta(AgentEvent):
    """One streamed content token from the current assistant message. Emitted
    per token when the LLM streams (via lingo's on_token). Additive: consumers
    that don't stream simply ignore it and use AssistantMessageFinalized."""
    text: str


@dataclass
class AssistantMessageFinalized(AgentEvent):
    message: Any  # lingo.Message — typed loosely here to avoid import cycle


@dataclass
class ToolExecutionStart(AgentEvent):
    call_id: str
    name: str
    args: dict


@dataclass
class ToolExecutionUpdate(AgentEvent):
    call_id: str
    partial_content: str


@dataclass
class ToolExecutionEnd(AgentEvent):
    call_id: str
    result: Any  # lovelaice ToolResult — avoid cycle
    is_error: bool


@dataclass
class ToolCallRepaired(AgentEvent):
    """A tool call whose args failed validation was repaired by a focused,
    forced-JSON LLM shot before execution. Transparency/telemetry channel."""

    call_id: str
    name: str
    original_args: dict
    repaired_args: dict
    error: str


@dataclass
class SessionAppend(AgentEvent):
    entry_id: str
    entry_type: str


@dataclass
class ContextCompacted(AgentEvent):
    """The view sent to the model was cut down. `report` is a
    lovelaice.agent.budget.CompactionReport."""
    report: Any


@dataclass
class ContextOverheadWarning(AgentEvent):
    """The system prompt and tool schemas alone are over the usable window.
    The call is sent anyway; the provider decides."""
    fixed_tokens: int
    usable: int


@dataclass
class ContextOverflow(AgentEvent):
    """The provider refused the prompt even after compaction; the turn ends."""
    message: str
    fixed_tokens: int
    usable: int | None
    prompt_tokens: int | None
