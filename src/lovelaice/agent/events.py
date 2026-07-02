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
