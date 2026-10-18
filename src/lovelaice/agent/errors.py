"""Lovelaice agent error types and stop-reason enum."""
from enum import Enum


class StopReason(str, Enum):
    """ACP-aligned stop reasons returned from a turn.

    Mirrors the ACP (Agent Client Protocol) StopReason enum so that values
    can be serialized directly onto the wire as strings.
    """

    END_TURN = "end_turn"
    MAX_TOKENS = "max_tokens"
    MAX_TURN_REQUESTS = "max_turn_requests"
    REFUSAL = "refusal"
    CANCELLED = "cancelled"


class AgentError(Exception):
    """Base class for lovelaice.agent errors that should surface to the user."""


class ToolValidationError(AgentError):
    """The LLM produced tool-call arguments that don't validate against the tool's schema."""


class ContextOverflowError(AgentError):
    """The prompt does not fit the model's window, even after compaction."""

    def __init__(self, message: str, *, fixed_tokens: int, usable: int | None,
                 prompt_tokens: int | None):
        super().__init__(message)
        self.message = message
        self.fixed_tokens = fixed_tokens
        self.usable = usable
        self.prompt_tokens = prompt_tokens
