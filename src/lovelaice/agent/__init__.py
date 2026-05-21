"""Lovelaice agentic engine — public surface."""
from lovelaice.agent.agent import Agent, AgentConfig
from lovelaice.agent.tools import (
    AgentTool, ToolRegistry, ToolResult, ToolCallLocation,
    build_arg_model, validate_args,
)
from lovelaice.agent.hooks import Allow, Block, AskUser, HookRegistry
from lovelaice.agent.errors import StopReason, AgentError, ToolValidationError
from lovelaice.agent.session import Session
from lovelaice.agent.harness import Harness
from lovelaice.agent import events

__all__ = [
    "Agent", "AgentConfig",
    "AgentTool", "ToolRegistry", "ToolResult", "ToolCallLocation",
    "build_arg_model", "validate_args",
    "Allow", "Block", "AskUser", "HookRegistry",
    "StopReason", "AgentError", "ToolValidationError",
    "Session", "Harness", "events",
]
