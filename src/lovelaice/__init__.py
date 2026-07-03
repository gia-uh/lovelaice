"""Lovelaice — a sovereign coding agent for the terminal.

The engine is the native `agent` package: `Agent` + `ReActNative` (real
tool-calls). The CLI (`lovelaice.cli`) drives it via an in-process ACP client;
`lovelaice-acp` exposes it over stdio ACP.
"""
from .agent import Agent, AgentConfig
from .workflows import workflow

__version__ = "2.4.0"
__all__ = ["Agent", "AgentConfig", "workflow", "__version__"]
