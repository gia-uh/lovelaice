"""Recognising a provider's "the prompt is too long" error.

Each provider words it differently, and lingo passes the provider's exception
through. The phrases below come from messages recorded off real providers
(tests/agent/fixtures/context_errors.json); add a provider by recording its
message there first.
"""
from __future__ import annotations

import re

_PHRASES = (
    "maximum context length",       # OpenRouter, OpenAI
    "exceeds the available context size",  # llama.cpp server
    "exceed_context_size_error",
    "context_length_exceeded",      # OpenAI error code
    "context length",
    "context window",
    "too many tokens",
    "input token limit",
    "maximum input tokens",
    "max input tokens",
    "max_input_tokens",
    "input_token_limit",
)
_PROMPT_PATTERNS = (
    re.compile(r"\((\d+) of text input"),        # OpenRouter
    re.compile(r"request \((\d+) tokens\)"),     # llama.cpp
    re.compile(r"n_prompt_tokens'?\"?:\s*(\d+)"),
    re.compile(r"requested about (\d+) tokens"),
)
_WINDOW_PATTERNS = (
    re.compile(r"maximum context length is (\d+) tokens"),
    re.compile(r"available context size \((\d+) tokens\)"),
    re.compile(r"n_ctx'?\"?:\s*(\d+)"),
)


def _first_int(patterns, text: str) -> int | None:
    for p in patterns:
        m = p.search(text)
        if m:
            return int(m.group(1))
    return None


def context_error_tokens(exc: BaseException) -> tuple[bool, int | None]:
    """Whether `exc` says the prompt does not fit, and the prompt size it names."""
    status = getattr(exc, "status_code", None)
    if status is not None and status not in (400, 413):
        return False, None
    text = str(exc)
    low = text.lower()
    if not any(p in low for p in _PHRASES):
        return False, None
    return True, _first_int(_PROMPT_PATTERNS, text)


def context_error_window(exc: BaseException) -> int | None:
    """The context window a context-length error names, if it names one."""
    return _first_int(_WINDOW_PATTERNS, str(exc).lower())


_INPUT_LIMIT_PATTERNS = (
    re.compile(r"(?:maximum|max) input tokens(?: is|:| of)?\s*(\d+)"),
    re.compile(r"input token limit(?: is|:| of)?\s*(\d+)"),
    re.compile(r"(?:max_input_tokens|input_token_limit)'?\"?:\s*(\d+)"),
)


def context_error_input_limit(exc: BaseException) -> int | None:
    """An explicit input-only limit (no output reserve should be subtracted)."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        for key in ("max_input_tokens", "input_token_limit"):
            value = body.get(key)
            if type(value) is int and value > 0:
                return value
    return _first_int(_INPUT_LIMIT_PATTERNS, str(exc).lower())
