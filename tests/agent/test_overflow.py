"""Recognising context-length errors, from messages recorded off real providers."""
import json
from pathlib import Path

import pytest

from lovelaice.agent.overflow import context_error_tokens, context_error_window

CASES = json.loads(
    (Path(__file__).parent / "fixtures" / "context_errors.json").read_text())["cases"]


class RecordedError(Exception):
    def __init__(self, message, status_code):
        super().__init__(message)
        self.status_code = status_code


def rebuild(case):
    return RecordedError(case["message"], case["status_code"])


@pytest.mark.parametrize("case", CASES, ids=[f'{c["provider"]}-{c["type"]}' for c in CASES])
def test_context_error_tokens_matches_recorded(case):
    is_ctx, tokens = context_error_tokens(rebuild(case))
    assert is_ctx == case["is_context"]
    assert tokens == case["prompt_tokens"]


def test_window_named_by_provider():
    by = {c["provider"]: c for c in CASES if c["is_context"]}
    assert context_error_window(rebuild(by["openrouter"])) == 16384
    assert context_error_window(rebuild(by["llama.cpp"])) == 2048


def test_context_words_on_a_500_are_not_a_context_error():
    exc = RecordedError("internal error while computing context length", 500)
    assert context_error_tokens(exc) == (False, None)


def test_plain_exception_without_status():
    assert context_error_tokens(ValueError("boom")) == (False, None)
