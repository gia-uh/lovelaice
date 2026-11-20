"""``on_error`` on ``map`` and ``parallel`` nodes: keep partial results.

One bad item (a model that misbehaves, an ``output_schema`` that never
validates, a tool that raises) must not throw away every result already
computed in a fan-out over dozens of items.
"""

import pytest
from pydantic import ValidationError

from lovelaice.workflows import (
    MapNode,
    OutputSchemaError,
    ParallelNode,
    WorkflowSpec,
    run,
)


class _Agent:
    """Replies ``echo:<prompt>``, or raises when the prompt names a bad item."""

    def __init__(self, fail_on="bad", exc_factory=None):
        self._fail_on = fail_on
        self._exc_factory = exc_factory or (lambda text: RuntimeError(f"boom: {text}"))
        self._reply = ""

    async def prompt(self, text):
        if self._fail_on is not None and self._fail_on in text:
            raise self._exc_factory(text)
        self._reply = f"echo:{text}"
        return "end"

    def messages_for_llm(self):
        reply = self._reply

        class _M:
            role = "assistant"
            content = reply

        return [_M()]


def _map_spec(**node_extra) -> WorkflowSpec:
    return WorkflowSpec.model_validate({
        "name": "m",
        "root": {
            "kind": "map",
            "over": "xs",
            "as": "x",
            "name": "res",
            "node": {"kind": "agent", "prompt": "do {x}"},
            **node_extra,
        },
    })


# --- model surface -------------------------------------------------------


def test_on_error_defaults_to_raise():
    mp = WorkflowSpec.model_validate({
        "name": "m",
        "root": {"kind": "map", "over": "xs", "as": "x",
                 "node": {"kind": "agent", "prompt": "do {x}"}},
    }).root
    par = WorkflowSpec.model_validate({
        "name": "p",
        "root": {"kind": "parallel", "children": [
            {"kind": "agent", "prompt": "one"},
        ]},
    }).root
    assert isinstance(mp, MapNode) and mp.on_error == "raise"
    assert isinstance(par, ParallelNode) and par.on_error == "raise"


def test_on_error_accepts_the_three_policies():
    for policy in ("raise", "skip", "keep"):
        node = _map_spec(on_error=policy).root
        assert node.on_error == policy


def test_on_error_rejects_an_unknown_policy():
    with pytest.raises(ValidationError):
        _map_spec(on_error="ignore")


# --- map -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_map_raise_is_the_default_and_propagates():
    def factory():
        return _Agent()

    with pytest.raises(RuntimeError, match="boom: do bad"):
        await run(_map_spec(), agent_factory=factory, inputs={"xs": ["a", "bad", "c"]})


@pytest.mark.asyncio
async def test_map_skip_leaves_failed_items_out_and_keeps_order():
    def factory():
        return _Agent()

    result = await run(
        _map_spec(on_error="skip"),
        agent_factory=factory,
        inputs={"xs": ["a", "bad", "c"]},
    )
    assert result["items"] == [{"text": "echo:do a"}, {"text": "echo:do c"}]


@pytest.mark.asyncio
async def test_map_skip_publishes_the_shorter_list_under_the_node_name():
    observed = {}

    async def handler(args, vars):
        observed.update(vars)
        return "ok"

    spec = WorkflowSpec.model_validate({
        "name": "m",
        "root": {"kind": "sequence", "children": [
            _map_spec(on_error="skip").root.model_dump(by_alias=True),
            {"kind": "tool", "tool": "peek"},
        ]},
    })
    await run(
        spec,
        agent_factory=lambda: _Agent(),
        handlers={"peek": handler},
        inputs={"xs": ["a", "bad"]},
    )
    assert observed["res"] == [{"text": "echo:do a"}]


@pytest.mark.asyncio
async def test_map_keep_puts_an_error_entry_in_place():
    def factory():
        return _Agent()

    result = await run(
        _map_spec(on_error="keep"),
        agent_factory=factory,
        inputs={"xs": ["a", "bad", "c"]},
    )
    items = result["items"]
    assert len(items) == 3
    assert items[0] == {"text": "echo:do a"}
    assert items[1] == {"error": "boom: do bad"}
    assert items[2] == {"text": "echo:do c"}


@pytest.mark.asyncio
async def test_map_keep_lets_a_tool_count_the_failures():
    """The consumer's use case: report how many items failed and why."""
    counts = {}

    async def count(args, vars):
        items = vars["res"]
        counts["failed"] = [i for i in items if "error" in i]
        counts["ok"] = len(items) - len(counts["failed"])
        return counts

    spec = WorkflowSpec.model_validate({
        "name": "m",
        "root": {"kind": "sequence", "children": [
            _map_spec(on_error="keep").root.model_dump(by_alias=True),
            {"kind": "tool", "tool": "count"},
        ]},
    })
    await run(
        spec,
        agent_factory=lambda: _Agent(),
        handlers={"count": count},
        inputs={"xs": ["a", "bad", "alsobad"]},
    )
    assert counts == {"failed": [{"error": "boom: do bad"},
                                  {"error": "boom: do alsobad"}], "ok": 1}


@pytest.mark.asyncio
async def test_map_keep_captures_an_output_schema_failure():
    """The real fan-out case: the item's schema never validates."""

    class _BadJSON(_Agent):
        async def prompt(self, text):
            self._reply = "not json at all"
            return "end"

    spec = WorkflowSpec.model_validate({
        "name": "m",
        "root": {
            "kind": "map",
            "over": "xs",
            "as": "x",
            "name": "res",
            "on_error": "keep",
            "node": {
                "kind": "agent",
                "prompt": "do {x}",
                "output_schema": {"type": "object", "required": ["k"]},
            },
        },
    })
    result = await run(
        spec,
        agent_factory=lambda: _BadJSON(fail_on=None),
        inputs={"xs": ["a"]},
    )
    assert len(result["items"]) == 1
    assert "output_schema not satisfied" in result["items"][0]["error"]


@pytest.mark.asyncio
async def test_map_raise_still_propagates_output_schema_error():
    class _BadJSON(_Agent):
        async def prompt(self, text):
            self._reply = "not json at all"
            return "end"

    spec = WorkflowSpec.model_validate({
        "name": "m",
        "root": {
            "kind": "map",
            "over": "xs",
            "as": "x",
            "node": {
                "kind": "agent",
                "prompt": "do {x}",
                "output_schema": {"type": "object", "required": ["k"]},
                "retries": 0,
            },
        },
    })
    with pytest.raises(OutputSchemaError):
        await run(spec, agent_factory=lambda: _BadJSON(fail_on=None),
                  inputs={"xs": ["a"]})


# --- parallel ------------------------------------------------------------


def _parallel_spec(**node_extra) -> WorkflowSpec:
    return WorkflowSpec.model_validate({
        "name": "p",
        "root": {
            "kind": "parallel",
            "name": "outs",
            "children": [
                {"kind": "agent", "prompt": "do good-one"},
                {"kind": "agent", "prompt": "do bad-two"},
                {"kind": "agent", "prompt": "do good-three"},
            ],
            **node_extra,
        },
    })


@pytest.mark.asyncio
async def test_parallel_skip_drops_only_the_failed_child():
    result = await run(
        _parallel_spec(on_error="skip"),
        agent_factory=lambda: _Agent(),
    )
    assert result["items"] == [
        {"text": "echo:do good-one"},
        {"text": "echo:do good-three"},
    ]


@pytest.mark.asyncio
async def test_parallel_keep_keeps_every_child_in_order():
    result = await run(
        _parallel_spec(on_error="keep"),
        agent_factory=lambda: _Agent(),
    )
    assert result["items"] == [
        {"text": "echo:do good-one"},
        {"error": "boom: do bad-two"},
        {"text": "echo:do good-three"},
    ]


@pytest.mark.asyncio
async def test_parallel_raise_is_the_default():
    with pytest.raises(RuntimeError, match="boom: do bad-two"):
        await run(_parallel_spec(), agent_factory=lambda: _Agent())


# --- cancellation is not an item failure --------------------------------


@pytest.mark.asyncio
async def test_keep_does_not_swallow_cancellation():
    """A cancelled run must still cancel, not become an error entry."""
    import asyncio

    class _Cancelled(_Agent):
        async def prompt(self, text):
            raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await run(
            _map_spec(on_error="keep"),
            agent_factory=lambda: _Cancelled(),
            inputs={"xs": ["a"]},
        )
