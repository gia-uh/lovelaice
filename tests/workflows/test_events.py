"""Progress events from ``run(on_event=...)`` (issue #11)."""

import asyncio

import pytest

from lovelaice import workflow
from lovelaice.workflows import WorkflowSpec, run
from lovelaice.workflows.events import ItemEnd, NodeEnd, NodeError, NodeStart

DELAYS = {"do slow": 0.09, "do mid": 0.03, "do fast": 0.0}


class _Agent:
    """Echoes its prompt back as the reply (so results are identifiable).

    ``prompt`` sleeps for the delay its prompt names, which is how the tests
    force a fan-out to finish out of list order.
    """

    def __init__(self):
        self.prompts = []

    async def prompt(self, text):
        self.prompts.append(text)
        await asyncio.sleep(DELAYS.get(text, 0))
        return "end"

    def messages_for_llm(self):
        reply = self.prompts[-1] if self.prompts else ""

        class _M:
            role = "assistant"
            content = reply

        return [_M()]


def _factory():
    return _Agent()


class _Boom(_Agent):
    async def prompt(self, text):
        self.prompts.append(text)
        raise RuntimeError("model exploded")


@pytest.mark.asyncio
async def test_sequence_with_map_emits_node_and_item_events():
    async def count(args, vars):
        return len(args["items"])

    spec = WorkflowSpec.model_validate(
        {
            "name": "seq-map",
            "root": {
                "kind": "sequence",
                "children": [
                    {"kind": "agent", "prompt": "start", "name": "first"},
                    {
                        "kind": "map",
                        "over": "xs",
                        "as": "x",
                        "name": "res",
                        "node": {"kind": "agent", "prompt": "do {x}"},
                    },
                    {"kind": "tool", "tool": "count", "args": {"items": "{res}"}, "name": "n"},
                ],
            },
        }
    )
    events = []

    async def on_event(event):
        events.append(event)

    out = await run(
        spec,
        agent_factory=_factory,
        handlers={"count": count},
        inputs={"xs": ["a", "b", "c"]},
        on_event=on_event,
    )
    assert out == {"tool": "count", "result": 3}

    # The frame is fully deterministic; the three item runs interleave, so the
    # order of their own events is checked by set below.
    assert events[0] == NodeStart(path="root", kind="sequence", name=None)
    assert events[1] == NodeStart(path="root.0", kind="agent", name="first")
    assert events[2] == NodeEnd(
        path="root.0", kind="agent", name="first", result={"text": "start"}
    )
    assert events[3] == NodeStart(path="root.1", kind="map", name="res")

    item_starts = [e for e in events if isinstance(e, NodeStart) and e.path.startswith("root.1[")]
    assert {e.path for e in item_starts} == {"root.1[0]", "root.1[1]", "root.1[2]"}

    item_ends = [e for e in events if isinstance(e, ItemEnd)]
    assert all(e.path == "root.1" and e.total == 3 for e in item_ends)
    assert sorted(e.index for e in item_ends) == [0, 1, 2]
    assert {e.result["text"] for e in item_ends} == {"do a", "do b", "do c"}
    # Every ItemEnd belongs to this map and arrives before the map itself ends.
    map_end = next(e for e in events if isinstance(e, NodeEnd) and e.path == "root.1")
    assert events.index(map_end) > max(events.index(e) for e in item_ends)

    assert map_end.kind == "map" and map_end.name == "res"
    assert len(map_end.result["items"]) == 3
    assert events[-1] == NodeEnd(
        path="root", kind="sequence", name=None, result={"tool": "count", "result": 3}
    )


@pytest.mark.asyncio
async def test_item_end_is_emitted_in_completion_order_not_list_order():
    spec = WorkflowSpec.model_validate(
        {
            "name": "order",
            "root": {
                "kind": "map",
                "over": "xs",
                "as": "x",
                "name": "res",
                "node": {"kind": "agent", "prompt": "do {x}"},
            },
        }
    )
    items = []

    async def on_event(event):
        if isinstance(event, ItemEnd):
            items.append(event)

    out = await run(
        spec,
        agent_factory=_factory,
        inputs={"xs": ["slow", "fast", "mid"]},
        on_event=on_event,
    )

    assert [e.index for e in items] == [1, 2, 0]  # fast, mid, slow
    assert [e.result["text"] for e in items] == ["do fast", "do mid", "do slow"]
    # The collected list still follows the input order.
    assert [r["text"] for r in out["items"]] == ["do slow", "do fast", "do mid"]


@pytest.mark.asyncio
async def test_node_error_is_emitted_then_the_exception_propagates():
    async def boom(args, vars):
        raise ValueError("handler exploded")

    spec = WorkflowSpec.model_validate(
        {
            "name": "err",
            "root": {"kind": "tool", "tool": "boom", "name": "bad"},
        }
    )
    events = []

    async def on_event(event):
        events.append(event)

    with pytest.raises(ValueError, match="handler exploded"):
        await run(spec, agent_factory=_factory, handlers={"boom": boom}, on_event=on_event)

    assert [type(e) for e in events] == [NodeStart, NodeError]
    assert events[1] == NodeError(
        path="root", kind="tool", name="bad", error="handler exploded"
    )


@pytest.mark.asyncio
async def test_failed_item_inside_map_emits_node_error_but_no_item_end():
    spec = WorkflowSpec.model_validate(
        {
            "name": "keep",
            "root": {
                "kind": "map",
                "over": "xs",
                "as": "x",
                "name": "res",
                "on_error": "keep",
                "node": {"kind": "agent", "prompt": "do {x}"},
            },
        }
    )
    events = []

    async def on_event(event):
        events.append(event)

    out = await run(
        spec,
        agent_factory=_Boom,
        inputs={"xs": ["a", "b"]},
        on_event=on_event,
    )

    assert [r["error"] for r in out["items"]] == ["model exploded", "model exploded"]
    assert not [e for e in events if isinstance(e, ItemEnd)]  # failures are not items done
    child_errors = [e for e in events if isinstance(e, NodeError)]
    assert {e.path for e in child_errors} == {"root[0]", "root[1]"}
    assert {e.error for e in child_errors} == {"model exploded"}
    # The map itself succeeded, with the kept error entries.
    assert isinstance(events[-1], NodeEnd) and events[-1].path == "root"


@pytest.mark.asyncio
async def test_paths_nest_through_map_items_and_sequences():
    spec = WorkflowSpec.model_validate(
        {
            "name": "nested",
            "root": {
                "kind": "map",
                "over": "xs",
                "as": "x",
                "node": {
                    "kind": "sequence",
                    "children": [
                        {"kind": "agent", "prompt": "one {x}"},
                        {"kind": "agent", "prompt": "two {x}"},
                    ],
                },
            },
        }
    )
    paths = []

    async def on_event(event):
        if isinstance(event, NodeStart):
            paths.append(event.path)

    await run(spec, agent_factory=_factory, inputs={"xs": ["a", "b"]}, on_event=on_event)
    assert paths[0] == "root"
    assert set(paths[1:]) == {"root[0]", "root[0].0", "root[0].1", "root[1]", "root[1].0", "root[1].1"}


@pytest.mark.asyncio
async def test_workflow_entrypoint_accepts_on_event():
    spec = {
        "name": "entry",
        "root": {"kind": "agent", "prompt": "hi", "name": "only"},
    }
    events = []

    async def on_event(event):
        events.append(event)

    await workflow(spec).run(agent_factory=_factory, on_event=on_event)
    assert [type(e) for e in events] == [NodeStart, NodeEnd]
    assert events[0].kind == "agent" and events[1].result == {"text": "hi"}


@pytest.mark.asyncio
async def test_sync_callback_is_tolerated_and_runs_unchanged():
    spec = WorkflowSpec.model_validate(
        {"name": "sync", "root": {"kind": "agent", "prompt": "hi"}}
    )
    events = []
    out = await run(spec, agent_factory=_factory, on_event=events.append)
    assert out == {"text": "hi"}
    assert len(events) == 2
