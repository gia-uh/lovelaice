"""Typed progress events emitted by a workflow run.

A host that shows a run live passes ``on_event`` to ``run()`` (or to
``workflow(spec).run()``) and receives these as the run goes:

- :class:`NodeStart` / :class:`NodeEnd` / :class:`NodeError` for every node in
  the tree, before and after it runs;
- :class:`ItemEnd` for every successful item of a ``map``/``parallel`` fan-out,
  at the moment it finishes (completion order, not list order).

Without a callback the engine emits nothing and behaves exactly as before.

``path`` identifies the node in the tree, rooted at ``"root"``: the *i*-th
child of a ``sequence``/``parallel`` is ``{parent}.{i}``, and the *i*-th item
of a ``map`` is ``{parent}[{i}]`` — e.g. ``root.1[3].0`` is the first child of
the sequence run for item 3 of the map at ``root.1``.
"""

from __future__ import annotations

from typing import Any, Literal, Union

from pydantic import BaseModel


class NodeStart(BaseModel):
    """A node is about to run."""

    type: Literal["node_start"] = "node_start"
    path: str
    kind: str
    name: str | None = None


class NodeEnd(BaseModel):
    """A node finished; ``result`` is the same dict the node returned."""

    type: Literal["node_end"] = "node_end"
    path: str
    kind: str
    name: str | None = None
    result: Any = None


class NodeError(BaseModel):
    """A node raised; ``error`` is the exception's message (its class name
    when the message is empty). The exception itself still propagates."""

    type: Literal["node_error"] = "node_error"
    path: str
    kind: str
    name: str | None = None
    error: str


class ItemEnd(BaseModel):
    """One item of a ``map``/``parallel`` fan-out finished successfully.

    ``path`` is the fan-out node's own path, ``index`` the item's position in
    the input list (``0 <= index < total``) and ``result`` what that item
    returned. A failed item emits no ``ItemEnd``: its own node emitted
    ``NodeError`` instead.
    """

    type: Literal["item_end"] = "item_end"
    path: str
    index: int
    total: int
    result: Any = None


WorkflowEvent = Union[NodeStart, NodeEnd, NodeError, ItemEnd]
"""Any event the engine emits."""

__all__ = ["NodeStart", "NodeEnd", "NodeError", "ItemEnd", "WorkflowEvent"]
