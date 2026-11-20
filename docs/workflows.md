# Workflows

`lovelaice.workflows` is a small async engine for declarative, multi-step work:
a tree of typed nodes that a host runs against its agents and tools. It has no
coupling to a particular host — the caller supplies `agent_factory` (how a
fresh agent is built), optional `handlers` (the host's tools) and an optional
`prompt_handler` (the host's live conversation agent).

```python
from lovelaice import workflow

spec = {
    "name": "digest",
    "root": {
        "kind": "sequence",
        "children": [
            {"kind": "agent", "prompt": "summarise {doc}", "name": "raw"},
            {"kind": "agent", "prompt": "turn this into a haiku: {raw}"},
        ],
    },
}

result = await workflow(spec).run(agent_factory=make_agent)
```

A spec is validated by `WorkflowSpec` (pydantic). Nodes are discriminated by
`kind`, and every node may set `name` to publish its result as a variable that
later prompts can interpolate with `{name}` — a value that is exactly
`"{name}"` keeps the raw object (dict, list, …) instead of being stringified.

| `kind` | What it does |
| --- | --- |
| `agent` | Prompts a **fresh** agent built by `agent_factory`; returns `{"text": …}`, or the validated object when `output_schema` is set. |
| `prompt` | Prompts the **host's live/primary** agent through `prompt_handler` (shared conversation). Not allowed inside a `parallel`/`map` fan-out. |
| `tool` | Calls a host `handler(args, vars)` deterministically — no model involved. |
| `sequence` | Runs children in order; the result is the last child's. |
| `parallel` | Runs children concurrently in isolated variable scopes; collects their results as a list. |
| `map` | Fans `node` over the list in `vars[over]`, binding each element to `{as}` in an isolated scope; collects the results. |

Leaf work is limited by a `MAX_CONCURRENCY` semaphore per run. A fan-out
(`parallel` or `map`) decides what one failing item does to the rest with
`on_error`; see [Partial results](#partial-results-on_error).

## Structured output: `output_schema`

`agent` and `prompt` nodes accept `output_schema`, a JSON Schema describing the
value the step must produce. Small local models are unreliable here: they wrap
the JSON in a fence, prefix it with a sentence, or drift from the shape. The
executor therefore does three things.

1. **Extract** the first JSON value from the final reply. The whole text is
   tried first, then every fenced block in order, then the first `{` or `[`
   from which a complete value decodes — so leading prose and trailing
   commentary are harmless, and prose like `[1]` cannot win over a fenced
   answer. A reply with no JSON value at all fails extraction.
2. **Validate** the value against the schema with `jsonschema` (the dialect is
   chosen by `$schema`, `2020-12` by default). The reported error is the first
   failure in document order, prefixed by its path (`widgets/0/title: …`).
3. **Retry** on failure: the same agent is prompted again with the validation
   error and the schema, up to `retries` more times (`retries: 2` by default,
   `0` means a single attempt). Using the same agent matters — it still holds
   the task and its own wrong answer, so a small model can repair it. Each
   retry counts as a call against the run's concurrency semaphore.

`retries` sits on the node:

```json
{
  "kind": "agent",
  "name": "ficha",
  "prompt": "Fiche this chunk as JSON: {chunk}",
  "output_schema": {
    "type": "object",
    "required": ["afirmaciones", "confianza"],
    "properties": {
      "afirmaciones": {"type": "array", "items": {"type": "string"}},
      "confianza": {"type": "number", "minimum": 0, "maximum": 1}
    },
    "additionalProperties": false
  },
  "retries": 2
}
```

On success the parsed object — not the text — becomes the node's result and the
value of its `name`, so a later `map` can fan over it or a `tool` can receive it
whole. Note that a node with `output_schema` returns the JSON value itself while
a node without one returns `{"text": ...}`; this is the existing contract and
the schema simply says the caller wants a typed result.

If every attempt fails, the run raises `lovelaice.workflows.OutputSchemaError`
(a `RuntimeError`) naming the node and carrying the last error:

```text
OutputSchemaError: node 'ficha': output_schema not satisfied after 3 attempts;
last error: afirmaciones: 'afirmaciones' is a required property
```

An invalid schema is an authoring bug, not a model failure: it raises
`OutputSchemaError` immediately (`invalid output_schema: …`) instead of
spending the retries.

`prompt` nodes enforce the schema the same way, re-prompting the shared live
agent through `prompt_handler`; that agent's conversation therefore shows the
task, the bad answer and the repair request.

## Partial results: `on_error`

A fan-out over dozens of items (one model call each) should not lose everything
because one item misbehaves. `map` and `parallel` nodes take `on_error`:

| Value | On one failed item |
| --- | --- |
| `raise` (default) | The run fails, as it always has. |
| `skip` | The failed item is left out of the result list. |
| `keep` | `{"error": "<message>"}` takes its place, in order. |

```json
{
  "kind": "map",
  "over": "chunks",
  "as": "chunk",
  "name": "fichas",
  "on_error": "keep",
  "node": { "kind": "agent", "prompt": "Fiche this chunk: {chunk}", "output_schema": { "type": "object" } }
}
```

`keep` is the policy to report on: `{"text": …}` entries are successes and
`{"error": …}` entries are failures, so a later `tool` node can count them and
say which items need a retry. `skip` is for when only the successful results
matter and a shorter list is fine. `raise` is unchanged — a plain `gather`, so
the first failure propagates exactly as before.

The exception's message is what lands in `error` (its class name when the
message is empty). Everything a branch can raise is covered, including
`OutputSchemaError` from a child with `output_schema`, so a model that never
produces valid JSON fails one item instead of the run. Cancellation is not an
item failure: a cancelled run still cancels under every policy.

## Host seams

- `agent_factory()` — build a fresh agent (the same seam ACP servers use). The
  executor only calls `prompt(text)` and `messages_for_llm()`.
- `handlers[tool]` — `async (args, vars) -> Any` for `tool` nodes; the engine
  ships none so it stays independent of any concrete tool set.
- `prompt_handler(prompt, vars) -> str` — run a prompt on the host's live agent.
  Required only when the spec contains a `prompt` node.

`run()` also takes `inputs` to seed the variables visible to the first nodes.
