# Precise tool-calling for small models — design

**Date:** 2026-07-02
**Status:** approved (design), pending implementation plan
**Repos touched:** `lovelaice` (primary), `lingo`, `magpie`, `peacock`, `superbot`

## Problem

Small models (9B-class — qwen3.5 9b is the driver here) call tools
imprecisely: they omit required args, invent values for optional ones,
supply wrong types, stringify JSON, or pick keys that don't exist. In
the AInBox stack these models drive `lovelaice` over ACP, and every
malformed call either fails validation (wasting a turn) or executes with
wrong arguments.

Three compounding causes, top to bottom of the stack:

1. **The MCP tool definitions the model sees are uneven.** The ainbox
   MCP servers (`magpie`, `peacock`, `superbot`) ship rich, typed FastMCP
   tools, but the *descriptions* vary from excellent (`magpie.list_dir`,
   `superbot.workflow`) to thin (`peacock_list_projects` one-liner;
   `peacock_write_html(spec: dict)` hands the model an untyped dict for a
   highly-structured artifact spec).

2. **Lovelaice's MCP bridge discards the good schemas.**
   `lovelaice/src/lovelaice/mcp.py::_params_from_input_schema` collapses
   FastMCP's rich `inputSchema` down to `name → python_type`, dropping
   per-param descriptions, defaults/required, enums, `list[X]` item
   types, and nested object schemas. A perfectly-written MCP tool arrives
   at the model as `{spec: {type: object}}` with no guidance.

3. **Lingo's schema builder is lossy.**
   `lingo/lingo/llm.py::tool_to_openai_schema` +
   `_python_type_to_json_schema` emit bare `{"type": ...}` per param:
   no descriptions, everything forced `required` (params with defaults
   included — a real bug), and unrecognized types (`Literal`, `Optional`,
   `list[X]`) collapse to `{"type":"string"}`, losing enums and item
   types.

Even when the model is well-guided, malformed calls still happen; today
the only recovery is to bounce the pydantic validation error back into
the agentic loop, which a 9B recovers from poorly.

## Goal

Lift tool-call precision for 9B-class models driving lovelaice over ACP,
chiefly for ainbox, without adding hot-path cost for models that don't
need it.

Non-goals: touching the legacy TUI tool-call path (`core.py` /
`commands/react.py` / `_on_tool_call`) — ainbox is on the ACP /
`agent.harness` path; the TUI is out of scope. No new tool-authoring API
(no decorators); reuse existing docstrings/signatures.

## Architecture

Three workstreams, ordered by where they sit in the request pipeline —
Workstream 0 improves what the model sees *before* it calls; Workstream 1
fixes how that reaches the wire; Workstream 2 heals the residual bad
calls *after* validation.

### The two rendering paths (why this is safe)

A tool call has two independent "views":

- The **assistant message** `tool_calls` — raw, straight from the model.
- The **`ToolExecutionStart` event** — emitted from inside
  `Harness.execute_tool` with the *validated* args.

The ACP consumer (`acp/server.py`) renders tool calls **only** from
`ToolExecutionStart.args` (`rawInput`); `AssistantMessageFinalized`
forwards text content only. So the raw assistant-message args are never
shown to the ACP client. This is what makes the repair transparent (see
Workstream 2).

## Workstream 0 — MCP tool-definition audit (dev-time, in-plan)

Two prerequisites plus an in-plan rewrite pass.

### 0a. Bridge pass-through (prerequisite, `lovelaice/mcp.py`)

Change the MCP bridge to carry FastMCP's `inputSchema` **verbatim** to
the wire instead of collapsing it to `name → type`. The `_MCPTool`
wrapper retains the original JSON schema dict; the schema builder
(Workstream 1) gets a pass-through path that serializes a pre-built
schema as-is rather than re-deriving one from `parameters()`.

`parameters()` is kept for backwards compatibility (any caller that still
wants `name → type`), but it is no longer the source of the wire schema
for MCP tools.

### 0b. Definition audit + rewrite (in-plan, per app repo)

An in-context pass — done by the implementing agent as plan steps, **not**
a reusable tool/CLI — over each MCP server's tool catalog
(`magpie`/`peacock`/`superbot`). For each tool:

- Flag: missing/weak param docs, untyped `dict` params, absent enums,
  terse one-liner descriptions, and (conversely) over-long descriptions.
- Rewrite the docstring + annotations to be more informative under an
  explicit **no-bloat budget**: description conveys *what it does + when
  to reach for it*, at most one worked cue, no walls of prose. Params get
  a one-line doc each; constrained-choice params get `Literal[...]`
  enums; structured `dict` params either get a typed model or an explicit
  pointer to the schema-discovery tool (e.g. peacock's `peacock_spec`).

Each app repo gets its own commit, shipped to `main` (ainbox convention:
no PR cycle). The improved definitions then flow through the fixed bridge
(0a) + enriched builder (Workstream 1) to every model.

## Workstream 1 — lingo schema quality (`lingo/llm.py`)

Benefits every lingo consumer, not just lovelaice.

- **Honor optional/defaults.** A param with a default becomes *not*
  required and carries its `default` in the schema. Requires exposing
  defaults from the tool signature (introspection on `DelegateTool`),
  which `parameters()` does not surface today.
- **Richer type mapping** in `_python_type_to_json_schema`:
  - `Literal[a, b, ...]` → `{"enum": [a, b, ...]}` (+ inferred `type`).
  - `list[X]` → `{"type": "array", "items": <schema(X)>}`.
  - `Optional[X]` / `X | None` → nullable (schema of `X` + null).
  - `dict` unchanged (`{"type": "object"}`).
- **Per-param descriptions.** Parse the tool's Google-style docstring
  `Args:` block into per-property `description`. No change to how tools
  are authored.
- **Pass-through path.** `tool_to_openai_schema` accepts a pre-built
  JSON schema (set by the MCP bridge in 0a) and serializes it verbatim,
  skipping the lossy `parameters()` round-trip. Native (non-MCP) tools
  continue through the enriched derivation path above.

### Where per-param descriptions come from

Google-style `Args:` docstring parsing was chosen over a new decorator /
metadata API: lovelaice's built-in tools and the ainbox MCP tools already
carry rich docstrings, so this extracts existing value with zero
call-site churn.

## Workstream 2 — focused repair layer (lovelaice, opt-in)

**Seam:** `Harness.execute_tool`, exactly where `validate_args` returns
an error string today.

**Trigger:** *only* pydantic argument-validation failure. Tool
*execution* errors (`tool.inner.run` raising) are never repaired — they
return straight to the loop, unchanged.

**Mechanism (one shot — no repair loop, no agentic step):**

1. Build the pydantic arg model from the tool (reuse existing
   `agent/tools.py::build_arg_model`).
2. Compose a single, focused prompt: tool description + JSON schema +
   the failed args + the pydantic validation error, plus a
   *parameterizable* slice of grounding context.
3. Call `self.llm.create(model, messages)` — lingo's forced-JSON
   `client.chat.completions.parse(response_format=model)` path.
   Constrained decoding against one schema is near-always well-formed.
4. Re-validate the result:
   - **valid** → proceed to run the tool with the repaired args.
   - **`create` raised, or re-validation still fails** → fall back:
     return the original `is_error` ToolResult to the loop (today's
     behavior, unchanged).

**Grounding context** (`repair_context`):

- `none` — schema + failed args + error only (pure shape repair).
- `turn` (default) — also the assistant message that emitted the call +
  the immediately preceding user message. Rescues the *value-error* case
  (model dropped a required arg whose value only exists in the
  conversation) while staying a tiny, focused prompt.
- `full` — the session's message history for the turn.

**History + transparency:**

- Repair runs *before* emitting `ToolExecutionStart`, so
  `ToolExecutionStart.args` carries the repaired args → ACP/stream
  consumers see the fixed args with zero special handling.
- **Rewrite** the in-session assistant message's tool-call arguments to
  the repaired values. Invisible to consumers (they never render from the
  assistant message), and it gives the model clean, well-formed examples
  of its own prior calls — in-context few-shot that compounds over a
  session. The `tool_call_id` correlation is preserved.
- Emit a new **`ToolCallRepaired`** event
  (`call_id, name, original_args, repaired_args, error`) — an optional
  transparency/telemetry channel (consumers can show "auto-fixed";
  evals can measure repair rate). No fake "second tool call" — that would
  make the model believe it invoked the tool twice and would need its own
  id + result, bloating history.

**Config** (`AgentConfig`, `agent/agent.py`):

- `repair_tool_calls: bool = False` — opt-in; ainbox sets `True`.
- `repair_context: Literal["none", "turn", "full"] = "turn"`.

Off by default → zero hot-path cost for models that don't need it.

## Error handling

- Repair failure is non-fatal: falls back to the existing is-error path;
  the loop sees the original validation error as it does today.
- The repair `llm.create` call is wrapped — any exception (provider
  doesn't support structured outputs, timeout, parse failure) is caught
  and treated as "repair failed" → fallback.
- Tool-execution exceptions bypass repair entirely.
- Unknown-tool and permission-block paths are unchanged.

## Testing

**lingo** (`tool_to_openai_schema` / `_python_type_to_json_schema`):

- optional param with default → not in `required`, `default` present.
- `Literal[...]` → `enum`.
- `list[str]` → `array` + `items`.
- `Optional[X]` / `X | None` → nullable.
- Google-style `Args:` docstring → per-property `description`.
- pass-through: a pre-built schema serializes verbatim.

**lovelaice** repair (`Harness.execute_tool`, fake LLM via
`tests/_fake_openrouter.py` driving `create`):

- happy path: bad args → repaired → tool runs with corrected args.
- value recovery: missing required arg recovered from `turn` grounding.
- fallback: repair returns invalid → loop sees original error.
- execution error is NOT repaired.
- history rewrite: in-session assistant tool-call args become the
  repaired args.
- `ToolCallRepaired` emitted on success; not emitted on fallback.
- `repair_tool_calls=False` → repair never fires (behavior identical to
  today).

**lovelaice** MCP bridge (`mcp.py`):

- a FastMCP-style `inputSchema` (with descriptions/enums/defaults) is
  carried through to the wire schema verbatim, not flattened.

**ainbox** app repos: the audited tool definitions are exercised by each
app's existing MCP tests; no new harness — the rewrites are docstring /
annotation changes validated by existing suites.

## Final validation — live smoke with a real 9B

The unit tests above use a fake LLM; they prove the *mechanism* but not
that precision actually improved for a small model. The plan's final step
is a live end-to-end smoke against a real 9B, since that is the whole
point of the effort.

- **Model:** `qwen/qwen3.5-9b` (OpenRouter, confirmed slug, 262k ctx).
- **Endpoint:** `https://openrouter.ai/api/v1`.
- **Key:** workspace token at `/home/apiad/Workspace/.claude/openrouter.token`
  (never overwrite it; read-only use).
- **Harness:** a real `lovelaice` `Agent` on the ACP/`agent.harness` path,
  `repair_tool_calls=True`, driving a representative tool catalog — the
  built-in tools plus a sample of the audited ainbox MCP tools.
- **Fixture:** a small set of prompts that each require a specific,
  non-trivial tool call (right tool, right args, some with optional args
  and enums, some where a value must come from the conversation).

Measure, before vs after the three workstreams:

- **first-try validation rate** — fraction of tool calls that pass
  `validate_args` on the model's first emission (the headline metric;
  Workstreams 0+1 should move this).
- **repair success rate** — of the calls that fail validation, the
  fraction the one-shot repair heals (Workstream 2).
- **end-to-end task success** — the call executes and does the right
  thing.

Pass bar: first-try validation materially up vs a baseline run with the
current `main` (flat schemas, no repair), and repair recovering the clear
majority of the residual failures. This is a manual/scripted step run
once at the end, not part of CI (it costs real tokens and needs network).

## Open questions

None outstanding — all forks resolved during design.
