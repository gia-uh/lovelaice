# Precise tool-calling for small models — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans
> to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Raise tool-call precision for 9B-class models driving lovelaice over
ACP, by enriching the schemas the model sees, preserving MCP schemas end-to-end,
auditing the ainbox MCP definitions, and adding an opt-in one-shot repair layer.

**Architecture:** Three layers. (1) `lingo` builds richer OpenAI tool schemas
(optional/defaults, enums, arrays, nullable, per-param descriptions) and gains a
pass-through path for pre-built schemas. (2) `lovelaice`'s MCP bridge stops
flattening FastMCP `inputSchema` and carries it verbatim; the ainbox MCP tool
definitions get an in-plan audit pass. (3) `lovelaice`'s `Harness.execute_tool`
gains an opt-in focused repair: on pydantic arg-validation failure, one forced-JSON
`llm.create` shot heals the args, rewrites session history, and emits a
`ToolCallRepaired` event before falling back to the loop.

**Tech Stack:** Python 3.12+, pydantic v2, `lingo` (LLM framework), `lovelaice`
(agent runtime), FastMCP (ainbox servers), pytest / pytest-asyncio.

**Design:** `docs/2026-07-02-precise-tool-calling-design.md`.

## Global Constraints

- Design is canonical; this plan implements it verbatim.
- TDD throughout: failing test → minimal code → green → commit.
- `lingo` and `lovelaice` are independent git repos — commit in each with
  conventional messages. ainbox app repos (`magpie`/`peacock`/`superbot`) ship to
  `main`, one commit each.
- Repair is **off by default** (`AgentConfig.repair_tool_calls=False`) — zero
  hot-path change for models that don't opt in.
- Repair fires **only** on pydantic arg-validation failure, **never** on
  tool-execution errors, unknown-tool, or permission-block.
- No new tool-authoring API; reuse existing docstrings/signatures.
- Do not touch the legacy TUI tool-call path (`core.py`, `commands/react.py`).
- Run `uv run pytest` in each repo before every commit; never pipe the gate.

---

## Task 1: lingo — richer type mapping (`_python_type_to_json_schema`)

**Files:**
- Modify: `lingo/lingo/llm.py` (`_python_type_to_json_schema`, ~line 231)
- Test: `lingo/tests/test_tool_schema.py` (create)

**Interfaces:**
- Produces: `_python_type_to_json_schema(t) -> dict` now handles `Literal[...]`
  → `{"enum":[...], "type": <inferred>}`, `list[X]` → `{"type":"array","items":
  <schema(X)>}`, `Optional[X]`/`X|None` → `{"type":[<t>,"null"]}` (schema of X
  plus null), falling back to `{"type":"string"}` as today.

- [ ] **Step 1: Write failing tests**

```python
# lingo/tests/test_tool_schema.py
from typing import Literal, Optional
from lingo.llm import _python_type_to_json_schema as j

def test_literal_becomes_enum():
    s = j(Literal["a", "b"])
    assert s["enum"] == ["a", "b"]
    assert s["type"] == "string"

def test_list_becomes_array_with_items():
    s = j(list[str])
    assert s["type"] == "array"
    assert s["items"] == {"type": "string"}

def test_optional_is_nullable():
    s = j(Optional[int])
    assert "null" in s["type"] and "integer" in s["type"]

def test_plain_types_unchanged():
    assert j(str) == {"type": "string"}
    assert j(int) == {"type": "integer"}
    assert j(bool) == {"type": "boolean"}
```

- [ ] **Step 2: Run to verify failure** — `cd lingo && uv run pytest tests/test_tool_schema.py -x` → FAIL (enum/items/null not produced).

- [ ] **Step 3: Implement** — extend `_python_type_to_json_schema`:

```python
def _python_type_to_json_schema(t) -> dict:
    import typing
    origin = typing.get_origin(t)
    args = typing.get_args(t)
    # Literal -> enum
    if origin is typing.Literal:
        vals = list(args)
        base = {str: "string", int: "integer", float: "number", bool: "boolean"}
        elem = base.get(type(vals[0]), "string") if vals else "string"
        return {"type": elem, "enum": vals}
    # Optional[X] / X | None
    if origin in (typing.Union, getattr(__import__("types"), "UnionType", None)):
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            inner = _python_type_to_json_schema(non_none[0])
            it = inner.get("type", "string")
            inner["type"] = [it, "null"] if isinstance(it, str) else it
            return inner
    # list[X]
    if t is list or origin is list:
        items = _python_type_to_json_schema(args[0]) if args else {"type": "string"}
        return {"type": "array", "items": items}
    if t is dict or origin is dict:
        return {"type": "object"}
    if t is str: return {"type": "string"}
    if t is int: return {"type": "integer"}
    if t is float: return {"type": "number"}
    if t is bool: return {"type": "boolean"}
    return {"type": "string"}
```

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/test_tool_schema.py -x` → PASS.

- [ ] **Step 5: Commit** — `cd lingo && git add -A && git commit -m "feat(schema): map Literal/list/Optional to enum/array/nullable"`

---

## Task 2: lingo — expose defaults + docstring param docs; honor them in the schema

**Files:**
- Modify: `lingo/lingo/tools.py` (`DelegateTool` — add `defaults()` and
  `param_docs()`)
- Modify: `lingo/lingo/llm.py` (`tool_to_openai_schema`, ~line 250)
- Test: `lingo/tests/test_tool_schema.py` (extend)

**Interfaces:**
- Produces: `Tool.defaults() -> dict[str, Any]` (params with signature defaults;
  base `Tool` returns `{}`), `Tool.param_docs() -> dict[str, str]` (parsed from the
  Google-style `Args:` docstring block; base returns `{}`).
- Consumes (in `tool_to_openai_schema`): `tool.defaults()`, `tool.param_docs()`,
  Task 1's `_python_type_to_json_schema`. `required` = params **without** a
  default; each property gains `description` (if documented) and `default` (if
  present).

- [ ] **Step 1: Write failing tests**

```python
from lingo.tools import tool as lingo_tool
from lingo.llm import tool_to_openai_schema

@lingo_tool
async def grep(pattern: str, path: str = ".") -> str:
    """Search files.

    Args:
        pattern: Regex to search for.
        path: Directory to search under.
    """
    return ""

def test_default_param_not_required_and_documented():
    s = tool_to_openai_schema(grep)["function"]
    props = s["parameters"]["properties"]
    assert s["parameters"]["required"] == ["pattern"]
    assert props["path"]["default"] == "."
    assert props["pattern"]["description"] == "Regex to search for."
    assert props["path"]["description"] == "Directory to search under."
```

- [ ] **Step 2: Run to verify failure** — FAIL (`path` still required, no descriptions).

- [ ] **Step 3: Implement.** In `lingo/lingo/tools.py`, add to `Tool` base:

```python
    def defaults(self) -> dict:
        return {}
    def param_docs(self) -> dict:
        return {}
```

and to `DelegateTool`:

```python
    def defaults(self) -> dict:
        sig = inspect.signature(self._target)
        return {n: p.default for n, p in sig.parameters.items()
                if not n.startswith("_") and p.default is not inspect.Parameter.empty
                and not isinstance(p.default, _Depends)}

    def param_docs(self) -> dict:
        return _parse_args_docstring(self._description or "")
```

Add module-level `_parse_args_docstring(doc)` (Google-style `Args:` block → `{name: one-line desc}`; tolerant of missing block → `{}`). Then rewrite
`tool_to_openai_schema` in `llm.py`:

```python
def tool_to_openai_schema(tool_obj) -> dict:
    if getattr(tool_obj, "json_schema", None):        # pass-through (Task 3)
        params_schema = tool_obj.json_schema
    else:
        params = tool_obj.parameters()
        defaults = tool_obj.defaults()
        docs = tool_obj.param_docs()
        properties = {}
        for name, ptype in params.items():
            prop = _python_type_to_json_schema(ptype)
            if name in docs: prop["description"] = docs[name]
            if name in defaults: prop["default"] = defaults[name]
            properties[name] = prop
        params_schema = {
            "type": "object",
            "properties": properties,
            "required": [n for n in params if n not in defaults],
        }
    return {"type": "function", "function": {
        "name": tool_obj.name,
        "description": tool_obj.description.strip(),
        "parameters": params_schema,
    }}
```

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/test_tool_schema.py -x` → PASS.

- [ ] **Step 5: Commit** — `git commit -m "feat(schema): honor optional defaults + docstring Args param descriptions"`

---

## Task 3: lingo — pass-through path for pre-built schemas

**Files:**
- Modify: `lingo/lingo/tools.py` (add optional `json_schema` attr on `Tool`)
- Test: `lingo/tests/test_tool_schema.py` (extend)

**Interfaces:**
- Produces: any `Tool` may carry `.json_schema: dict | None` (default `None`). When
  set, `tool_to_openai_schema` (Task 2) emits it verbatim as `parameters`.

- [ ] **Step 1: Failing test**

```python
def test_prebuilt_schema_passthrough():
    grep.json_schema = {"type": "object",
        "properties": {"pattern": {"type": "string", "description": "rich"}},
        "required": ["pattern"]}
    s = tool_to_openai_schema(grep)["function"]
    assert s["parameters"]["properties"]["pattern"]["description"] == "rich"
    grep.json_schema = None  # reset
```

- [ ] **Step 2: Run to verify failure** — FAIL (`AttributeError`/ignored).

- [ ] **Step 3: Implement** — in `Tool.__init__`, `self.json_schema = None`. (The
  `tool_to_openai_schema` branch from Task 2 already reads it.)

- [ ] **Step 4: Run to verify pass** — PASS.

- [ ] **Step 5: Commit** — `git commit -m "feat(schema): pass-through pre-built json_schema on Tool"`

---

## Task 4: lovelaice — MCP bridge carries inputSchema verbatim

**Files:**
- Modify: `lovelaice/src/lovelaice/mcp.py` (`_MCPTool`, `_wrap_mcp_tool`, ~line 124)
- Test: `lovelaice/tests/test_mcp.py` (extend)

**Interfaces:**
- Consumes: Task 3's `Tool.json_schema`.
- Produces: `_MCPTool` sets `self.json_schema = <original inputSchema dict>` so the
  lingo builder serializes it verbatim; `parameters()` still returns the flattened
  `name→type` map for back-compat.

- [ ] **Step 1: Failing test**

```python
def test_mcp_tool_carries_input_schema_verbatim():
    from lovelaice.mcp import _wrap_mcp_tool
    class FakeTool:
        name = "read_note"
        description = "Read a note."
        inputSchema = {"type": "object",
            "properties": {"vault_id": {"type": "string", "description": "vault"},
                           "path": {"type": "string", "description": "note path"}},
            "required": ["vault_id", "path"]}
    t = _wrap_mcp_tool(server_name="magpie", tool=FakeTool(), session=None)
    assert t.json_schema == FakeTool.inputSchema
```

- [ ] **Step 2: Run to verify failure** — `cd lovelaice && uv run pytest tests/test_mcp.py::test_mcp_tool_carries_input_schema_verbatim -x` → FAIL.

- [ ] **Step 3: Implement** — `_MCPTool.__init__` gains `json_schema` param and sets
  `self.json_schema = json_schema`; `_wrap_mcp_tool` passes
  `json_schema=getattr(tool, "inputSchema", None) or None`.

- [ ] **Step 4: Run to verify pass** — PASS. Also run `uv run pytest tests/test_mcp.py tests/test_mcp_http.py -x`.

- [ ] **Step 5: Commit** — `cd lovelaice && git add -A && git commit -m "feat(mcp): carry FastMCP inputSchema verbatim to the wire"`

---

## Task 5: lovelaice — repair config, event, and harness threading

**Files:**
- Modify: `lovelaice/src/lovelaice/agent/events.py` (add `ToolCallRepaired`)
- Modify: `lovelaice/src/lovelaice/agent/agent.py` (`AgentConfig` fields; pass to
  Harness; set `harness.session`)
- Modify: `lovelaice/src/lovelaice/agent/harness.py` (`Harness.__init__` accepts
  `repair_tool_calls`, `repair_context`; `self.session = None`)
- Modify: `lovelaice/src/lovelaice/agent/session.py` (`update_tool_call_args`)
- Test: `lovelaice/tests/agent/test_session.py`, `tests/agent/test_events.py`

**Interfaces:**
- Produces:
  - `ToolCallRepaired(call_id: str, name: str, original_args: dict,
    repaired_args: dict, error: str)` dataclass event.
  - `AgentConfig.repair_tool_calls: bool = False`,
    `AgentConfig.repair_context: Literal["none","turn","full"] = "turn"`.
  - `Harness.repair_tool_calls`, `Harness.repair_context`, `Harness.session`
    (default `None`).
  - `Session.update_tool_call_args(call_id: str, new_args: dict) -> None` — mutates
    the stored assistant entry whose `tool_calls[].id == call_id`, setting its
    `arguments`; no-op if not found. In-memory `_entries` only (JSONL stays as the
    honest emitted record).

- [ ] **Step 1: Failing tests**

```python
# tests/agent/test_session.py — new test
def test_update_tool_call_args_rewrites_entry(tmp_path):
    from lingo.llm import Message, ToolCall
    s = Session.create(tmp_path/"s.jsonl", model="m",
                       system_prompt_hash="h", loop="L", cwd=".")
    s.append(Message.assistant("", tool_calls=[ToolCall(id="c1", name="grep",
             arguments={"pattern": "x"})]))
    s.update_tool_call_args("c1", {"pattern": "x", "path": "."})
    msgs = s.messages_for_llm("sys")
    assert msgs[-1].tool_calls[0].arguments == {"pattern": "x", "path": "."}
```

```python
# tests/agent/test_events.py — new test
def test_tool_call_repaired_event_shape():
    from lovelaice.agent.events import ToolCallRepaired
    e = ToolCallRepaired(call_id="c1", name="grep",
        original_args={"pattern": "x"}, repaired_args={"pattern": "x", "path": "."},
        error="path required")
    assert e.repaired_args["path"] == "."
```

- [ ] **Step 2: Run to verify failure** — both FAIL (attr/method missing).

- [ ] **Step 3: Implement** — add the dataclass to `events.py`; add the two
  `AgentConfig` fields; thread them through `Agent.__init__` into `Harness(...)`
  and set `self.harness.session = self.session` after the session is built (both in
  `__init__` and `from_conversation`); add `session=None`, `repair_tool_calls`,
  `repair_context` to `Harness.__init__`; implement `Session.update_tool_call_args`
  (walk `self._entries` in reverse, find `type=="message"` entry with a matching
  `tool_calls` id, update `arguments`).

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/agent/test_session.py tests/agent/test_events.py -x` → PASS.

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(agent): repair config, ToolCallRepaired event, session arg-rewrite"`

---

## Task 6: lovelaice — focused repair in `Harness.execute_tool`

**Files:**
- Modify: `lovelaice/src/lovelaice/agent/harness.py` (`execute_tool`; add
  `_repair_args` helper)
- Test: `lovelaice/tests/agent/test_repair.py` (create)

**Interfaces:**
- Consumes: Task 5's `Harness.repair_tool_calls`/`repair_context`/`session`,
  `ToolCallRepaired`; existing `build_arg_model`, `validate_args`, `ToolResult`.
- Produces: when `repair_tool_calls` is on and `validate_args` fails, `execute_tool`
  runs one `self.llm.create(model, messages)` shot, re-validates, and on success:
  mutates `call.arguments`, calls `self.session.update_tool_call_args(call.id, ...)`
  (if session set), emits `ToolCallRepaired`, then proceeds (emits
  `ToolExecutionStart` with repaired args → runs tool). On failure → today's
  is-error path. Repair is skipped entirely when the flag is off.

- [ ] **Step 1: Write failing tests** (stub `llm` with async `create`):

```python
# tests/agent/test_repair.py
import pytest
from lingo.tools import tool as lingo_tool
from lingo.llm import ToolCall
from lovelaice.agent.tools import AgentTool, ToolRegistry, build_arg_model
from lovelaice.agent.hooks import HookRegistry
from lovelaice.agent.harness import Harness
from lovelaice.agent.events import ToolCallRepaired, ToolExecutionStart

@lingo_tool
async def grep(pattern: str, path: str) -> str:
    """Search. Args: pattern: rx. path: dir."""
    return f"{pattern}@{path}"

def _h(llm, repair=True):
    reg = ToolRegistry(); reg.register(AgentTool(inner=grep))
    h = Harness(llm=llm, tools=reg, hooks=HookRegistry(), system_prompt="x",
                repair_tool_calls=repair, repair_context="none")
    return h

class FakeLLM:
    def __init__(self, obj): self.obj = obj; self.calls = 0
    async def create(self, model, messages, **kw):
        self.calls += 1
        return model(**self.obj)

@pytest.mark.asyncio
async def test_repair_heals_missing_arg():
    llm = FakeLLM({"pattern": "x", "path": "."})
    h = _h(llm)
    events = []; h.subscribe(events.append)
    call = ToolCall(id="c1", name="grep", arguments={"pattern": "x"})  # missing path
    r = await h.execute_tool(call)
    assert r.is_error is False
    assert r.content[0]["text"] == "x@."
    assert call.arguments == {"pattern": "x", "path": "."}
    assert llm.calls == 1
    assert any(isinstance(e, ToolCallRepaired) for e in events)
    start = [e for e in events if isinstance(e, ToolExecutionStart)][0]
    assert start.args == {"pattern": "x", "path": "."}

@pytest.mark.asyncio
async def test_repair_disabled_returns_error():
    h = _h(FakeLLM({}), repair=False)
    r = await h.execute_tool(ToolCall(id="c1", name="grep", arguments={"pattern": "x"}))
    assert r.is_error is True and "validation failed" in r.content[0]["text"]

@pytest.mark.asyncio
async def test_repair_failure_falls_back_to_error():
    class BadLLM:
        async def create(self, model, messages, **kw):
            raise RuntimeError("no structured output")
    h = _h(BadLLM())
    r = await h.execute_tool(ToolCall(id="c1", name="grep", arguments={"pattern": "x"}))
    assert r.is_error is True

@pytest.mark.asyncio
async def test_execution_error_not_repaired():
    @lingo_tool
    async def boom(x: str) -> str:
        """boom. Args: x: v."""
        raise RuntimeError("boom")
    reg = ToolRegistry(); reg.register(AgentTool(inner=boom))
    llm = FakeLLM({"x": "y"})
    h = Harness(llm=llm, tools=reg, hooks=HookRegistry(), system_prompt="x",
                repair_tool_calls=True, repair_context="none")
    r = await h.execute_tool(ToolCall(id="c1", name="boom", arguments={"x": "y"}))
    assert r.is_error is True
    assert llm.calls == 0  # valid args → no repair; execution error not repaired
```

- [ ] **Step 2: Run to verify failure** — `cd lovelaice && uv run pytest tests/agent/test_repair.py -x` → FAIL.

- [ ] **Step 3: Implement** — in `execute_tool`, replace the `validate_args`
  early-return with: if invalid and `self.repair_tool_calls`, call
  `repaired = await self._repair_args(tool, call, error_str)`; if `repaired` is a
  dict, mutate `call.arguments = repaired`, `update_tool_call_args` (guarded by
  `self.session`), emit `ToolCallRepaired`, set `validated = repaired`, continue;
  else return the is-error result. `_repair_args` builds the model via
  `build_arg_model(tool)`, composes the focused messages (schema + failed args +
  error + grounding per `repair_context`), calls `self.llm.create(...)` inside
  try/except, re-validates via `validate_args`, returns dict or `None`.

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/agent/test_repair.py -x` then full `uv run pytest` → PASS.

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(agent): opt-in one-shot forced-JSON tool-arg repair"`

---

## Task 7: ainbox — audit + rewrite MCP tool definitions (magpie/peacock/superbot)

**Files:**
- Modify: `magpie/src/magpie/mcp_server.py`
- Modify: `peacock/src/peacock/mcp_server.py`
- Modify: `superbot/src/superbot/mcp_superbot.py` (+ `mcp_web.py`, `tools_lingo.py`
  if they expose model-facing tools)

**No-bloat budget (apply to every tool):** description ≤ ~350 chars, leads with
*what it does + when to reach for it*, at most one worked cue; every non-injected
param gets a one-line doc; constrained-choice params use `Literal[...]`; a
structured `dict` param either gets a typed model or an explicit pointer to its
schema-discovery tool (e.g. `peacock_spec`).

- [ ] **Step 1:** For each server, enumerate `@mcp.tool()` functions and grade each
  (thin description? undocumented params? untyped dict? missing enum?). Record the
  grades inline in the commit body.

- [ ] **Step 2:** Rewrite docstrings + annotations per the budget. Add `Literal`
  enums where a param has a fixed value set. Do **not** change tool behavior or
  signatures' arity — only annotations/docstrings (and `str`→`Literal[...]` where
  safe).

- [ ] **Step 3:** Run each repo's tests — `cd magpie && uv run pytest`, likewise
  peacock, superbot. Expected: PASS (docstring/annotation-only changes).

- [ ] **Step 4: Commit each repo** —
  `cd magpie && git add -A && git commit -m "docs(mcp): tighten tool definitions for small-model precision"`
  (repeat for peacock, superbot).

---

## Task 8: Live smoke — qwen3.5-9b via OpenRouter

**Files:**
- Create: `lovelaice/scripts/smoke_toolcalls.py` (scratch harness; not CI)

**Interfaces:**
- Consumes: real `lovelaice.agent.Agent`, model `qwen/qwen3.5-9b`, base_url
  `https://openrouter.ai/api/v1`, key read from
  `/home/apiad/Workspace/.claude/openrouter.token` (read-only).

- [ ] **Step 1:** Write `smoke_toolcalls.py` — builds an `Agent` with the built-in
  tools (files/search) over a temp dir, `repair_tool_calls=True`, and a fixture of
  ~8 prompts each demanding a specific call (some optional-arg, some enum, some
  value-from-context). Run each twice: once against current-`main` config
  (`repair_tool_calls=False`, and a flag to force flat schemas for baseline) and
  once with the full stack. Count first-try validation, repair success, e2e
  success by subscribing to `ToolExecutionStart`/`ToolExecutionEnd`/`ToolCallRepaired`.

- [ ] **Step 2:** Run — `OPENROUTER_API_KEY=$(cat .claude/openrouter.token) uv run
  python scripts/smoke_toolcalls.py`. Expected: first-try validation materially up
  vs baseline; repair heals the clear majority of residual failures.

- [ ] **Step 3:** Record the numbers in the design doc's validation section and in
  the agent journal. No commit of the token or results secrets.

---

## Self-Review

- **Spec coverage:** WS0 bridge = Task 4; WS0 audit = Task 7; WS1 schema = Tasks
  1–3; WS2 repair = Tasks 5–6; final smoke = Task 8. All spec sections covered.
- **Type consistency:** `Tool.json_schema` produced in Task 3, consumed in Tasks 2
  (builder branch) and 4 (MCP). `update_tool_call_args` produced in Task 5,
  consumed in Task 6. `repair_tool_calls`/`repair_context` produced in Task 5,
  consumed in Task 6. `ToolCallRepaired` produced in Task 5, consumed in Task 6.
- **Placeholder scan:** `_parse_args_docstring` body is described precisely (Google
  `Args:` → `{name: desc}`, tolerant); `_repair_args` prompt composition is
  specified by inputs (schema + failed args + error + grounding). No TBD/TODO.
- **Ordering note:** Task 2's `tool_to_openai_schema` reads `tool_obj.json_schema`
  before Task 3 sets it on `Tool.__init__` — guarded by `getattr(..., None)`, so
  Task 2 is green on its own and Task 3 only wires the attribute.
