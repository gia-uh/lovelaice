# Retire the old Lingo agent path — Design

**Status:** design.
**Date:** 2026-07-03.

> lovelaice carries two agentic engines. The **native** `agent/` package
> (`Agent` + `ReActNative`, real `Message.tool_calls`) is where all recent work
> lives (2.2.0 workflows, 2.3.0 precise tool-calling + repair) and is what ACP,
> the coding host, and workflows already use. The **old** Lingo path
> (`core.Lovelaice`, `commands/react.py`'s structured-output decide/equip/invoke,
> and `Config.build()`) survives only because the interactive CLI/TUI never
> migrated. The old ReAct loop is unreliable — with Claude Haiku via OpenRouter
> its `decide()` short-circuits to "done" on step 1, so no tool ever runs and the
> model emits its native tool XML as *text*. This slice retires the old path.

## 1. Goal

The CLI one-shot runs on the native `Agent`; the TUI is deleted; the old Lingo
agent path is removed. `.lovelaice.py` keeps its shape — only what `build()`
produces changes (a native `Agent`, not a `Lovelaice`).

## 2. Scope

**In:**
- **Config bridge:** `Config.build()` returns a native `Agent` (ReActNative).
- **Streaming parity:** `AgentConfig` gains optional `on_token`,
  `on_reasoning_token`, and `reasoning` passthrough (threaded into `_build_llm`),
  so the CLI keeps token-level streaming and the `thinking=` knob.
- **one-shot rewire:** `oneshot.py`'s three output modes (rich / plain / json)
  are re-driven by streaming callbacks + the agent event bus.
- **Delete the TUI:** remove `tui/` entirely (not a goal — per owner).
- **Delete the old path:** `core.py` (`Lovelaice`), `commands/react.py`, the
  `Config.build()→Lovelaice` construction, stale `__init__` exports, and the
  `config.command(react)` wiring in `template.py`.
- **Refresh `AGENTS.md`** (currently describes the retired core/react).

**Out (YAGNI / separate concerns):**
- Simplifying the `Config` / `.lovelaice.py` model beyond what the bridge needs.
- ACP, coding host, workflows — already native, untouched.
- The `nell.py` LETO demo — already uses `agent.Agent` directly.

## 3. Config bridge (`config.py`)

`.lovelaice.py` keeps declaring `MODELS`, `PROMPT`, built-in + custom tools, and
`mcp`. `Config.build(model, on_token=None, on_reasoning_token=None)` changes what
it returns:

1. Resolve `model_kwargs` from `self.models[model]`; pop `thinking`.
2. Build an `AgentConfig`:
   - `model=model_kwargs["model"]`, `api_key`, `base_url` from `model_kwargs`.
   - `system_prompt=self.prompt`, `cwd=os.getcwd()`.
   - `max_tokens` from `model_kwargs` if present.
   - `on_token`, `on_reasoning_token` passed through.
   - `reasoning` derived from `thinking` **only when** `base_url` is OpenRouter
     (reuse the existing `thinking._resolve_reasoning_kwargs` translation).
3. Wrap tools as `AgentTool`s (the `coding/host.py` pattern). Built-in `tools/`
   functions are plain callables → `AgentTool(inner=lingo.tool(fn), kind=…)`:

   | tool | kind | sequential |
   |------|------|------------|
   | `bash` | `execute` | yes |
   | `read` | `read` | no |
   | `write` | `edit` | no |
   | `edit` | `edit` | no |
   | `list` (`list_`) | `search` | no |
   | `glob` | `search` | no |
   | `grep` | `search` | no |
   | `fetch` | `fetch` | no |

   Custom tools registered via `config.tool(fn, name=…)` map to
   `AgentTool(inner=lingo.tool(fn), kind="other")` with the name override applied
   to `inner._name`.
4. MCP tools: wrap the `mcp:<server>:<tool>` lingo tools produced by
   `mcp.register_mcp_tools` as `AgentTool(inner=…, kind="other")`. (Reconcile:
   `register_mcp_tools` currently attaches to a `Lovelaice`; refactor it to return
   a list of lingo tools the bridge wraps.)
5. `session_path`: one-shot is stateless → an ephemeral temp path (created per
   process, discarded). Drop the `config.command(react)` step entirely.

`Config.build()` returns the `Agent`. The `agent` attribute + the
"already called once" guard stay.

## 4. Streaming parity (`agent/agent.py`)

`AgentConfig` gains three optional fields (all default `None`, backward-compatible
— ACP/coding/workflows never set them):

```python
on_token: Callable[[str], Any] | None = None
on_reasoning_token: Callable[[str], Any] | None = None
reasoning: dict | None = None
```

`_build_llm(cfg)` threads them into `lingo.LLM(...)` (which already supports
`on_token`, `on_reasoning_token`, and `reasoning`). No other agent-package change.

## 5. one-shot rewire (`oneshot.py`)

`run_oneshot(config_path, *, model, prompt, verbose, output)` keeps its signature.
Each mode builds a native `Agent` via the bridge (with the right token callbacks),
subscribes to the event bus for tool + turn signals, and runs
`await agent.prompt(prompt)` (returns a `StopReason`).

- **json (NDJSON):** unchanged event vocabulary plus tool visibility —
  `on_token → {"type":"content","delta":…}`, `on_reasoning_token →
  {"type":"reasoning","delta":…}`; subscribe `ToolExecutionStart →
  {"type":"tool","name":…,"args":…}` and `ToolExecutionEnd →
  {"type":"tool_result","is_error":…}`; on completion emit
  `{"type":"done","content":<final assistant text>}`; on exception
  `{"type":"error","stage":…,"message":…}`.
- **plain:** content tokens → stdout, reasoning tokens → stderr (as today);
  tool activity is not printed (keeps pipes clean).
- **rich:** Live panels fed by the token buffers; a compact tool-activity line per
  `ToolExecutionStart` (name + rendered `title_for(args)`). Quiet-pipe fallback
  prints only the final assistant text.
- **Exit code:** map `StopReason` → process code (END_TURN/​MAX_TOKENS → 0;
  REFUSAL/​CANCELLED and build/chat exceptions → non-zero), preserving the current
  0/2 contract.

The final assistant text is read from the agent's message list after
`prompt()` (last `assistant` message with non-empty content) — mirrors `nell.py`.

## 6. `cli.py` after the TUI

`cli.py` currently dispatches: prompt given → one-shot; no prompt → TUI. With the
TUI gone, no-argument invocation prints usage/help and exits non-zero (a prompt is
now required). All one-shot flags (`--plain`, `--json`, `--verbose`, `--model`)
are unchanged.

## 7. Deletions & housekeeping

- Delete `src/lovelaice/tui/` (whole package) and its `cli.py` dispatch + imports.
- Delete `src/lovelaice/core.py` (`Lovelaice`) and `src/lovelaice/commands/`
  (`react.py`, `__init__.py`).
- `__init__.py`: drop the `Lovelaice` export; export `Agent`, `AgentConfig` from
  `agent/` as the package's public agent API.
- `template.py`: remove the `from lovelaice.commands import react` +
  `config.command(react)` block; keep the `MODELS`/`PROMPT`/tool registrations.
- Refresh `AGENTS.md`: describe the native `agent/` engine, the one-shot CLI, and
  drop the TUI + core/react sections.
- Remove now-dead deps if any (e.g. Textual) from `pyproject.toml`.

## 8. Testing

Deterministic, no real LLM (use `lingo.mock.MockLLM`, per lovelaice's existing
smoke tests):

- **Config bridge:** `Config.build()` returns an `Agent`; its `ToolRegistry`
  contains the expected tool names with the mapped `kind`/`sequential`; the
  `thinking=` knob produces the right `reasoning` only for OpenRouter base URLs.
- **one-shot json mode:** with a `MockLLM` scripted to emit content (and a
  tool call → tool result → final content), assert the NDJSON event sequence
  (`content`/`reasoning`/`tool`/`tool_result`/`done`) and exit code.
- **one-shot plain mode:** content on stdout, reasoning on stderr.
- **Deletion safety net:** grep the tree for `core.Lovelaice`, `commands.react`,
  `config.command(` and `tui` imports → none remain; `import lovelaice` succeeds
  and exposes `Agent`.
- **Manual smoke:** `lovelaice "list the files in this repo" --model <slug>` (real
  OpenRouter key) → a real tool call fires and the reply renders in rich; repeat
  with `--json` and `--plain`; no-arg → usage + non-zero exit.

## 9. Open questions

- **`commands/` fate:** with `react.py` gone the package is empty; delete the
  directory (custom "commands" as a concept are superseded by native tools /
  workflows). Confirm no external `.lovelaice.py` relies on `config.command`.
- **`Config.command` API:** keep the method as a no-op/deprecation shim for one
  release, or remove outright? Default: remove (pre-1.0 latitude); revisit if it
  breaks a known consumer.
