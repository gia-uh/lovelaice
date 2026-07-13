# know-how: the ACP v1 server

**Reach for this when** you touch anything under `src/lovelaice/acp/`, embed
lovelaice as an ACP agent (aegis, Zed), or debug an ACP client talking to
`lovelaice-acp`.

## What it is

`lovelaice-acp` runs `lovelaice.acp.v1.server.AcpServerV1` over stdio via
`acp.run_agent`. It implements the official **`agent-client-protocol` (ACP) v1**
`Agent` interface, so any ACP client drives lovelaice as a native, harness-free
coding agent (direct API or local models via lingo — no external CLI harness).

`AcpServerV1(agent_factory, conversation_store=None)`. The `agent_factory`
signature is `agent_factory(*, mcp_tools=None, session_path=None)` and returns a
lovelaice `Agent`. The default factory (`v1/__main__.py:_default_factory`) builds
a `create_coding_agent` (read/bash/write/edit/glob/list_dir); hosts can supply
their own factory (e.g. warden wires MCP tools) — the constructor mirrors the
legacy `AcpServer` so a host migrates by import + dialect only.

## Legacy vs v1 — do NOT break warden

There are **two** ACP servers in this repo:

- `acp/v1/` — the current official-v1 server. `lovelaice-acp` points here.
- `acp/server.py` (`AcpServer`) + `acp/protocol.py` — a **frozen** hand-rolled
  "0.1" flat dialect (`protocolVersion "0.1"`, flat `session/update` params,
  non-ACP `workflow/run` / `conversation/archive` extensions, `conversationId`
  session load). **warden** (`repos/warden`) spawns the `AcpServer` *class*
  directly (not the script) and pins `lovelaice>=2.6,<3`.

**Keep `AcpServer` byte-compatible within the 2.x line.** Its test suite
(`tests/acp/test_server.py`, `test_workflow_*.py`, `test_wire_shape.py`) must
stay green — that is the warden compat guarantee. Only the `lovelaice-acp`
*script* moved to v1; warden is unaffected. The warden migration onto v1 (and
retirement of the legacy dialect) is deferred; until then, both coexist.

## What v1 supports

- **initialize** — advertises `PROTOCOL_VERSION` (int `1`), `load_session=True`.
- **new_session / load_session** — read ACP `mcp_servers`, build the agent on a
  **deterministic** `~/.lovelaice/acp-sessions/<session_id>.jsonl`
  (`LOVELAICE_SESSIONS_DIR`-overridable). `load_session` rebuilds on the same
  path so `Agent`'s `Session.load` restores prior context — across subprocess
  restarts.
- **prompt / cancel** — `prompt` runs one turn, returns `stop_reason` + token
  `usage` (accumulated from each `AssistantMessageFinalized.message.usage`).
  `cancel` aborts the in-flight prompt task.
- **Streaming** — `Harness` wires lingo's `on_token` to emit
  `AssistantMessageDelta`; the server streams these as incremental
  `AgentMessageChunk`s. The finalized message emits content **only as a
  fallback** when nothing streamed (`_streamed_any`), so text is never doubled.
- **Per-session MCP** — `new_session.mcp_servers` (HTTP + stdio) connect via
  `lovelaice.mcp.build_agent_tools` (run off-loop with `asyncio.to_thread` — it
  blocks on connect threads) and attach at agent construction. Torn down on
  `close_session`.

## Gotchas (each cost a real debug)

- **MCP tool names must match `^[a-zA-Z0-9_-]{1,128}$`.** The old
  `mcp:<server>:<tool>` form has colons → Anthropic/OpenAI 400 the whole
  request. `_mcp_display_name` sanitizes to `mcp_<server>_<tool>`; the original
  name is still used for the actual MCP call.
- **Don't block the event loop.** `build_agent_tools` waits on per-server connect
  threads; calling it directly in `new_session` freezes the ACP server (surfaces
  as a generic `RequestError: Internal error`). Use `asyncio.to_thread`.
- **The SDK swallows agent-side exceptions as "Internal error".** To see the
  real traceback, reproduce **in-process** (build `AcpServerV1`, call `prompt`
  directly) rather than through the `lovelaice-acp` subprocess.
- **Prompt text arrives as typed `TextContentBlock` objects** over the wire, not
  dicts — `_prompt_text` handles both. A dict-only reader gets an empty prompt.

## Verifying (real model, not just FAKE_LLM)

`LOVELAICE_FAKE_LLM=1` proves protocol wiring but **not** that input reaches the
model or that tool schemas are API-valid. Every substantive change here was
caught green by hermetic tests and then broken by a real model. Keep a
real-model probe: point an ACP client (or aegis's `LovelaiceDriver`) at a live
`anthropic/claude-haiku-4-5` via OpenRouter and assert the actual behavior
(tool called, context recalled, tokens non-zero).

Cross-repo design + slice history:
`repos/aegis/docs/superpowers/specs/2026-07-10-lovelaice-native-acp-agent-design.md`
and the `plans/2026-07-*` VS1–VS5 plans.
