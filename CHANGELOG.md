## 2.10.0 — 2026-07-13

### Added

- **Token usage in the ACP v1 `PromptResponse`.** `AcpServerV1` accumulates
  prompt/completion/total tokens from each finalized assistant message across a
  turn and returns them as an ACP `Usage`, so ACP clients (aegis) show real
  token metrics instead of 0/0. Server-side only — no engine change.

## 2.9.0 — 2026-07-13

### Added

- **Full native coding toolset.** The default `lovelaice-acp` coding agent now
  wires `write`, `edit`, `glob`, and `list_dir` alongside the existing `read`
  and `bash` — on par with the basic toolset of other harnesses. New coding-host
  tool wrappers (`coding/tools/{write,edit,glob,list_dir}.py`) delegate to the
  already-robust `lovelaice.tools.files`/`search` logic (parent-dir creation,
  unique-match edit, `.gitignore`-aware glob). The cwd `path_guard` now also
  covers `list_dir`. No subagents, no skills.

## 2.8.0 — 2026-07-12

### Added

- **Per-session MCP attach in the ACP v1 server.** `AcpServerV1.new_session`
  reads ACP `mcp_servers` (HTTP + stdio), connects each, wraps their tools, and
  builds the session's agent **with** those tools (so the system prompt
  advertises them). Managed sessions are torn down on `close_session`. This lets
  an ACP client (e.g. aegis) give a native lovelaice agent access to its own MCP
  plane per session.
- **`lovelaice.mcp.ManagedMcpSession` + `start_managed_session` +
  `build_agent_tools`** — a first-class background MCP session supporting **HTTP
  and stdio** with explicit teardown (`aclose`), generalizing the previous
  stdio-only helper.
- **`create_coding_agent(extra_tools=…)`** — inject extra tools (e.g. per-session
  MCP tools) at agent construction.

### Fixed

- MCP tool display names are sanitized to the LLM tool-name pattern
  (`^[a-zA-Z0-9_-]{1,128}$`) — the old `mcp:<server>:<tool>` form contains colons
  that Anthropic/OpenAI reject, 400-ing the whole request.

## 2.7.0 — 2026-07-10

### Added

- **ACP v1 server (`lovelaice.acp.v1`).** A clean-room ACP server built on the
  official `agent-client-protocol` SDK (`acp.Agent` + `acp.run_agent`):
  `initialize` (protocol version 1), `new_session`, `prompt`, `cancel`, and
  agent-event → `session/update` translation via the SDK's builder helpers
  (correct nested wire shape). Returns `stopReason` from the run. The
  `lovelaice-acp` console script now runs this v1 server. Enables official ACP
  clients (aegis, Zed) to drive lovelaice as a native, harness-free agent over
  local or direct-API models.

### Changed

- `lovelaice-acp` entrypoint repointed from the legacy hand-rolled server to the
  new v1 server. The legacy `lovelaice.acp.server.AcpServer` (0.1 flat dialect)
  is **preserved unchanged** for existing clients (warden) — importable as a
  class; only the CLI default moved.

### Fixed

- ACP prompt text is now extracted from typed `TextContentBlock` objects (as the
  SDK delivers over the wire), not just dicts — the dict-only path produced an
  empty prompt so the agent ignored the user message.

## 2.6.0 — 2026-07-03

### Added

- **Workflow fan-out: `parallel` and `map` nodes.** `ParallelNode`
  (`kind: "parallel"`) runs its children concurrently; `MapNode`
  (`kind: "map"`, with `over` naming a list var and `as` the loop var) fans one
  child template over that list. Each branch runs in an **isolated copy of vars**
  (siblings never race); the aggregate list of results is collected into the
  node's optional `name`. Concurrency is bounded at **4**, enforced at the leaves
  (`agent`/`tool`/`prompt` calls acquire a shared semaphore) so nested fan-out
  can't deadlock. A `prompt` node inside a fan-out raises — it would race the
  single live host agent; use an `agent` node in fan-outs.

## 2.5.0 — 2026-07-03

### Added

- **Workflow `prompt` node.** A new `PromptNode` (`kind: "prompt"`) runs a prompt
  against the host's *live/primary* agent (shared context) instead of a fresh
  isolated one, via a new `prompt_handler` seam on `workflows.executor.run(...)`.
  Same shape as `AgentNode`; the executor routes it to the host-supplied handler
  and raises if none is provided (a `prompt` node has no meaning headless). The
  ACP server's `workflow/run` binds `prompt_handler` to the session's live agent
  (`self._sessions[sessionId]`), so a `prompt` node streams its chunks as
  `session/update` notifications with no extra plumbing. Enables superbot
  workflows to drive the ongoing conversation, not only fan out to sub-agents.

## 2.4.0 — 2026-07-03

### Removed

- **Retired the old Lingo-based agent path.** The structured-output
  decide/equip/invoke loop (`commands/react.py`) was unreliable — with some
  models `decide()` short-circuits and no tool ever runs. Deleted `core.py`
  (`Lovelaice`), `config.py` (`Config`/`.lovelaice.py` build path),
  `commands/`, `oneshot.py`, `template.py`, `thinking.py`, and the Textual
  `tui/` (dropped the `textual` dependency). The CLI already ran on the native
  engine; these modules were dead weight.

### Changed

- `import lovelaice` now exposes the native engine: `Agent`, `AgentConfig`,
  `workflow`. The single agentic engine is the `agent/` package
  (`Agent` + `ReActNative`, native `Message.tool_calls`), driven by `cli.py`
  via an in-process ACP client and the coding host.
- `mcp.py` is retained as an (currently unwired) capability; rewiring MCP into
  the agent path is a follow-up.

## 2.1.2 — 2026-06-30

### Fixed

- Agent loop (`ReActNative`) no longer ends the turn on an empty assistant
  message (no tool calls and no answer text). Thinking models can return
  reasoning-only with empty content on the continuation after a tool result;
  the turn now ends only on a genuine final answer or tool soft-termination,
  nudging and continuing on an empty turn (bounded by `MAX_EMPTY_CONTINUATIONS`).

## 2.0.3 — 2026-05-21

### Added

- HTTP MCP transport in `lovelaice.mcp`. The same `mcpServers` config
  shape now dispatches on `url` (HTTP) vs `command`+`args` (stdio).
  `{"name": "...", "url": "http://...", "auth": {"bearer": "..."}}`
  connects to an HTTP MCP server via the `mcp` SDK's `streamable_http`
  client. Existing stdio path unchanged.
- Release CI workflow (`.github/workflows/release.yaml`) that publishes
  to PyPI on every GitHub Release using OIDC trusted publishing.

## 2.0.0 — 2026-05-21

### Breaking

- Rebuilt around a new generic agent engine (`lovelaice.agent`).
  The coding-agent behavior is now one host module
  (`lovelaice.coding`) on top of it. Callers of the legacy
  `lovelaice.core` module need to migrate to the new `Agent` API.

### Added

- `lovelaice.agent` — host-agnostic agentic engine: turn loop, tool
  dispatch, hooks, append-only JSONL sessions. See
  `vault/Atlas/Architecture/2026-05-20-lovelaice-agentic-engine-design.md`.
- `lovelaice.acp` — native ACP (Agent Client Protocol) support:
  `lovelaice-acp` stdio server + in-process client. JSON-RPC 2.0 over
  stdio per https://agentclientprotocol.com.
- `lovelaice.coding` — first host module: `read` (kind=read) + `bash`
  (kind=execute) tools + path/command-prefix guard hooks.
- New CLI: `lovelaice "prompt"` routes through the in-process ACP
  client. Existing one-shot UX preserved.

### Changed

- `lingo-ai` pin bumped to `>=2.0.0,<3.0`.
- Standalone CLI and (future) external ACP clients (aegis, Zed) drive
  the same engine — one event taxonomy, two consumers.

### Deferred (VS2+)

- `session/request_permission`, `session/load`, full hook set,
  `Direct` loop, remaining coding tools (write/edit/list/glob/grep/fetch),
  TUI cutover, MCP-client wired through `session/new.mcpServers`,
  caching knobs (lingo §3.4), image/audio/embedded content blocks.
  Legacy `lovelaice.core` and 5 legacy CLI test files removed in VS2.
