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
