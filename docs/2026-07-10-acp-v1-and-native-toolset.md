# ACP v1 server + native toolset + MCP wiring (companion note)

**Date:** 2026-07-10
**Canonical spec:** `repos/aegis/docs/superpowers/specs/2026-07-10-lovelaice-native-acp-agent-design.md`

This note is the lovelaice-side pointer to a cross-repo design owned by the aegis
spec above. Read that first. Summary of what lands in **this** repo (target
release **2.7.0**, additive):

1. **New `lovelaice.acp.v1` server** on the official `agent-client-protocol` SDK
   (`acp.Agent` + `acp.run_agent`): `initialize` / `new_session` / `prompt` /
   `cancel` / `load_session`, SDK-builder event translation (nested wire shape),
   `usage` + `stopReason` in `PromptResponse`. Lovelaice-specific
   `workflow/run` + `conversation/archive` preserved as ACP **ext-methods**;
   constructor mirrors the legacy `AcpServer(agent_factory=…, conversation_store=…)`.
2. **First-class per-session MCP attach:** the v1 server reads ACP
   `new_session.mcp_servers`, connects each (HTTP + stdio) on a managed
   background session, wraps tools via `lovelaice.mcp._wrap_mcp_tool`, and tears
   them down on session close. The HTTP-on-a-thread lifecycle is upstreamed into
   `lovelaice.mcp` (warden's `_start_http_mcp` is the reference impl).
3. **Full native toolset** in the default `lovelaice-acp` factory
   (`create_coding_agent`): `read`, `write`, `edit`, `glob`, `list`, `bash`,
   reusing `tools/files` + `tools/search`, with `path_guard` / `bash_prefix_guard`
   and ACP `kind`s. No subagents, no skills.
4. **`lovelaice-acp` script repointed to v1.** The legacy
   `lovelaice.acp.server.AcpServer` ("0.1" flat dialect) is **frozen and
   preserved** — warden spawns the class directly, and its `>=2.6.0,<3.0` pin is
   satisfied by 2.7.0 with zero changes.

**warden upgrade checklist** (opt-in, not part of this project) will live at
`repos/lovelaice/know-how/` once the v1 server lands — the bounded delta to move
warden's client + `_acp_driver` onto v1 and drop its duplicated HTTP-MCP bridge.
