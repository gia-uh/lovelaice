# AGENTS.md — lovelaice

A sovereign, local-first terminal **coding agent**. The engine is a **native
tool-calling ReAct loop** (`Message.tool_calls` on the wire) — one assistant
message → N tool calls → tool results → loop until the model gives a final
answer. Read this before touching the repo.

## Architecture (the native engine)

Everything runs through the `agent/` package. The old Lingo-based agent
(`core.Lovelaice`, `commands/react.py`'s structured-output decide/equip/invoke,
`config.Config`, the Textual `tui/`, `oneshot.py`, `template.py`, `thinking.py`)
was **retired 2026-07-03** — do not look for it.

- `src/lovelaice/cli.py` — typer entrypoint. `lovelaice "prompt"` runs one prompt
  through the engine via an **in-process ACP client** (`acp.client` →
  `acp.server` → coding host). Piped stdin works. Model via `--model` /
  `LOVELAICE_MODEL`. No-arg (tty) prints usage.
- `src/lovelaice/agent/` — the engine.
  - `agent.py` — `Agent` + `AgentConfig` (the user-facing API: `agent.prompt(text)
    -> StopReason`). `AgentConfig` carries model/api_key/base_url/max_tokens and
    the opt-in tool-arg repair knobs.
  - `loops/react_native.py` — `ReActNative`, the canonical loop.
  - `harness.py` — LLM-call seam (`llm_call` → `lingo.LLM.chat(tools=…)`), tool
    dispatch (`execute_tools_batch`), and the `AgentEvent` bus.
  - `tools.py` — `AgentTool` (wraps a `lingo.Tool` + kind/sequential/title) +
    `ToolRegistry` + arg validation.
  - `session.py` / `conversation.py` — persistence; `events.py` — typed events
    (`TurnStart`, `AssistantMessageFinalized`, `ToolExecutionStart/End`, …);
    `hooks.py` — `before_llm_call` / `tool_call` reducer chain; `prompt.py` —
    system-prompt assembly.
- `src/lovelaice/acp/` — ACP surface. **`v1/`** is the current server
  (`v1/server.py` `AcpServerV1` + `v1/__main__.py`), built on the official
  `agent-client-protocol` SDK (`acp.Agent` + `acp.run_agent`); the `lovelaice-acp`
  console script runs it. It speaks **official ACP v1** and supports per-session
  MCP attach, streaming (`AssistantMessageDelta` → `AgentMessageChunk`), token
  `usage` in `PromptResponse`, and `load_session` resume (deterministic
  `~/.lovelaice/acp-sessions/<sid>.jsonl`, `LOVELAICE_SESSIONS_DIR`-overridable).
  The legacy `server.py` (`AcpServer`) / `client.py` (`InProcessAcpClient`) /
  `protocol.py` implement an older hand-rolled "0.1" flat dialect and are
  **frozen** — warden spawns the `AcpServer` class directly and pins
  `lovelaice>=2.6,<3`; do not edit them. See `know-how/acp-v1-server.md`.
- `src/lovelaice/coding/` — the coding **host**. `host.py`'s
  `create_coding_agent(model, session_path, cwd, base_url, api_key, extra_tools)`
  wires the full `coding/tools/` set (`read`, `bash`, `write`, `edit`, `glob`,
  `list_dir`) + `coding/hooks.py` guards (`path_guard` covers read/write/edit/
  list_dir; `bash_prefix_guard`) onto a `ReActNative` agent. `extra_tools` injects
  per-session MCP tools at construction so the system prompt advertises them.
  Used by `cli.py` and by the ACP v1 default factory.
- `src/lovelaice/workflows/` — the native workflow engine (agent/tool/sequence
  nodes; `workflow` decorator). `executor.py` runs a spec against a host.
- `src/lovelaice/tools/` — standalone utility tools (`bash`, `files`, `search`,
  `web.fetch`). Not wired into the coding host (which has its own `coding/tools`);
  kept as a reusable library (external consumers import `lovelaice.tools.web.fetch`).
- `src/lovelaice/mcp.py` — wrap stdio/HTTP MCP servers as `lingo.Tool`s.
  `ManagedMcpSession` + `start_managed_session` keep a session alive on a
  background loop (HTTP **and** stdio) with `aclose()` teardown;
  `build_agent_tools(specs) -> (tools, sessions)` wraps them as `AgentTool`s.
  Tool display names are sanitized to the LLM name pattern
  (`mcp_<server>_<tool>` — colons in the old form 400'd Anthropic/OpenAI).
  Wired into the ACP v1 server per session (`new_session.mcp_servers`).

## Running tests

```bash
uv run pytest
```

## Know-how

Specific procedure docs in `know-how/`. Match the task; load the matching doc.

- **writing-a-tool** — adding a new tool (a `lingo.Tool` wrapped as an `AgentTool`
  in a host, e.g. `coding/host.py`).
- **acp-v1-server** — the ACP v1 server (`acp/v1/`): protocol, legacy-vs-v1
  dialect, warden compatibility, per-session MCP attach, streaming, `load_session`
  resume, token usage. Reach for it when touching `acp/` or embedding lovelaice
  as an ACP agent (aegis, Zed).

## Manual smoke checklist

Before tagging a release, with `OPENROUTER_API_KEY` set:

- `lovelaice "list the files in this repo" --model <slug>` → a real tool call
  fires (dimmed `→ tool(...)` line) and the reply streams in a Rich panel.
- `echo "say hi" | lovelaice` → reads the piped prompt, prints the reply.
- `lovelaice` (tty, no arg) → prints usage and exits non-zero.
- `lovelaice-acp` → starts the stdio **ACP v1** server. Smoke with an official
  `agent-client-protocol` client (initialize → new_session → prompt); set
  `LOVELAICE_FAKE_LLM=1` for a hermetic, keyless handshake.
