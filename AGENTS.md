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
- `src/lovelaice/acp/` — ACP surface. `server.py` (`AcpServer`), `client.py`
  (`InProcessAcpClient`), `protocol.py`; `__main__.py` is the `lovelaice-acp`
  stdio server. ACP translates `AgentEvent`s to `session/update` notifications.
- `src/lovelaice/coding/` — the coding **host**. `host.py`'s
  `create_coding_agent(model, session_path, cwd)` wires `coding/tools/`
  (`read`, `bash`) + `coding/hooks.py` guards onto a `ReActNative` agent. This is
  the host `cli.py` uses.
- `src/lovelaice/workflows/` — the native workflow engine (agent/tool/sequence
  nodes; `workflow` decorator). `executor.py` runs a spec against a host.
- `src/lovelaice/tools/` — standalone utility tools (`bash`, `files`, `search`,
  `web.fetch`). Not wired into the coding host (which has its own `coding/tools`);
  kept as a reusable library (external consumers import `lovelaice.tools.web.fetch`).
- `src/lovelaice/mcp.py` — wrap stdio/HTTP MCP servers as `lingo.Tool`s. A
  **capability, currently unwired** to the agent path (its old consumer was
  `Config.build`). Rewiring MCP into the coding host / ACP is a future task.

## Running tests

```bash
uv run pytest
```

## Know-how

Specific procedure docs in `know-how/`. Match the task; load the matching doc.

- **writing-a-tool** — adding a new tool (a `lingo.Tool` wrapped as an `AgentTool`
  in a host, e.g. `coding/host.py`).

## Manual smoke checklist

Before tagging a release, with `OPENROUTER_API_KEY` set:

- `lovelaice "list the files in this repo" --model <slug>` → a real tool call
  fires (dimmed `→ tool(...)` line) and the reply streams in a Rich panel.
- `echo "say hi" | lovelaice` → reads the piped prompt, prints the reply.
- `lovelaice` (tty, no arg) → prints usage and exits non-zero.
- `lovelaice-acp` → starts the stdio ACP server (smoke via an ACP client).
