# Lovelaice

A sovereign, local-first coding agent for the terminal. A native
tool-calling ReAct loop (read/bash/write/edit/glob/list_dir), yolo by default.
Three ways to run it:

- **One-shot** — streams to stdout and exits: `lovelaice "<prompt>"` (piped
  stdin works too).
- **ACP server** — `lovelaice-acp` speaks the official
  [Agent Client Protocol](https://agentclientprotocol.com) v1 over stdio, so any
  ACP client (e.g. [aegis](https://github.com/syalia-srl/aegis), Zed) can drive
  lovelaice as a native, harness-free agent — local or direct-API models, no
  external CLI. Supports streaming, per-session MCP tools, token usage, and
  `load_session` resume. See `know-how/acp-v1-server.md`.
- **Library** — `lovelaice.agent.Agent` / `lovelaice.coding.create_coding_agent`
  embed the engine directly.

## Install

```bash
pipx install lovelaice
```

## Configure

`lovelaice --init` writes a `.lovelaice.py` in the current directory.
This file *grounds the workspace*: when you run `lovelaice` from any
subdirectory, it walks up to the nearest `.lovelaice.py` and `chdir`s
to its directory before running. Exactly one config grounds the
workspace — there is no stacking.

Set `OPENROUTER_API_KEY` in your environment before running.

```bash
export OPENROUTER_API_KEY=sk-or-...
lovelaice
```

## Commands and tools

The `.lovelaice.py` registers tools and commands as decorators on a
`Config` object. Built-in tools: `bash`, `read`, `write`, `edit`,
`list`, `glob`, `grep`, `fetch`. Add your own with `@config.tool`.
See `know-how/writing-a-tool.md` and `know-how/writing-a-command.md`.

## Thinking mode

Add `thinking="high"` (or `"low"`/`"medium"`, or an integer token
budget) on a model entry to enable OpenRouter's reasoning passthrough.
Reasoning chunks render in a separate dim-italic panel above the
agent's reply. Non-OpenRouter base URLs silently ignore the knob —
v1 does not translate reasoning protocols across providers.

## MCP

Pass `mcp=[...]` to `Config(...)` to spawn stdio MCP servers and
register their tools. Tool names are prefixed `mcp:<server>:<tool>`.

```python
config = Config(
    models=MODELS,
    prompt=PROMPT,
    mcp=[
        {"name": "filesystem", "command": "npx",
         "args": ["@modelcontextprotocol/server-filesystem", "."]},
    ],
)
```

## Context compaction for embedding hosts

Set `AgentConfig.context_window` for proactive compaction. A
`lovelaice.agent.CompactionPolicy` supplied as `AgentConfig.compaction_policy`
sets thresholds and the ordered operations:

```python
from lovelaice.agent import AgentConfig, CompactionPolicy

config = AgentConfig(
    model="my-model",
    context_window=8192,
    compaction_policy=CompactionPolicy.model_validate({
        "trigger_percent": 75,
        "target_percent": 50,
        "protected_turns": 2,
        "steps": ["finished_results", "summarize_old_turns", "read_results"],
    }),
)
```

These defaults measure the space left for conversation after reserving room
for the output, system instructions, and tool schemas. Add
`finished_tool_batches` to replace complete closed-turn tool exchanges or
`truncate_old_text` to clip old dialogue when there is still an overflow.
Only listed operations run, in their listed order. The current user message
and the unread tool batch are protected. Conversation storage retains the
original messages. Reopening under a different policy rebuilds their
projection. Existing fractional `AgentConfig.compaction_trigger` and
`compaction_target` still apply when no explicit policy is supplied.

On the legacy ACP session stream, `context` adds a `compaction_detail` event
with an ordered list of `operations` (each has `step` and `count`), estimated
`before_tokens` and `after_tokens`, `estimated`, `window`, `usable`, and
`fixed_tokens`. Counts refer to results, turns, batches, or messages as
appropriate to the step. The additive `overflow_detail` event reports the
limit and size when available, `estimated`, and the ordered `attempted`
step names. Keep reading `measured` for the provider-reported prompt count.

## License

MIT.
