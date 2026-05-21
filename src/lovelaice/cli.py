"""Lovelaice CLI entrypoint.

Standalone: `lovelaice "prompt"` runs one prompt through the new engine
via an in-process ACP client. ACP server mode is invoked via
`lovelaice-acp` (see lovelaice.acp.__main__).

Host module selection: VS1 only supports the coding host. Future hosts
will be selectable via `--host` or env var.
"""
import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from lovelaice.acp.client import InProcessAcpClient
from lovelaice.acp.server import AcpServer
from lovelaice.acp.protocol import JsonRpcNotification
from lovelaice.agent.errors import StopReason


app = typer.Typer(add_completion=False, no_args_is_help=False)
console = Console()


async def run_one_shot(
    *,
    prompt: str,
    model: str,
    session_path: Path,
    cwd: str,
) -> StopReason:
    """Run one prompt through the engine via the in-process ACP client.

    Prints the assistant text via Rich; tool calls/results render dimmed.
    """
    # Late import — coding host module may not exist at import time on
    # partial installs. This is the seam where host selection lives.
    from lovelaice.coding.host import create_coding_agent

    def agent_factory():
        return create_coding_agent(
            model=model, session_path=session_path, cwd=cwd)

    server = AcpServer(agent_factory=agent_factory)
    client = InProcessAcpClient(server)

    def on_notification(n: JsonRpcNotification):
        if n.method != "session/update":
            return
        p = n.params or {}
        kind = p.get("sessionUpdate")
        if kind == "agent_message_chunk":
            content = p.get("content", {})
            if content.get("type") == "text":
                console.print(content.get("text", ""), end="")
        elif kind == "tool_call":
            console.print(
                f"[dim]→ {p.get('title')}({p.get('rawInput')})[/dim]")
        elif (kind == "tool_call_update"
                and p.get("status") in ("completed", "failed")):
            status_color = "green" if p["status"] == "completed" else "red"
            console.print(f"[{status_color}]  ✓ {p['status']}[/]")

    client.on_notification(on_notification)

    await client.initialize()
    sid = await client.session_new(cwd)
    result = await client.session_prompt(sid, prompt)
    console.print()
    return StopReason(result["stopReason"])


@app.command()
def main(
    prompt: Optional[str] = typer.Argument(None),
    model: str = typer.Option(
        os.getenv("LOVELAICE_MODEL", "anthropic/claude-haiku-4-5"),
        "--model", "-m"),
    session_path: Optional[Path] = typer.Option(None, "--session-path"),
    cwd: Optional[str] = typer.Option(None, "--cwd"),
):
    """Lovelaice — one-shot agent CLI."""
    if not prompt:
        # Piped stdin?
        if not sys.stdin.isatty():
            prompt = sys.stdin.read().strip()
        else:
            typer.echo("Usage: lovelaice \"<prompt>\"")
            raise typer.Exit(1)

    cwd_str = cwd or os.getcwd()
    if session_path is None:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        session_path = Path.home() / ".lovelaice" / "sessions" / f"{ts}.jsonl"

    stop = asyncio.run(run_one_shot(
        prompt=prompt, model=model, session_path=session_path, cwd=cwd_str))
    raise typer.Exit(0 if stop == StopReason.END_TURN else 1)


if __name__ == "__main__":
    app()
