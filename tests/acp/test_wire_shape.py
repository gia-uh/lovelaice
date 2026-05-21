import asyncio
import json
import os
import sys
import pytest


@pytest.mark.asyncio
async def test_acp_wire_shape_subprocess(tmp_path):
    """Canary #4 — spawn `python -m lovelaice.acp` with a fake LLM,
    drive initialize → session/new → session/prompt; assert event shape."""
    env = os.environ.copy()
    env["LOVELAICE_FAKE_LLM"] = "1"
    env["LOVELAICE_SESSION_PATH"] = str(tmp_path / "s.jsonl")

    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "lovelaice.acp",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )

    async def send(obj):
        proc.stdin.write((json.dumps(obj) + "\n").encode())
        await proc.stdin.drain()

    async def recv_id(expected_id, timeout=10.0):
        """Read lines until we see a response with the given id.
        Returns (response_obj, list_of_notifications_seen)."""
        notifications = []
        async def _read():
            while True:
                line = await proc.stdout.readline()
                if not line:
                    raise RuntimeError("server closed stdout")
                obj = json.loads(line)
                if obj.get("id") == expected_id:
                    return obj, notifications
                if "method" in obj:
                    notifications.append(obj)
        return await asyncio.wait_for(_read(), timeout=timeout)

    try:
        # 1) initialize
        await send({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        resp, _ = await recv_id(1)
        assert resp["result"]["protocolVersion"] == "0.1"

        # 2) session/new
        await send({"jsonrpc": "2.0", "id": 2, "method": "session/new",
                    "params": {"cwd": str(tmp_path)}})
        resp, _ = await recv_id(2)
        sid = resp["result"]["sessionId"]

        # 3) session/prompt
        await send({"jsonrpc": "2.0", "id": 3, "method": "session/prompt",
                    "params": {"sessionId": sid,
                               "prompt": [{"type": "text", "text": "hi"}]}})
        resp, notifications = await recv_id(3)
        assert resp["result"]["stopReason"] == "end_turn"

        # At least one session/update notification fired with agent_message_chunk.
        kinds = [n.get("params", {}).get("sessionUpdate")
                 for n in notifications if n.get("method") == "session/update"]
        assert "agent_message_chunk" in kinds, f"got kinds: {kinds}"
    finally:
        proc.stdin.close()
        try:
            await asyncio.wait_for(proc.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
