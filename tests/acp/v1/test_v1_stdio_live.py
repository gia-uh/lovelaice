import asyncio
import os
import sys

import pytest
import acp


class _Client(acp.Client):
    def __init__(self):
        self.messages = []

    def on_connect(self, conn):
        return None

    async def session_update(self, session_id, update, **kw):
        if type(update).__name__ == "AgentMessageChunk":
            self.messages.append(getattr(update.content, "text", ""))

    async def request_permission(self, options, session_id, tool_call, **kw):
        return acp.RequestPermissionResponse(
            outcome={"outcome": "selected", "optionId": options[0].option_id})

    async def read_text_file(self, path, session_id, limit=None, line=None, **kw):
        return acp.ReadTextFileResponse(content="")

    async def write_text_file(self, content, path, session_id, **kw):
        return None

    async def create_terminal(self, *a, **kw): return None
    async def terminal_output(self, *a, **kw): return None
    async def wait_for_terminal_exit(self, *a, **kw): return None
    async def kill_terminal(self, *a, **kw): return None
    async def release_terminal(self, *a, **kw): return None
    async def ext_method(self, method, params): return {}
    async def ext_notification(self, method, params): return None


@pytest.mark.asyncio
async def test_v1_server_handshakes_with_official_sdk_client():
    env = dict(os.environ, LOVELAICE_FAKE_LLM="1", LOVELAICE_MODEL="fake/model")
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "lovelaice.acp.v1",
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE, env=env, limit=16 * 1024 * 1024)
    client = _Client()
    conn = acp.connect_to_agent(client, proc.stdin, proc.stdout)
    try:
        init = await conn.initialize(
            protocol_version=1,
            client_capabilities={"fs": {"readTextFile": True, "writeTextFile": True}},
            client_info={"name": "test", "version": "0"})
        assert init.protocol_version == 1
        sess = await conn.new_session(cwd=".", mcp_servers=[])
        resp = await conn.prompt(
            session_id=sess.session_id, prompt=[{"type": "text", "text": "hi"}])
        assert resp.stop_reason in ("end_turn", "cancelled")
    finally:
        proc.terminate()
        await proc.wait()
