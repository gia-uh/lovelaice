import pytest

from lovelaice.acp.server import AcpServer


class _Tools:
    @staticmethod
    def all():
        return []


class _Harness:
    tools = _Tools()


class _FakeAgent:
    """Minimal live-agent stand-in: records prompts, exposes an empty tool
    harness, and a messages_for_llm() so _final_text can read the reply."""

    def __init__(self, reply="LIVE-REPLY"):
        self._reply = reply
        self.prompted = []
        self.harness = _Harness()

    async def prompt(self, text):
        self.prompted.append(text)
        return "end_turn"

    def messages_for_llm(self):
        class _M:
            role = "assistant"
            content = self._reply

        return [_M()]


@pytest.mark.asyncio
async def test_workflow_prompt_node_runs_on_the_live_session_agent():
    live = _FakeAgent("LIVE-REPLY")
    fresh = _FakeAgent("FRESH")
    server = AcpServer(agent_factory=lambda **k: fresh)
    server._sessions["sid-1"] = live

    out = await server._handle_workflow_run(
        {
            "sessionId": "sid-1",
            "spec": {
                "name": "wf",
                "root": {
                    "kind": "sequence",
                    "children": [
                        {"kind": "prompt", "prompt": "critique this", "name": "c"},
                    ],
                },
            },
            "inputs": None,
        }
    )
    assert live.prompted == ["critique this"]      # ran on the LIVE agent
    assert fresh.prompted == []                     # not the fresh factory agent
    assert out["result"]["text"] == "LIVE-REPLY"


@pytest.mark.asyncio
async def test_workflow_prompt_node_without_live_session_raises():
    fresh = _FakeAgent("FRESH")
    server = AcpServer(agent_factory=lambda **k: fresh)
    # No session registered → no live agent → prompt node must raise.
    with pytest.raises(Exception):
        await server._handle_workflow_run(
            {
                "sessionId": "missing",
                "spec": {"name": "wf", "root": {"kind": "prompt", "prompt": "x"}},
            }
        )
