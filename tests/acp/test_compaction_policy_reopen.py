"""The persistent session path and additive ACP detail used by Warden."""
from lingo.llm import Message, ToolCall, Usage

from lovelaice.acp.client import InProcessAcpClient
from lovelaice.acp.server import AcpServer
from lovelaice.agent import Agent, AgentConfig, CompactionPolicy
from lovelaice.agent.compaction import CompactionState, check_tool_pairs
from lovelaice.agent.loops.react_native import ReActNative

from tests.agent.test_conversation_compaction import store
from tests.agent.test_model_call import ScriptedLLM


async def test_acp_operation_report_and_policy_replay(store, monkeypatch, tmp_path):
    original = "EXACT-7219 " + "v" * 24_000
    conv = await store.create(model="m", system_prompt_hash="h")
    await store.append(conv.id, Message.user("Read a result."))
    await store.append(conv.id, Message(role="assistant", content="", tool_calls=[
        ToolCall(id="lookup-1", name="lookup", arguments={"name": "sample"})]))
    await store.append(conv.id, Message.tool(original, tool_call_id="lookup-1"))
    await store.append(conv.id, Message.assistant("Done."))
    await store.save_compaction(conv.id, CompactionState(clipped={0: 10}))
    answer = Message.assistant("Understood.", stop_reason="stop")
    answer.usage = Usage(prompt_tokens=300, completion_tokens=5, total_tokens=305)
    provider = ScriptedLLM([answer, answer, answer])
    monkeypatch.setattr("lovelaice.agent.agent._build_llm", lambda cfg: provider)

    async def open_chat(policy):
        def factory(*, conversation):
            return Agent.from_conversation(
                config=AgentConfig(model="m", system_prompt="SYS", context_window=8192,
                                   compaction_policy=policy),
                tools=[], loop=ReActNative(), conversation=conversation, store=store)

        server = AcpServer(agent_factory=factory, conversation_store=store)
        client = InProcessAcpClient(server)
        updates = []
        client.on_notification(lambda notification: updates.append(notification.params))
        await client.initialize()
        new = await client._call("session/new", {"cwd": str(tmp_path), "conversationId": conv.id})
        outcome = await client.session_prompt(new["sessionId"], "Continue.")
        assert outcome["stopReason"] == "end_turn"
        return updates

    default = CompactionPolicy()
    updates = await open_chat(default)
    [detail] = [ev for ev in updates if ev.get("event") == "compaction_detail"]
    assert detail["operations"] == [{"step": "finished_results", "count": 1}]
    assert detail["before_tokens"] > detail["after_tokens"] and detail["estimated"] is True
    assert any(ev.get("event") == "measured" and ev["estimated"] is False for ev in updates)
    assert any(ev.get("event") == "compacted" and ev["stubbed"] == 1 for ev in updates)
    checkpoint = (await store.get(conv.id)).compaction
    assert checkpoint.stubbed == [2] and not checkpoint.clipped

    unchanged = await open_chat(default)
    assert not any(ev.get("event") == "compaction_detail" for ev in unchanged)
    assert "result elided" in provider.calls[1].messages[3].content
    await open_chat(CompactionPolicy(steps=()))
    assert provider.calls[2].messages[3].content == original
    latest = await store.get(conv.id)
    assert latest.row.messages[2].content == original
    assert latest.compaction.policy_key != checkpoint.policy_key
    assert not latest.compaction.stubbed
    for call in provider.calls:
        check_tool_pairs(call.messages)
