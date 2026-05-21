import asyncio
import pytest
from unittest.mock import AsyncMock
from lingo.llm import Message
from lovelaice.agent.tools import ToolRegistry
from lovelaice.agent.hooks import HookRegistry
from lovelaice.agent.harness import Harness


@pytest.mark.asyncio
async def test_harness_llm_call_delegates_to_lingo():
    mock_llm = AsyncMock()
    mock_llm.chat = AsyncMock(return_value=Message.assistant("hi", stop_reason="stop"))

    h = Harness(
        llm=mock_llm,
        tools=ToolRegistry(),
        hooks=HookRegistry(),
        system_prompt="SYS",
    )

    msg = await h.llm_call([Message.user("hello")])
    assert msg.content == "hi"
    mock_llm.chat.assert_awaited_once()


@pytest.mark.asyncio
async def test_harness_llm_call_passes_tools_kwarg():
    mock_llm = AsyncMock()
    mock_llm.chat = AsyncMock(return_value=Message.assistant("hi"))

    from lingo.tools import tool as lingo_tool

    @lingo_tool
    async def echo(text: str) -> str:
        """Echo."""
        return text

    h = Harness(
        llm=mock_llm,
        tools=ToolRegistry(),
        hooks=HookRegistry(),
        system_prompt="SYS",
    )

    msg = await h.llm_call([Message.user("hi")], tools=[echo])
    call = mock_llm.chat.call_args
    assert call.kwargs.get("tools") == [echo]


@pytest.mark.asyncio
async def test_harness_subscribe_and_emit():
    h = Harness(llm=None, tools=ToolRegistry(), hooks=HookRegistry(),
                system_prompt="x")
    seen = []
    h.subscribe(lambda ev: seen.append(ev))
    h.emit("some_event")
    assert seen == ["some_event"]


@pytest.mark.asyncio
async def test_harness_before_llm_call_reducer_can_transform():
    """A `before_llm_call` hook receives (messages, tools) and may return a
    rewritten tuple — the harness uses the rewritten values for the LLM call."""
    mock_llm = AsyncMock()
    mock_llm.chat = AsyncMock(return_value=Message.assistant("hi"))

    hooks = HookRegistry()
    sentinel = [Message.user("rewritten")]

    def rewrite(messages, tools):
        return sentinel, tools

    hooks.register("before_llm_call", rewrite)
    h = Harness(llm=mock_llm, tools=ToolRegistry(), hooks=hooks,
                system_prompt="x")
    await h.llm_call([Message.user("original")])
    call = mock_llm.chat.call_args
    # First positional arg or `messages` kwarg should be the rewritten list.
    sent_messages = call.args[0] if call.args else call.kwargs.get("messages")
    assert sent_messages is sentinel


@pytest.mark.asyncio
async def test_harness_default_abort_signal():
    h = Harness(llm=None, tools=ToolRegistry(), hooks=HookRegistry(),
                system_prompt="x")
    assert isinstance(h.abort, asyncio.Event)
    assert not h.abort.is_set()
