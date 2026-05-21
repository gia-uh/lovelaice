import pytest
from lovelaice.agent.hooks import HookRegistry, Allow, Block, AskUser


def test_allow_block_askuser():
    a = Allow()
    b = Block("nope")
    u = AskUser(options=[{"optionId": "ok", "name": "OK", "kind": "allow_once"}])
    assert a.allowed is True
    assert b.allowed is False
    assert b.reason == "nope"
    assert u.allowed is False  # VS1: treated as block (pending → no permission flow yet)


@pytest.mark.asyncio
async def test_observational_hooks_fire_in_order():
    reg = HookRegistry()
    seen = []
    reg.register("turn_start", lambda ev: seen.append(("a", ev)))
    reg.register("turn_start", lambda ev: seen.append(("b", ev)))
    await reg.emit("turn_start", "x")
    assert seen == [("a", "x"), ("b", "x")]


@pytest.mark.asyncio
async def test_observational_hook_async_supported():
    reg = HookRegistry()
    seen = []

    async def async_handler(ev):
        seen.append(("async", ev))

    reg.register("turn_end", async_handler)
    await reg.emit("turn_end", "y")
    assert seen == [("async", "y")]


@pytest.mark.asyncio
async def test_reducer_tool_call_short_circuits_on_block():
    reg = HookRegistry()
    reg.register("tool_call", lambda call: Allow())
    reg.register("tool_call", lambda call: Block("not this"))
    reg.register("tool_call", lambda call: Allow())  # never reached
    result = await reg.reduce_tool_call("anything")
    assert isinstance(result, Block)
    assert result.reason == "not this"


@pytest.mark.asyncio
async def test_reducer_tool_call_all_allow_returns_allow():
    reg = HookRegistry()
    reg.register("tool_call", lambda call: Allow())
    reg.register("tool_call", lambda call: Allow())
    result = await reg.reduce_tool_call("call")
    assert isinstance(result, Allow)


@pytest.mark.asyncio
async def test_reducer_tool_call_empty_chain_returns_allow():
    reg = HookRegistry()
    result = await reg.reduce_tool_call("call")
    assert isinstance(result, Allow)


@pytest.mark.asyncio
async def test_reducer_tool_call_askuser_becomes_block_in_vs1():
    """AskUser is a stub in VS1 — treated as Block with a 'not yet implemented' reason."""
    reg = HookRegistry()
    reg.register("tool_call", lambda call: AskUser(options=[
        {"optionId": "ok", "name": "OK", "kind": "allow_once"},
    ]))
    result = await reg.reduce_tool_call("call")
    assert isinstance(result, Block)
    assert "VS2" in result.reason or "not yet" in result.reason.lower()


@pytest.mark.asyncio
async def test_reducer_tool_call_async_handler():
    reg = HookRegistry()

    async def async_block(call):
        return Block("async block")

    reg.register("tool_call", async_block)
    result = await reg.reduce_tool_call("call")
    assert isinstance(result, Block)
    assert result.reason == "async block"


@pytest.mark.asyncio
async def test_reducer_tool_call_none_return_means_allow():
    """Handler that returns None is equivalent to Allow (passes to next handler)."""
    reg = HookRegistry()
    seen = []
    reg.register("tool_call", lambda call: seen.append("first") or None)
    reg.register("tool_call", lambda call: Block("blocked"))
    result = await reg.reduce_tool_call("call")
    assert seen == ["first"]
    assert isinstance(result, Block)
