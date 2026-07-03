import pytest

from lovelaice.workflows import WorkflowSpec, run


class _Agent:
    """Records prompts; returns a canned reply via messages_for_llm."""

    def __init__(self, reply="ok"):
        self._r = reply
        self.prompts = []

    async def prompt(self, text):
        self.prompts.append(text)
        return "end"

    def messages_for_llm(self):
        class _M:
            role = "assistant"
            content = self._r

        return [_M()]


@pytest.mark.asyncio
async def test_parallel_runs_children_and_collects_list():
    seen = []

    def factory():
        a = _Agent()
        a.prompts = seen  # share the sink so we see every child's prompt
        return a

    spec = WorkflowSpec.model_validate({
        "name": "p", "root": {"kind": "parallel", "name": "outs", "children": [
            {"kind": "agent", "prompt": "one", "name": "a"},
            {"kind": "agent", "prompt": "two", "name": "b"},
        ]}})
    result = await run(spec, agent_factory=factory)
    assert set(seen) == {"one", "two"}                  # both ran
    assert isinstance(result["items"], list) and len(result["items"]) == 2


@pytest.mark.asyncio
async def test_map_fans_over_named_list_binding_as():
    prompts = []

    def factory():
        a = _Agent()
        a.prompts = prompts
        return a

    spec = WorkflowSpec.model_validate({
        "name": "m", "root": {"kind": "sequence", "children": [
            {"kind": "map", "over": "xs", "as": "x", "name": "res",
             "node": {"kind": "agent", "prompt": "do {x}"}},
        ]}})
    await run(spec, agent_factory=factory, inputs={"xs": ["p", "q", "r"]})
    assert sorted(prompts) == ["do p", "do q", "do r"]  # each element bound to {x}


@pytest.mark.asyncio
async def test_map_over_non_list_raises():
    spec = WorkflowSpec.model_validate({
        "name": "m", "root": {"kind": "map", "over": "xs", "as": "x",
                              "node": {"kind": "agent", "prompt": "do {x}"}}})
    with pytest.raises(RuntimeError, match="over"):
        await run(spec, agent_factory=lambda: _Agent(), inputs={"xs": "notalist"})


@pytest.mark.asyncio
async def test_prompt_node_inside_parallel_is_rejected():
    async def handler(p, v):
        return "x"

    spec = WorkflowSpec.model_validate({
        "name": "bad", "root": {"kind": "parallel", "children": [
            {"kind": "prompt", "prompt": "hi"},
        ]}})
    with pytest.raises(RuntimeError, match="fan-out"):
        await run(spec, agent_factory=lambda: None, prompt_handler=handler)
