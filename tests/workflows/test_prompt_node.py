import pytest

from lovelaice.workflows import WorkflowSpec, run


def _spec():
    return WorkflowSpec.model_validate(
        {
            "name": "wf",
            "root": {
                "kind": "sequence",
                "children": [
                    {"kind": "prompt", "prompt": "hello {who}", "name": "greeting"},
                ],
            },
        }
    )


@pytest.mark.asyncio
async def test_prompt_node_routes_to_handler_and_threads_vars():
    seen = {}

    async def handler(prompt, vars):
        seen["prompt"] = prompt
        seen["vars"] = dict(vars)
        return "HI THERE"

    result = await run(
        _spec(),
        agent_factory=lambda: None,
        prompt_handler=handler,
        inputs={"who": "world"},
    )
    assert seen["prompt"] == "hello world"          # {who} rendered from vars
    assert result == {"text": "HI THERE"}           # last child's result


@pytest.mark.asyncio
async def test_prompt_node_without_handler_raises():
    with pytest.raises(RuntimeError, match="prompt_handler"):
        await run(_spec(), agent_factory=lambda: None, inputs={"who": "x"})
