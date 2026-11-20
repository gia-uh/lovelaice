"""``output_schema`` is enforced: extract JSON, validate, retry, then fail loud.

The model replies are canned; each test asserts what the workflow does with
them — pull the first JSON value out of fenced or prosey text, validate it
against the node's schema, re-prompt the same agent with the validation error,
and raise a named error once the attempts run out.
"""

import pytest
from lingo.llm import Message

from lovelaice.agent import Agent, AgentConfig
from lovelaice.agent.loops.react_native import ReActNative
from lovelaice.workflows import WorkflowSpec, run
from lovelaice.workflows.executor import OutputSchemaError, _extract_json

SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "count": {"type": "integer"},
    },
    "required": ["title", "count"],
    "additionalProperties": False,
}


def _chat_factory(monkeypatch, tmp_path, replies):
    """A factory whose agents draw canned replies in order; returns the prompts
    each LLM call received (last message: the user/repair turn)."""
    queue = list(replies)
    seen: list[str] = []
    made = {"n": 0}

    def make() -> Agent:
        i = made["n"]
        made["n"] = i + 1
        fake = _FakeChat(queue, seen)
        monkeypatch.setattr(
            "lovelaice.agent.agent._build_llm", lambda cfg, _f=fake: _f
        )
        return Agent(
            config=AgentConfig(model="m", system_prompt="SYS"),
            tools=[],
            loop=ReActNative(),
            session_path=tmp_path / f"s_{i}.jsonl",
        )

    return make, seen


class _FakeChat:
    def __init__(self, queue, seen):
        self._queue = queue
        self._seen = seen
        self.tools = None

    async def chat(self, messages, tools=None):
        if not self._queue:
            raise AssertionError("the model was called more times than replies were canned")
        self._seen.append(messages[-1].content)
        return Message.assistant(self._queue.pop(0), stop_reason="stop")


def _spec(node: dict, **root_extra):
    return WorkflowSpec.model_validate(
        {
            "name": "schema",
            "root": {"kind": "agent", "prompt": "emit it", **node, **root_extra},
        }
    )


def test_retries_defaults_to_two_and_can_be_set_per_node():
    spec = WorkflowSpec.model_validate(
        {"name": "x", "root": {"kind": "agent", "prompt": "p", "output_schema": {"type": "object"}}}
    )
    assert spec.root.retries == 2  # type: ignore[union-attr]
    spec = WorkflowSpec.model_validate(
        {
            "name": "x",
            "root": {"kind": "prompt", "prompt": "p", "output_schema": {"type": "object"}, "retries": 5},
        }
    )
    assert spec.root.retries == 5  # type: ignore[union-attr]


def test_retries_cannot_be_negative():
    with pytest.raises(ValueError):
        WorkflowSpec.model_validate(
            {"name": "x", "root": {"kind": "agent", "prompt": "p", "retries": -1}}
        )


def test_extract_json_returns_a_fenced_object():
    text = 'Sure!\n```json\n{"title": "T", "count": 2}\n```\nDone.'
    assert _extract_json(text) == {"title": "T", "count": 2}


def test_extract_json_takes_the_first_value_after_prose():
    text = 'Here is the result:\n{"title": "T", "count": 2}\nHope that helps.'
    assert _extract_json(text) == {"title": "T", "count": 2}


def test_extract_json_prefers_the_fenced_value_over_prose_brackets():
    text = 'See [1] and {not json}.\n```\n{"title": "T", "count": 2}\n```'
    assert _extract_json(text) == {"title": "T", "count": 2}


def test_extract_json_raises_when_there_is_no_json():
    with pytest.raises(ValueError, match="no JSON"):
        _extract_json("I could not do that.")


@pytest.mark.asyncio
async def test_fenced_json_is_accepted(monkeypatch, tmp_path):
    factory, seen = _chat_factory(
        monkeypatch, tmp_path, ['```json\n{"title": "T", "count": 2}\n```']
    )
    out = await run(_spec({"name": "draft", "output_schema": SCHEMA}), agent_factory=factory)
    assert out == {"title": "T", "count": 2}
    assert len(seen) == 1  # no retry needed


@pytest.mark.asyncio
async def test_json_after_leading_prose_is_accepted(monkeypatch, tmp_path):
    factory, seen = _chat_factory(
        monkeypatch, tmp_path, ['The answer:\n{"title": "T", "count": 2}\n-- end']
    )
    out = await run(_spec({"output_schema": SCHEMA}), agent_factory=factory)
    assert out == {"title": "T", "count": 2}
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_schema_mismatch_is_retried_on_the_same_agent(monkeypatch, tmp_path):
    factory, seen = _chat_factory(
        monkeypatch,
        tmp_path,
        [
            '{"title": "T"}',                      # missing "count"
            '{"title": "T", "count": 2}',          # fixed on retry
        ],
    )
    result = await run(
        _spec({"name": "draft", "output_schema": SCHEMA, "retries": 1}),
        agent_factory=factory,
    )
    assert result == {"title": "T", "count": 2}
    assert len(seen) == 2
    assert "count" in seen[1] and "required property" in seen[1]
    assert "emit it" not in seen[1]                # the retry is about the output, not a new task


@pytest.mark.asyncio
async def test_unparseable_reply_is_also_retried(monkeypatch, tmp_path):
    factory, seen = _chat_factory(
        monkeypatch,
        tmp_path,
        ["I would rather chat.", '{"title": "T", "count": 2}'],
    )
    out = await run(
        _spec({"output_schema": SCHEMA, "retries": 1}), agent_factory=factory
    )
    assert out == {"title": "T", "count": 2}
    assert "no JSON" in seen[1]


@pytest.mark.asyncio
async def test_exhausted_retries_raise_an_error_naming_node_and_last_error(
    monkeypatch, tmp_path
):
    factory, seen = _chat_factory(
        monkeypatch, tmp_path, ["not json"] * 3
    )
    with pytest.raises(OutputSchemaError) as excinfo:
        await run(
            _spec({"name": "draft", "output_schema": SCHEMA}),  # default retries=2
            agent_factory=factory,
        )
    message = str(excinfo.value)
    assert "draft" in message
    assert "no JSON" in message
    assert len(seen) == 3  # one attempt + two retries


@pytest.mark.asyncio
async def test_invalid_schema_fails_before_calling_the_model(monkeypatch, tmp_path):
    factory, seen = _chat_factory(monkeypatch, tmp_path, [])  # no reply canned
    bad = {"type": "object", "properties": {"x": {"type": "nope"}}}
    with pytest.raises(OutputSchemaError, match="invalid output_schema"):
        await run(_spec({"name": "draft", "output_schema": bad}), agent_factory=factory)
    assert seen == []


@pytest.mark.asyncio
async def test_retries_zero_means_one_attempt(monkeypatch, tmp_path):
    factory, seen = _chat_factory(monkeypatch, tmp_path, ['{"title": "T"}'])
    with pytest.raises(OutputSchemaError):
        await run(
            _spec({"name": "draft", "output_schema": SCHEMA, "retries": 0}),
            agent_factory=factory,
        )
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_validated_output_is_stored_raw_for_passthrough(monkeypatch, tmp_path):
    # The parsed list must reach a later map as a list, not as its text.
    factory, seen = _chat_factory(
        monkeypatch,
        tmp_path,
        ['```json\n["a", "b"]\n```', "echo a", "echo b"],
    )
    spec = WorkflowSpec.model_validate(
        {
            "name": "chain",
            "root": {
                "kind": "sequence",
                "children": [
                    {
                        "kind": "agent",
                        "prompt": "emit it",
                        "name": "rows",
                        "output_schema": {"type": "array", "items": {"type": "string"}},
                    },
                    {
                        "kind": "map",
                        "over": "rows",
                        "as": "r",
                        "node": {"kind": "agent", "prompt": "echo {r}"},
                    },
                ],
            },
        }
    )
    out = await run(spec, agent_factory=factory)
    assert sorted(seen[1:]) == ["echo a", "echo b"]
    assert len(out["items"]) == 2


@pytest.mark.asyncio
async def test_prompt_node_enforces_its_schema_and_retries():
    replies = iter(['{"title": "T"}', '{"title": "T", "count": 2}'])
    seen = []

    async def handler(prompt, vars):
        seen.append(prompt)
        return next(replies)

    spec = WorkflowSpec.model_validate(
        {
            "name": "wf",
            "root": {
                "kind": "prompt",
                "prompt": "emit it",
                "name": "crit",
                "output_schema": SCHEMA,
                "retries": 1,
            },
        }
    )
    out = await run(spec, agent_factory=lambda: None, prompt_handler=handler)
    assert out == {"title": "T", "count": 2}
    assert len(seen) == 2 and "required property" in seen[1]


@pytest.mark.asyncio
async def test_prompt_node_raises_named_error_when_exhausted():
    async def handler(prompt, vars):
        return "still prose"

    spec = WorkflowSpec.model_validate(
        {
            "name": "wf",
            "root": {
                "kind": "prompt",
                "prompt": "emit it",
                "name": "crit",
                "output_schema": SCHEMA,
                "retries": 0,
            },
        }
    )
    with pytest.raises(OutputSchemaError, match="crit"):
        await run(spec, agent_factory=lambda: None, prompt_handler=handler)
