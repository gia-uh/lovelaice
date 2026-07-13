from lovelaice.agent.events import AssistantMessageDelta


def test_harness_emits_delta_on_token():
    from lovelaice.agent.harness import Harness
    from lovelaice.agent.tools import ToolRegistry
    from lovelaice.agent.hooks import HookRegistry

    class FakeLLM:
        _on_token = None

    h = Harness(llm=FakeLLM(), tools=ToolRegistry(), hooks=HookRegistry(),
                system_prompt="s")
    seen = []
    h.subscribe(lambda ev: seen.append(ev)
                if isinstance(ev, AssistantMessageDelta) else None)
    # lingo calls llm._on_token(token) per content token; simulate:
    h.llm._on_token("hel")
    h.llm._on_token("lo")
    assert [e.text for e in seen] == ["hel", "lo"]
