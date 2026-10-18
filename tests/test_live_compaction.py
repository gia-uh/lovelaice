"""Live: a long tool-heavy conversation with a small open model on OpenRouter.

Ten turns, each reading a ~12,000-character page or a ~3,000-character note,
with the last turn asking for a fact from turn 1. With a 16,384-token window
the budget must compact and every prompt the provider reports must fit; with
no window (the provider's real one is 262,144) nothing compacts.

Run: OPENROUTER_API_KEY=... uv run pytest tests/test_live_compaction.py -s
"""
import os

import pytest
from lingo.tools import tool as lingo_tool

from lovelaice.agent import Agent, AgentConfig
from lovelaice.agent.events import AssistantMessageFinalized, ContextCompacted
from lovelaice.agent.loops.react_native import ReActNative
from lovelaice.agent.tools import AgentTool

MODEL = "qwen/qwen3.5-9b"
FACTS = {1: "The project code name is BLUE HERON.",
         2: "The budget approved for phase two is 48,000 euros.",
         3: "The pilot site is the harbour office in Cienfuegos.",
         4: "The lead engineer for the sensors is Marta Ruiz.",
         5: "The deadline for the field test is 14 March."}
FILLER = ("This section of the report describes routine procedures, meeting notes and "
          "logistics that are not relevant to the key decisions. ")


@lingo_tool
async def fetch_page(n: int) -> str:
    """Fetch page n of the project report."""
    fact = FACTS.get(n, f"Page {n} has no key facts.")
    body = FILLER * 55
    return f"PAGE {n}\n{body[:6000]}\nKEY FACT: {fact}\n{body[:6000]}"


@lingo_tool
async def read_note(name: str) -> str:
    """Read a note by name."""
    return f"NOTE {name}\n" + FILLER * 25


PROMPTS = [
    "Use fetch_page to read page 1 of the project report and tell me its key fact.",
    "Now read page 2 with fetch_page. What is its key fact?",
    "Read the note called 'minutes' with read_note and summarise it in one line.",
    "Read page 3 with fetch_page and tell me the key fact.",
    "Read page 4 with fetch_page and tell me the key fact.",
    "Read the note called 'risks' with read_note. Anything important?",
    "Read page 5 with fetch_page and tell me its key fact.",
    "Compare the key facts of pages 3 and 5 in one sentence.",
    "Read page 6 with fetch_page. Does it add anything?",
    "Without reading anything again: what is the project code name from page 1?",
]


@pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"), reason="needs OPENROUTER_API_KEY")
@pytest.mark.parametrize("window", [16384, 8192, 4096, None])
@pytest.mark.asyncio
async def test_qwen_9b_long_conversation(tmp_path, window):
    agent = Agent(
        config=AgentConfig(model=MODEL, system_prompt="You are a careful research assistant.",
                           api_key=os.environ["OPENROUTER_API_KEY"],
                           base_url="https://openrouter.ai/api/v1",
                           max_tokens=1024, context_window=window),
        tools=[AgentTool(inner=fetch_page), AgentTool(inner=read_note)],
        loop=ReActNative(), session_path=tmp_path / "s.jsonl")
    prompt_tokens, reports = [], []

    def watch(ev):
        if isinstance(ev, AssistantMessageFinalized) and ev.message.usage:
            prompt_tokens.append(ev.message.usage.prompt_tokens)
        if isinstance(ev, ContextCompacted):
            reports.append(ev.report)

    agent.subscribe(watch)
    for p in PROMPTS:
        await agent.prompt(p)
    final = agent.session.messages_for_llm("")[-1].content

    print(f"\nwindow={window} prompt_tokens per call: {prompt_tokens}")
    for r in reports:
        print(f"  compacted {r.before_tokens} -> {r.after_tokens}: stubs "
              f"{r.stubbed_finished}+{r.stubbed_current}, turns summarized "
              f"{r.turns_summarized}, clipped {r.clipped}")
        if r.summary:
            print("  summary:", r.summary[:600].replace("\n", " "))
    print("final answer:", final)

    if window is None:
        assert not reports
    else:
        assert reports, f"a {window:,} window should have compacted"
        assert max(prompt_tokens) <= window
    # At 4,096 a 3,000-token page cannot fit the target, so step 4 clips its
    # middle, which is where the key fact sits: the model never sees it. That
    # loss is step 4's documented cost, so recall is asserted from 8,192 up.
    if window is None or window >= 8192:
        assert "heron" in final.lower()
        # The summary rolls: every run must still carry the turn-1 fact.
        for r in reports:
            if r.summary:
                assert "heron" in r.summary.lower(), r.summary


@pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"), reason="needs OPENROUTER_API_KEY")
@pytest.mark.asyncio
async def test_qwen_9b_writes_a_non_empty_summary(tmp_path):
    """Qwen 3.5 9B thinks before answering; a summary call capped at the
    summary's size came back empty. The summarizer must get real text."""
    from lingo.llm import Message
    agent = Agent(
        config=AgentConfig(model=MODEL, api_key=os.environ["OPENROUTER_API_KEY"],
                           base_url="https://openrouter.ai/api/v1", max_tokens=1024),
        tools=[], loop=ReActNative(), session_path=tmp_path / "s.jsonl")
    msgs = [Message.user("Read page 1 of the report."),
            Message.assistant("Page 1 says the project code name is BLUE HERON."),
            Message.user("And page 2?"),
            Message.assistant("Page 2 says the phase-two budget is 48,000 euros.")]
    text, usage = await agent.harness.summarizer()(None, msgs, 307)
    print("\nsummary:", text)
    assert "heron" in text.lower()
