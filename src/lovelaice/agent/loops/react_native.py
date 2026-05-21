"""ReActNative — canonical agentic loop.

One assistant message → N tool calls → tool-results → loop until no tool calls.
Uses lingo's native tool-call wire (Message.tool_calls).
"""
from typing import Protocol

from lingo.llm import Message
from lovelaice.agent.errors import StopReason
from lovelaice.agent.events import (
    TurnStart, TurnEnd, AssistantMessageFinalized,
)


class AgentLoop(Protocol):
    """Pluggable agentic loop. Implementations decide how to orchestrate
    LLM calls and tool dispatch against the harness."""

    async def run(self, harness, session, user_message: Message) -> StopReason: ...


def _map_stop_reason(lingo_reason: str | None, soft_terminate: bool) -> StopReason:
    """Map lingo's Message.stop_reason to ACP's StopReason enum.

    Soft-termination (all tools returned terminate=True) is end_turn.
    """
    if soft_terminate:
        return StopReason.END_TURN
    if lingo_reason in (None, "stop", "tool_calls"):
        return StopReason.END_TURN
    if lingo_reason == "length":
        return StopReason.MAX_TOKENS
    if lingo_reason in ("content_filter", "error"):
        return StopReason.REFUSAL
    if lingo_reason == "aborted":
        return StopReason.CANCELLED
    return StopReason.END_TURN


class ReActNative:
    """Canonical agentic loop.

    Each iteration: snapshot messages → LLM call → if no tool calls, end;
    else dispatch tools, append results, loop. Soft-terminates if all
    tools in a batch return terminate=True (unanimous).
    """

    async def run(self, harness, session, user_message: Message) -> StopReason:
        session.append(user_message)
        harness.emit(TurnStart(
            turn_no=session.turn_count,
            model=getattr(harness.llm, "model", "?"),
        ))

        while True:
            if harness.abort.is_set():
                harness.emit(TurnEnd(stop_reason=StopReason.CANCELLED,
                                     soft_terminate=False))
                return StopReason.CANCELLED

            messages = session.messages_for_llm(harness.system_prompt)
            assistant = await harness.llm_call(
                messages,
                tools=harness.tools.lingo_tools() or None,
            )
            session.append(assistant)
            harness.emit(AssistantMessageFinalized(message=assistant))

            if not assistant.tool_calls:
                stop = _map_stop_reason(assistant.stop_reason, soft_terminate=False)
                harness.emit(TurnEnd(stop_reason=stop, soft_terminate=False))
                return stop

            results = await harness.execute_tools_batch(assistant.tool_calls)
            for call, result in zip(assistant.tool_calls, results):
                # Append a tool-role lingo.Message with the textual content.
                # tool_call_id is required by the OpenAI API to correlate the
                # tool result with the tool call in the assistant message.
                content_text = (
                    result.content[0]["text"]
                    if result.content and isinstance(result.content[0], dict)
                       and "text" in result.content[0]
                    else ""
                )
                session.append(Message.tool(content_text, tool_call_id=call.id))

            if results and all(r.terminate for r in results):
                harness.emit(TurnEnd(stop_reason=StopReason.END_TURN,
                                     soft_terminate=True))
                return StopReason.END_TURN
            # Loop continues.
