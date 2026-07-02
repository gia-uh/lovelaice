from lovelaice.agent.events import (
    TurnStart, TurnEnd, AssistantMessageFinalized,
    ToolExecutionStart, ToolExecutionUpdate, ToolExecutionEnd,
    SessionAppend,
)
from lovelaice.agent.errors import StopReason


def test_stop_reason_enum():
    assert StopReason.END_TURN.value == "end_turn"
    assert StopReason.CANCELLED.value == "cancelled"
    assert StopReason.MAX_TOKENS.value == "max_tokens"
    assert StopReason.MAX_TURN_REQUESTS.value == "max_turn_requests"
    assert StopReason.REFUSAL.value == "refusal"


def test_event_dataclasses():
    ev = TurnStart(turn_no=1, model="claude-sonnet-4-6")
    assert ev.turn_no == 1
    assert ev.model == "claude-sonnet-4-6"

    ev = TurnEnd(stop_reason=StopReason.END_TURN, soft_terminate=False)
    assert ev.stop_reason == StopReason.END_TURN
    assert ev.soft_terminate is False


def test_tool_call_repaired_event_shape():
    from lovelaice.agent.events import ToolCallRepaired
    e = ToolCallRepaired(call_id="c1", name="grep",
                         original_args={"pattern": "x"},
                         repaired_args={"pattern": "x", "path": "."},
                         error="path required")
    assert e.repaired_args["path"] == "."
    assert e.original_args == {"pattern": "x"}
    assert e.name == "grep"
