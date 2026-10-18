"""The additive `context` session/update on the legacy (0.1) ACP server."""
from lingo.llm import Usage

from lovelaice.acp.server import AcpServer
from lovelaice.agent.budget import CompactionReport
from lovelaice.agent.events import ContextCompacted, ContextOverflow, ContextOverheadWarning


def notifications_for(ev):
    server = AcpServer(agent_factory=lambda: None)
    seen = []
    server.on_notification(lambda n: seen.append((n.method, n.params)))
    server._agent_event_to_notification("sid1", ev)
    return seen


def test_context_compacted_shape():
    rep = CompactionReport(before_tokens=24610, after_tokens=15900, usable=30720,
                           fixed_tokens=6100, stubbed_finished=5, stubbed_current=3,
                           turns_summarized=5, clipped=0, summary="S")
    [(method, params)] = notifications_for(ContextCompacted(report=rep))
    assert method == "session/update"
    assert params == {
        "sessionId": "sid1", "sessionUpdate": "context", "event": "compacted",
        "before_tokens": 24610, "after_tokens": 15900, "usable": 30720,
        "fixed_tokens": 6100, "stubbed": 8, "turns_summarized": 5, "clipped": 0,
        "summary": "S",
    }


def test_context_warning_shape():
    [(_, params)] = notifications_for(ContextOverheadWarning(fixed_tokens=6900, usable=6144))
    assert params == {"sessionId": "sid1", "sessionUpdate": "context", "event": "warning",
                      "fixed_tokens": 6900, "usable": 6144}


def test_context_overflow_shape():
    ev = ContextOverflow(message="The prompt does not fit…", fixed_tokens=6900,
                         usable=6144, prompt_tokens=9100)
    [(_, params)] = notifications_for(ev)
    assert params == {"sessionId": "sid1", "sessionUpdate": "context", "event": "overflow",
                      "fixed_tokens": 6900, "usable": 6144, "prompt_tokens": 9100,
                      "message": "The prompt does not fit…"}


def test_summary_usage_is_reported():
    rep = CompactionReport(before_tokens=10, after_tokens=5, usable=100, fixed_tokens=1,
                           turns_summarized=2, summary="S",
                           summary_usage=Usage(prompt_tokens=300, completion_tokens=40,
                                               total_tokens=340))
    seen = notifications_for(ContextCompacted(report=rep))
    assert [p["sessionUpdate"] for _, p in seen] == ["context", "usage"]
    assert seen[1][1]["usage"] == {"prompt_tokens": 300, "completion_tokens": 40,
                                   "total_tokens": 340}
