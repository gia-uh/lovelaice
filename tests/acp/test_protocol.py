import json
from lovelaice.acp.protocol import (
    JsonRpcRequest, JsonRpcResponse, JsonRpcNotification,
    parse_message, encode_message,
)


def test_parse_request():
    raw = '{"jsonrpc":"2.0","id":1,"method":"session/prompt","params":{"x":1}}'
    m = parse_message(json.loads(raw))
    assert isinstance(m, JsonRpcRequest)
    assert m.id == 1
    assert m.method == "session/prompt"
    assert m.params == {"x": 1}


def test_parse_response_with_result():
    raw = '{"jsonrpc":"2.0","id":1,"result":{"stopReason":"end_turn"}}'
    m = parse_message(json.loads(raw))
    assert isinstance(m, JsonRpcResponse)
    assert m.id == 1
    assert m.result == {"stopReason": "end_turn"}
    assert m.error is None


def test_parse_response_with_error():
    raw = '{"jsonrpc":"2.0","id":1,"error":{"code":-32601,"message":"not found"}}'
    m = parse_message(json.loads(raw))
    assert isinstance(m, JsonRpcResponse)
    assert m.id == 1
    assert m.error == {"code": -32601, "message": "not found"}


def test_parse_notification():
    raw = '{"jsonrpc":"2.0","method":"session/update","params":{"sessionUpdate":"agent_message_chunk"}}'
    m = parse_message(json.loads(raw))
    assert isinstance(m, JsonRpcNotification)
    assert m.method == "session/update"
    assert m.params == {"sessionUpdate": "agent_message_chunk"}


def test_parse_rejects_unknown_jsonrpc_version():
    import pytest
    raw = '{"jsonrpc":"1.0","method":"foo"}'
    with pytest.raises(ValueError, match="jsonrpc"):
        parse_message(json.loads(raw))


def test_parse_rejects_malformed():
    import pytest
    raw = '{"jsonrpc":"2.0"}'  # no method, no id+result/error
    with pytest.raises(ValueError):
        parse_message(json.loads(raw))


def test_encode_request_round_trip():
    r = JsonRpcRequest(method="session/prompt", id=1, params={"text": "hi"})
    s = encode_message(r)
    parsed = parse_message(json.loads(s))
    assert isinstance(parsed, JsonRpcRequest)
    assert parsed.method == "session/prompt"
    assert parsed.id == 1
    assert parsed.params == {"text": "hi"}


def test_encode_notification_round_trip():
    n = JsonRpcNotification(method="session/update", params={"sessionUpdate": "x"})
    s = encode_message(n)
    parsed = parse_message(json.loads(s))
    assert isinstance(parsed, JsonRpcNotification)
    assert parsed.method == "session/update"


def test_encode_response_with_result_round_trip():
    resp = JsonRpcResponse(id=1, result={"stopReason": "end_turn"})
    s = encode_message(resp)
    obj = json.loads(s)
    assert obj["jsonrpc"] == "2.0"
    assert obj["id"] == 1
    assert obj["result"] == {"stopReason": "end_turn"}
    assert "error" not in obj


def test_encode_response_with_error_round_trip():
    resp = JsonRpcResponse(id=1, error={"code": -32601, "message": "x"})
    s = encode_message(resp)
    obj = json.loads(s)
    assert obj["error"] == {"code": -32601, "message": "x"}
    assert "result" not in obj


def test_encode_request_without_params():
    r = JsonRpcRequest(method="initialize", id=1)
    s = encode_message(r)
    obj = json.loads(s)
    assert "params" not in obj
