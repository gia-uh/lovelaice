"""JSON-RPC 2.0 message types for the ACP (Agent Client Protocol) wire.

Per https://agentclientprotocol.com — JSON-RPC 2.0 over stdio.
"""
import json
from dataclasses import dataclass
from typing import Any, Union


JSONRPC_VERSION = "2.0"


@dataclass
class JsonRpcRequest:
    method: str
    id: int | str
    params: dict | None = None


@dataclass
class JsonRpcResponse:
    id: int | str
    result: Any = None
    error: dict | None = None


@dataclass
class JsonRpcNotification:
    method: str
    params: dict | None = None


JsonRpcMessage = Union[JsonRpcRequest, JsonRpcResponse, JsonRpcNotification]


def parse_message(obj: dict) -> JsonRpcMessage:
    """Parse a decoded JSON object into one of the three JSON-RPC message types."""
    if obj.get("jsonrpc") != JSONRPC_VERSION:
        raise ValueError(f"unsupported jsonrpc version: {obj.get('jsonrpc')!r}")
    if "method" in obj:
        if "id" in obj:
            return JsonRpcRequest(
                method=obj["method"], id=obj["id"], params=obj.get("params"),
            )
        return JsonRpcNotification(
            method=obj["method"], params=obj.get("params"),
        )
    if "id" in obj:
        return JsonRpcResponse(
            id=obj["id"],
            result=obj.get("result"),
            error=obj.get("error"),
        )
    raise ValueError("malformed JSON-RPC message: no method, no id+result/error")


def encode_message(msg: JsonRpcMessage) -> str:
    """Encode a JSON-RPC message to its on-the-wire JSON string."""
    if isinstance(msg, JsonRpcRequest):
        body: dict = {"jsonrpc": JSONRPC_VERSION, "id": msg.id, "method": msg.method}
        if msg.params is not None:
            body["params"] = msg.params
    elif isinstance(msg, JsonRpcNotification):
        body = {"jsonrpc": JSONRPC_VERSION, "method": msg.method}
        if msg.params is not None:
            body["params"] = msg.params
    elif isinstance(msg, JsonRpcResponse):
        body = {"jsonrpc": JSONRPC_VERSION, "id": msg.id}
        if msg.error is not None:
            body["error"] = msg.error
        else:
            body["result"] = msg.result
    else:
        raise TypeError(f"cannot encode {type(msg).__name__}")
    return json.dumps(body)
