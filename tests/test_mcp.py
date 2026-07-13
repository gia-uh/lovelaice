"""MCP integration: tool wrapping and registration with `mcp_server_tool` naming."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from lovelaice.mcp import _mcp_display_name, _params_from_input_schema, _wrap_mcp_tool


def test_mcp_display_name() -> None:
    assert _mcp_display_name("filesystem", "read_file") == "mcp_filesystem_read_file"


def test_params_from_input_schema_handles_types() -> None:
    schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "count": {"type": "integer"},
            "force": {"type": "boolean"},
        },
        "required": ["path"],
    }
    params = _params_from_input_schema(schema)
    assert params == {"path": str, "count": int, "force": bool}


def test_params_from_input_schema_missing_props_is_empty() -> None:
    assert _params_from_input_schema({}) == {}
    assert _params_from_input_schema({"type": "object"}) == {}


@pytest.mark.asyncio
async def test_wrap_mcp_tool_calls_session() -> None:
    """The wrapped tool's run() invokes session.call_tool and returns the text content."""
    session = MagicMock()
    text_part = MagicMock(); text_part.text = "hello from mcp"
    result_msg = MagicMock(); result_msg.content = [text_part]
    session.call_tool = AsyncMock(return_value=result_msg)

    mcp_tool = MagicMock()
    mcp_tool.name = "read_file"
    mcp_tool.description = "Read a file"
    mcp_tool.inputSchema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }

    wrapped = _wrap_mcp_tool(server_name="filesystem", tool=mcp_tool, session=session)
    assert wrapped.name == "mcp_filesystem_read_file"
    assert "Read a file" in wrapped.description
    assert wrapped.parameters() == {"path": str}

    out = await wrapped.run(path="/tmp/x")
    session.call_tool.assert_awaited_once_with("read_file", {"path": "/tmp/x"})
    assert "hello from mcp" in out


def test_mcp_tool_carries_input_schema_verbatim() -> None:
    """The wrapped tool retains the FastMCP inputSchema verbatim on json_schema,
    so the lingo schema builder serializes it without flattening."""
    input_schema = {
        "type": "object",
        "properties": {
            "vault_id": {"type": "string", "description": "vault"},
            "path": {"type": "string", "description": "note path"},
            "limit": {"type": "integer", "default": 50},
        },
        "required": ["vault_id", "path"],
    }
    mcp_tool = MagicMock()
    mcp_tool.name = "read_note"
    mcp_tool.description = "Read a note."
    mcp_tool.inputSchema = input_schema

    wrapped = _wrap_mcp_tool(server_name="magpie", tool=mcp_tool, session=None)
    assert wrapped.json_schema == input_schema
    # parameters() still returns the flattened map for back-compat.
    assert wrapped.parameters() == {"vault_id": str, "path": str, "limit": int}
