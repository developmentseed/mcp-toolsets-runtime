import sys
import types

import pytest
from langchain_core.tools import tool
from starlette.testclient import TestClient

from mcp_runtime.server import build_server
from mcp_runtime.tool_result import ToolResult
from mcp_runtime.credentials import (
    MissingCredentialError,
    credential_from_header,
    header_context,
)


def test_reads_header_case_insensitively():
    with header_context({"X-Demo-Token": "secret"}):
        assert credential_from_header("x-demo-token") == "secret"
        assert credential_from_header("X-Demo-Token") == "secret"


def test_missing_header_names_it_in_the_error():
    with header_context({"other": "value"}):
        with pytest.raises(MissingCredentialError, match="x-demo-token"):
            credential_from_header("x-demo-token")


def test_outside_a_request_raises():
    with pytest.raises(MissingCredentialError, match="x-demo-token"):
        credential_from_header("x-demo-token")


def test_header_context_resets():
    with header_context({"x-demo-token": "secret"}):
        credential_from_header("x-demo-token")
    with pytest.raises(MissingCredentialError):
        credential_from_header("x-demo-token")


def test_the_calling_requests_headers_reach_the_tool(monkeypatch):
    """End to end, because this is the part with no SDK equivalent.

    A tool body reads its credential from a context variable several frames
    below the handler, and what fills that variable is the middleware every
    built server carries. Reading it through a real request is the only check
    that the two are still connected.
    """

    @tool
    def whoami() -> ToolResult:
        """Report the caller's credential."""
        return {"message": credential_from_header("x-demo-token")}

    module = types.ModuleType("credential_toolset.tools")
    module.TOOLS = [whoami]
    module.CREDENTIAL_HEADERS = ["x-demo-token"]
    monkeypatch.setitem(sys.modules, "credential_toolset.tools", module)

    server = build_server("credential-toolset")
    # Bound to 127.0.0.1, where the SDK turns on DNS-rebinding protection and
    # checks the Host header — so the request has to come from there.
    with TestClient(
        server.streamable_http_app(), base_url="http://127.0.0.1:8000"
    ) as client:
        response = client.post(
            "/mcp",
            headers={
                "accept": "application/json, text/event-stream",
                "content-type": "application/json",
                "x-demo-token": "secret",
            },
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "whoami", "arguments": {}},
            },
        )
    assert response.status_code == 200
    assert "secret" in response.text
