import httpx2
import pytest
from pydantic import ValidationError
from typer.main import get_command

from mcp_agent.main import (
    app,
    AgentSettings,
    connect_error_hint,
    connect_failure,
    connections_from,
    credential_client_factory,
    credential_headers_from,
    first_leaf,
    health_url_for,
    user_credentials,
    with_credential_support,
)


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("PROVIDER_API_KEY", "sk-test")
    monkeypatch.setenv("PROVIDER_MODEL", "openai:gpt-4o-mini")
    settings = AgentSettings(_env_file=None)
    assert settings.provider_api_key.get_secret_value() == "sk-test"
    assert settings.provider_model == "openai:gpt-4o-mini"


def test_settings_from_dotenv(monkeypatch, tmp_path):
    monkeypatch.delenv("PROVIDER_API_KEY", raising=False)
    monkeypatch.delenv("PROVIDER_MODEL", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "PROVIDER_API_KEY=sk-dotenv\nPROVIDER_MODEL=openai:gpt-4o-mini\nUNRELATED=ignored\n"
    )
    settings = AgentSettings(_env_file=env_file)
    assert settings.provider_api_key.get_secret_value() == "sk-dotenv"
    assert settings.provider_model == "openai:gpt-4o-mini"


def test_settings_require_provider(monkeypatch):
    # Both PROVIDER_MODEL and PROVIDER_API_KEY are required — no default provider.
    monkeypatch.delenv("PROVIDER_API_KEY", raising=False)
    monkeypatch.delenv("PROVIDER_MODEL", raising=False)
    with pytest.raises(ValidationError):
        AgentSettings(_env_file=None)


def test_connections_from_index_payload():
    payload = {
        "connections": {
            "dataset-search": {
                "transport": "streamable_http",
                "url": "https://mcp.example.com/dataset-search/mcp",
            }
        },
        "toolsets": [],
    }
    assert (
        connections_from("https://mcp.example.com/", payload)
        == (payload["connections"])
    )


def test_connections_from_non_index_payload_wraps_url():
    expected = {
        "server": {"transport": "streamable_http", "url": "http://localhost:8000/mcp"}
    }
    assert connections_from("http://localhost:8000/mcp", None) == expected
    assert connections_from("http://localhost:8000/mcp", {"status": "ok"}) == expected


def test_credential_headers_from_index_payload():
    payload = {
        "connections": {},
        "toolsets": [
            {"name": "credential-demo", "credential_headers": ["X-Demo-Token"]},
            {"name": "dataset-search", "credential_headers": []},
            {"name": "hello"},
        ],
    }
    assert credential_headers_from(payload) == {
        "credential-demo": ["x-demo-token"],
        "dataset-search": [],
        "hello": [],
    }


def test_credential_headers_from_non_index_payload():
    assert credential_headers_from(None) is None
    assert credential_headers_from({"status": "ok"}) is None


async def test_credential_factory_injects_only_declared_headers():
    factory = credential_client_factory(["x-demo-token"])
    with user_credentials({"X-Demo-Token": "secret", "x-other-cred": "nope"}):
        client = factory(headers={"existing": "kept"})
    async with client:
        assert client.headers["x-demo-token"] == "secret"
        assert client.headers["existing"] == "kept"
        assert "x-other-cred" not in client.headers


async def test_credential_factory_without_declaration_sends_all():
    factory = credential_client_factory(None)
    with user_credentials({"x-demo-token": "secret", "x-other-cred": "yes"}):
        client = factory()
    async with client:
        assert client.headers["x-demo-token"] == "secret"
        assert client.headers["x-other-cred"] == "yes"


async def test_credential_factory_outside_context_injects_nothing():
    factory = credential_client_factory(["x-demo-token"])
    with user_credentials({"x-demo-token": "secret"}):
        pass  # context exited: credentials no longer available
    async with factory() as client:
        assert "x-demo-token" not in client.headers


def test_with_credential_support_wires_every_connection():
    connections = {
        "credential-demo": {"transport": "streamable_http", "url": "http://a/mcp"},
        "dataset-search": {"transport": "streamable_http", "url": "http://b/mcp"},
    }
    wired = with_credential_support(connections, {"credential-demo": ["x-demo-token"]})
    assert set(wired) == set(connections)
    assert all(
        callable(client.transport.httpx_client_factory) for client in wired.values()
    )
    assert connections["credential-demo"] == {  # untouched
        "transport": "streamable_http",
        "url": "http://a/mcp",
    }


def test_the_credential_factory_takes_what_the_transport_passes_it():
    """fastmcp calls the factory with `follow_redirects`, which the old adapter
    did not. A factory that refuses an argument the transport passes fails at
    connect time, where it reads as the server being unreachable."""
    factory = credential_client_factory(["x-demo-token"])

    client = factory(headers={"accept": "application/json"}, follow_redirects=True)

    assert client.headers["accept"] == "application/json"


def test_connect_error_hint_only_for_urls_missing_mcp_path():
    assert "under /mcp" in connect_error_hint("http://localhost:8000")
    assert "under /mcp" in connect_error_hint("http://localhost:8000/")
    assert connect_error_hint("http://localhost:8000/mcp") == ""
    assert connect_error_hint("https://mcp.example.com/credential-demo/mcp/") == ""


def test_first_leaf_unwraps_nested_groups():
    error = ValueError("inner")
    group = ExceptionGroup("outer", [ExceptionGroup("nested", [error])])
    assert first_leaf(group) is error
    assert first_leaf(error) is error


def test_health_url_for():
    assert health_url_for("http://localhost:8000/mcp") == "http://localhost:8000/health"
    assert (
        health_url_for("https://mcp.example.com/credential-demo/mcp/")
        == "https://mcp.example.com/credential-demo/health"
    )
    assert health_url_for("https://mcp.example.com/") is None


def test_chat_is_an_explicit_subcommand():
    """`mcp-agent chat <url>` is the documented invocation.

    Typer collapses a group down to its one command when only one is left, at
    which point `chat` parses as the URL and the real URL is an unexpected
    extra argument. ``_root`` exists to prevent that, and this guards it.

    Asserted on the structure rather than by invoking: `--help` short-circuits
    before argument parsing, so a run that passes `--help` cannot tell the two
    shapes apart, and a run without it would start a chat.
    """
    command = get_command(app)
    # Not an isinstance check: TyperGroup does not subclass click.Group on
    # every Typer version. A collapsed app is a plain Command, which carries
    # no sub-command mapping at all.
    assert "chat" in getattr(command, "commands", {}), "the command group collapsed"


def test_a_refused_connection_is_a_connect_failure_however_fastmcp_wraps_it():
    """fastmcp reports a connection that never came up as a bare `RuntimeError`
    with the transport's exception as its cause, so the types an `except` on
    the transport would name never reach the caller. The cause is what names
    the port, so that is what is reported."""
    refused = httpx2.ConnectError("All connection attempts failed")
    wrapped = RuntimeError("Client failed to connect: All connection attempts failed")
    wrapped.__cause__ = refused

    assert connect_failure(wrapped) is refused
    assert connect_failure(ExceptionGroup("build", [wrapped])) is refused
    assert connect_failure(refused) is refused
    assert connect_failure(RuntimeError("something else entirely")) is None
    assert connect_failure(ExceptionGroup("build", [ValueError("no")])) is None


async def test_overlapping_calls_each_carry_their_own_users_credentials(monkeypatch):
    """Two users' calls in flight at once, through the one client an agent holds.

    A fastmcp client is reentrant: a call that starts while another is in
    flight joins that session, whose HTTP client was built — headers and all —
    for whoever connected it. Listed through `list_tools`, each call opens a
    session of its own, so the second user's call cannot go out as the first.
    """
    import asyncio
    import json
    import sys
    import types

    import uvicorn
    from langchain_core.tools import tool

    from mcp_agent.main import list_tools
    from mcp_runtime.credentials import credential_from_header
    from mcp_runtime.server import build_server
    from mcp_runtime.tool_result import ToolResult

    @tool
    async def whoami() -> ToolResult:
        """Report the caller's credential, slowly enough for calls to overlap."""
        seen = credential_from_header("x-demo-token")
        await asyncio.sleep(0.2)
        return {"message": seen}

    module = types.ModuleType("overlap_toolset.tools")
    module.TOOLS = [whoami]
    module.CREDENTIAL_HEADERS = ["x-demo-token"]
    monkeypatch.setitem(sys.modules, "overlap_toolset.tools", module)

    # A real server on a real port: the session reuse under test is the HTTP
    # transport's, and a test client in-process would bypass it.
    app = build_server("overlap-toolset").streamable_http_app()
    uv = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error")
    )
    serving = asyncio.create_task(uv.serve())
    while not uv.started:
        await asyncio.sleep(0.01)
    try:
        port = uv.servers[0].sockets[0].getsockname()[1]
        connections = {"demo": {"url": f"http://127.0.0.1:{port}/mcp"}}
        (client,) = with_credential_support(
            connections, {"demo": ["x-demo-token"]}
        ).values()
        (whoami_tool,) = await list_tools(client)

        def as_user(name: str) -> asyncio.Task:
            with user_credentials({"x-demo-token": name}):
                return asyncio.create_task(whoami_tool.ainvoke({}))

        results = await asyncio.gather(as_user("alice"), as_user("bob"))
    finally:
        uv.should_exit = True
        await serving

    seen = [json.loads(blocks[0]["text"])["message"] for blocks in results]
    assert seen == ["alice", "bob"]
