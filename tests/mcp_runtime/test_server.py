import sys
import types

import pytest
from langchain_core.tools import tool
from pydantic import ValidationError
from starlette.testclient import TestClient

from mcp_runtime.server import (
    RuntimeSettings,
    build_server,
    credential_instructions,
    load_credential_headers,
    load_tools,
    normalise_path_prefix,
    toolset_module_name,
)
from mcp_runtime.tool_result import ToolResult


@tool
def echo(text: str) -> ToolResult:
    """Echo the text back."""
    return ToolResult(message=text)


@tool
def bare_echo(text: str) -> str:
    """Echo the text back without following the ToolResult contract."""
    return text


def tools_module(monkeypatch, name: str, **attrs) -> str:
    """Register a synthetic tools module, so tests don't depend on which
    real toolsets exist in the repo."""
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return name


def test_toolset_module_name():
    assert toolset_module_name("dataset-search") == "dataset_search.tools"
    assert toolset_module_name("hello-world") == "hello_world.tools"


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("TOOLSET", "dataset-search")
    monkeypatch.setenv("HOST", "0.0.0.0")
    monkeypatch.setenv("PORT", "9000")
    settings = RuntimeSettings()
    assert settings.toolset == "dataset-search"
    assert settings.toolset_module is None
    assert str(settings.host) == "0.0.0.0"
    assert settings.port == 9000


def test_settings_require_toolset(monkeypatch):
    monkeypatch.delenv("TOOLSET", raising=False)
    with pytest.raises(ValidationError):
        RuntimeSettings()


def test_settings_reject_bad_values(monkeypatch):
    monkeypatch.setenv("TOOLSET", "dataset-search")
    monkeypatch.setenv("HOST", "not-an-ip")
    with pytest.raises(ValidationError):
        RuntimeSettings()
    monkeypatch.setenv("HOST", "127.0.0.1")
    monkeypatch.setenv("PORT", "70000")
    with pytest.raises(ValidationError):
        RuntimeSettings()


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("", ""),
        ("   ", ""),
        ("/", ""),
        ("/hello", "/hello"),
        ("hello", "/hello"),
        ("/hello/", "/hello"),
        ("/mcp/hello", "/mcp/hello"),
    ],
)
def test_normalise_path_prefix(given, expected):
    assert normalise_path_prefix(given) == expected


@pytest.mark.parametrize("given", ["/hel lo", "/hello?x=1", "//hello", "/hello#frag"])
def test_normalise_path_prefix_rejects_a_non_path(given):
    """A prefix is about to be concatenated with /mcp: one that is quietly
    wrong serves a working endpoint at an address nobody routes to."""
    with pytest.raises(ValueError, match="path segments"):
        normalise_path_prefix(given)


def test_settings_read_the_prefix(monkeypatch):
    monkeypatch.setenv("TOOLSET", "hello")
    monkeypatch.setenv("MCP_PATH_PREFIX", "hello/")
    assert RuntimeSettings().path_prefix == "/hello"


def test_settings_default_to_no_prefix(monkeypatch):
    monkeypatch.setenv("TOOLSET", "hello")
    monkeypatch.delenv("MCP_PATH_PREFIX", raising=False)
    assert RuntimeSettings().path_prefix == ""


def test_settings_reject_a_bad_prefix(monkeypatch):
    monkeypatch.setenv("TOOLSET", "hello")
    monkeypatch.setenv("MCP_PATH_PREFIX", "/two words")
    with pytest.raises(ValidationError):
        RuntimeSettings()


def routes(server) -> list[str]:
    app = server.streamable_http_app()
    return sorted(route.path for route in app.routes if hasattr(route, "path"))


async def test_build_server_serves_at_the_root_by_default(monkeypatch):
    tools_module(monkeypatch, "rooted_toolset.tools", TOOLS=[echo])
    assert routes(build_server("rooted-toolset")) == ["/health", "/mcp"]


async def test_build_server_moves_the_endpoint_under_a_prefix(monkeypatch):
    """The endpoint moves; it is not served at both. A load balancer forwards
    one path, and two answers would be two things to keep in step."""
    tools_module(monkeypatch, "prefixed_toolset.tools", TOOLS=[echo])
    server = build_server("prefixed-toolset", path_prefix="/prefixed-toolset")
    assert routes(server) == [
        "/health",
        "/prefixed-toolset/health",
        "/prefixed-toolset/mcp",
    ]


async def test_build_server_keeps_health_at_the_root_as_well(monkeypatch):
    """Both answer, because they have different callers: a health check and
    the index reach the container below the routing that added the prefix."""
    tools_module(monkeypatch, "health_toolset.tools", TOOLS=[echo])
    server = build_server("health-toolset", path_prefix="health-toolset")
    with TestClient(server.streamable_http_app()) as client:
        for path in ("/health", "/health-toolset/health"):
            response = client.get(path)
            assert response.status_code == 200
            assert response.json()["tools"] == ["echo"]


async def test_load_tools_missing_module():
    with pytest.raises(ModuleNotFoundError):
        load_tools("no_such_toolset.tools")


def test_load_tools_missing_export():
    with pytest.raises(RuntimeError, match="non-empty TOOLS"):
        load_tools("mcp_runtime.server")


def test_load_credential_headers(monkeypatch):
    declaring = tools_module(
        monkeypatch, "declaring_tools", TOOLS=[echo], CREDENTIAL_HEADERS=["X-Fake"]
    )
    bare = tools_module(monkeypatch, "bare_tools", TOOLS=[echo])
    assert load_credential_headers(declaring) == ["x-fake"]
    assert load_credential_headers(bare) == []


def test_load_credential_headers_rejects_bad_export(monkeypatch):
    bad = tools_module(monkeypatch, "bad_tools", CREDENTIAL_HEADERS="x-fake")
    with pytest.raises(RuntimeError, match="CREDENTIAL_HEADERS"):
        load_credential_headers(bad)


def test_credential_instructions():
    assert credential_instructions([]) is None
    text = credential_instructions(["x-cds-token"])
    assert text is not None
    # Names the header and steers the model away from asking for it.
    assert "``x-cds-token``" in text
    assert "do not ask the user" in text


async def test_build_server_sets_credential_instructions(monkeypatch):
    tools_module(
        monkeypatch, "creds_toolset.tools", TOOLS=[echo], CREDENTIAL_HEADERS=["X-Key"]
    )
    server = build_server("creds-toolset")
    assert server.instructions is not None
    assert "``x-key``" in server.instructions


async def test_build_server_no_instructions_without_credentials(monkeypatch):
    tools_module(monkeypatch, "plain_toolset.tools", TOOLS=[echo])
    server = build_server("plain-toolset")
    assert server.instructions is None


async def test_build_server_derives_module_from_toolset_name(monkeypatch):
    tools_module(monkeypatch, "fake_toolset.tools", TOOLS=[echo])
    server = build_server("fake-toolset")
    tools = await server.list_tools()
    assert {t.name for t in tools} == {"echo"}
    for t in tools:
        assert t.description


async def test_build_server_module_override(monkeypatch):
    name = tools_module(monkeypatch, "custom_tools_module", TOOLS=[echo])
    server = build_server("anything", module_name=name)
    tools = await server.list_tools()
    assert {tool.name for tool in tools} == {"echo"}


async def test_build_server_advertises_output_schema(monkeypatch):
    tools_module(monkeypatch, "schema_toolset.tools", TOOLS=[echo])
    server = build_server("schema-toolset")
    (listed,) = await server.list_tools()
    assert listed.outputSchema is not None
    assert "message" in listed.outputSchema["required"]


def test_build_server_rejects_non_contract_tool(monkeypatch):
    tools_module(monkeypatch, "loose_toolset.tools", TOOLS=[bare_echo])
    with pytest.raises(RuntimeError, match="bare_echo"):
        build_server("loose-toolset")
