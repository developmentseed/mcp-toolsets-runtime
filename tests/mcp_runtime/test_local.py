import sys
import types

import pytest
from fastapi.testclient import TestClient
from langchain_core.tools import tool

from mcp_runtime.local import (
    LocalSettings,
    build_local_app,
    discover_toolsets,
)
from mcp_runtime.tool_result import ToolResult


@tool
def echo(text: str) -> ToolResult:
    """Echo the text back."""
    return ToolResult(message=text)


@tool
def whoami() -> ToolResult:
    """Report the caller's identity."""
    return ToolResult(message="nobody")


def tools_module(monkeypatch, name: str, **attrs) -> str:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return name


def register_toolsets(monkeypatch) -> None:
    tools_module(monkeypatch, "alpha.tools", TOOLS=[echo])
    tools_module(
        monkeypatch, "beta.tools", TOOLS=[whoami], CREDENTIAL_HEADERS=["X-Demo-Token"]
    )


def make_toolsets_dir(root, names: list[str]):
    """A conventional toolsets/ tree: one directory per toolset, each packaged."""
    toolsets_dir = root / "toolsets"
    for name in names:
        package = toolsets_dir / name
        package.mkdir(parents=True)
        (package / "pyproject.toml").write_text("[project]\n")
    return toolsets_dir


def test_settings_splits_csv(monkeypatch):
    monkeypatch.setenv("TOOLSETS", "alpha, beta ,, ")
    assert LocalSettings().toolsets == ["alpha", "beta"]


def test_settings_default_to_empty_list(monkeypatch):
    monkeypatch.delenv("TOOLSETS", raising=False)
    assert LocalSettings().toolsets == []


def test_discover_toolsets_lists_directories_with_a_pyproject(tmp_path):
    toolsets_dir = make_toolsets_dir(tmp_path, ["beta", "alpha"])
    # Not a real toolset: no pyproject.toml (e.g. stale __pycache__ leftovers).
    (toolsets_dir / "not-a-toolset").mkdir()

    assert discover_toolsets(toolsets_dir) == ["alpha", "beta"]


def test_discover_toolsets_resolves_relative_to_the_working_directory(
    tmp_path, monkeypatch
):
    """The default is a bare ``toolsets``, so running from a repo root finds it.

    The package is installed into a venv rather than living in the consumer's
    tree, so the path can only come from where the process runs.
    """
    make_toolsets_dir(tmp_path, ["alpha"])
    monkeypatch.chdir(tmp_path)

    assert discover_toolsets() == ["alpha"]


def test_discover_toolsets_requires_a_toolsets_directory(tmp_path):
    with pytest.raises(RuntimeError, match="does not exist"):
        discover_toolsets(tmp_path / "nope")


def test_build_local_app_rejects_empty():
    with pytest.raises(RuntimeError, match="at least one"):
        build_local_app([], base_url="http://localhost:8000")


def test_build_local_app_rejects_duplicates(monkeypatch):
    register_toolsets(monkeypatch)
    with pytest.raises(RuntimeError, match="duplicate"):
        build_local_app(["alpha", "alpha"], base_url="http://localhost:8000")


def test_index_lists_every_mounted_toolset(monkeypatch):
    register_toolsets(monkeypatch)
    app = build_local_app(["alpha", "beta"], base_url="http://localhost:8000/")

    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {
            "connections": {
                "alpha": {
                    "transport": "streamable_http",
                    "url": "http://localhost:8000/alpha/mcp",
                },
                "beta": {
                    "transport": "streamable_http",
                    "url": "http://localhost:8000/beta/mcp",
                },
            },
            "toolsets": [
                {
                    "name": "alpha",
                    "url": "http://localhost:8000/alpha/mcp",
                    "status": "ok",
                    "tools": ["echo"],
                    "credential_headers": [],
                },
                {
                    "name": "beta",
                    "url": "http://localhost:8000/beta/mcp",
                    "status": "ok",
                    "tools": ["whoami"],
                    "credential_headers": ["x-demo-token"],
                },
            ],
        }


def test_health_route(monkeypatch):
    register_toolsets(monkeypatch)
    app = build_local_app(["alpha"], base_url="http://localhost:8000")
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


def test_mounted_toolsets_expose_their_own_health(monkeypatch):
    """``/<toolset>/health`` matches the path production's ingress serves."""
    register_toolsets(monkeypatch)
    app = build_local_app(["alpha", "beta"], base_url="http://localhost:8000")
    with TestClient(app) as client:
        alpha = client.get("/alpha/health")
        beta = client.get("/beta/health")
        assert alpha.status_code == 200
        assert alpha.json()["tools"] == ["echo"]
        assert beta.status_code == 200
        assert beta.json()["tools"] == ["whoami"]


def test_mounted_toolsets_serve_mcp_traffic(monkeypatch):
    """The mounted sub-apps' session managers must actually be running —
    proof that entering their lifespans by hand (since a Mount never
    receives the ASGI "lifespan" scope) worked."""
    register_toolsets(monkeypatch)
    app = build_local_app(["alpha", "beta"], base_url="http://localhost:8000")
    with TestClient(app, base_url="http://localhost:8000") as client:
        response = client.post(
            "/alpha/mcp",
            headers={"Accept": "application/json, text/event-stream"},
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "0"},
                },
            },
        )
        assert response.status_code == 200
