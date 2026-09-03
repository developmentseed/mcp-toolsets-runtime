import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import mcp_runtime.index
from mcp_runtime.discovery import EcsDiscovery, KubernetesDiscovery, ToolsetService
from mcp_runtime.index import IndexSettings, build_app, discovery_backend


class FakeDiscovery:
    """A backend that already knows the answer."""

    def __init__(self, *services: ToolsetService):
        self._services = list(services)

    async def services(self) -> list[ToolsetService]:
        return self._services


def test_settings_require_public_url(monkeypatch):
    monkeypatch.delenv("PUBLIC_URL", raising=False)
    with pytest.raises(ValidationError):
        IndexSettings()


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("PUBLIC_URL", "https://mcp.example.com")
    monkeypatch.setenv("HOST", "0.0.0.0")
    monkeypatch.setenv("PORT", "9000")
    settings = IndexSettings()
    assert settings.public_url == "https://mcp.example.com"
    assert str(settings.host) == "0.0.0.0"
    assert settings.port == 9000


def test_discovery_defaults_to_kubernetes(monkeypatch):
    monkeypatch.setenv("PUBLIC_URL", "https://mcp.example.com")
    monkeypatch.delenv("MCP_INDEX_DISCOVERY", raising=False)
    settings = IndexSettings()
    assert settings.discovery == "kubernetes"
    assert isinstance(discovery_backend(settings), KubernetesDiscovery)


def test_ecs_discovery_from_env(monkeypatch):
    monkeypatch.setenv("PUBLIC_URL", "https://mcp.example.com")
    monkeypatch.setenv("MCP_INDEX_DISCOVERY", "ecs")
    monkeypatch.setenv("MCP_ECS_CLUSTER", "mcp-toolsets")
    monkeypatch.setenv("MCP_TOOLSET_PORT", "8080")
    monkeypatch.setattr(
        mcp_runtime.index.EcsDiscovery, "check", lambda self: None, raising=True
    )
    settings = IndexSettings()
    assert settings.ecs_cluster == "mcp-toolsets"
    assert settings.toolset_port == 8080
    assert isinstance(discovery_backend(settings), EcsDiscovery)


def test_ecs_discovery_requires_a_cluster(monkeypatch):
    """A misconfigured index fails to start rather than serving an empty directory."""
    monkeypatch.setenv("PUBLIC_URL", "https://mcp.example.com")
    monkeypatch.setenv("MCP_INDEX_DISCOVERY", "ecs")
    monkeypatch.delenv("MCP_ECS_CLUSTER", raising=False)
    with pytest.raises(ValidationError, match="MCP_ECS_CLUSTER"):
        IndexSettings()


def test_an_unknown_discovery_backend_is_refused(monkeypatch):
    monkeypatch.setenv("PUBLIC_URL", "https://mcp.example.com")
    monkeypatch.setenv("MCP_INDEX_DISCOVERY", "nomad")
    with pytest.raises(ValidationError):
        IndexSettings()


def test_health_route():
    client = TestClient(build_app("https://mcp.example.com"))
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_index_route(monkeypatch):
    async def fake_describe(client, service, public_url):
        return mcp_runtime.index.ToolsetEntry(
            name=service.toolset,
            url=f"{public_url}/{service.toolset}/mcp",
            status="ok",
            tools=["search_datasets"],
            credential_headers=["x-demo-token"],
        )

    monkeypatch.setattr(mcp_runtime.index, "describe", fake_describe)

    discovery = FakeDiscovery(
        ToolsetService("dataset-search", "http://mcp-dataset-search:8000")
    )
    client = TestClient(build_app("https://mcp.example.com/", discovery))
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "connections": {
            "dataset-search": {
                "transport": "streamable_http",
                "url": "https://mcp.example.com/dataset-search/mcp",
            }
        },
        "toolsets": [
            {
                "name": "dataset-search",
                "url": "https://mcp.example.com/dataset-search/mcp",
                "status": "ok",
                "tools": ["search_datasets"],
                "credential_headers": ["x-demo-token"],
                "state": {"produces": [], "not_authored": []},
            }
        ],
    }
