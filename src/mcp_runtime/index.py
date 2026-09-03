"""Serve a public directory of every deployed toolset.

``mcp-index`` runs as one service alongside the toolsets: it asks the platform
which toolsets are running (see :mod:`mcp_runtime.discovery`), asks each one's
``/health`` route for its tool names, and serves the aggregate as JSON at ``/``
(interactive docs at ``/docs``). Driven by environment variables:

- ``PUBLIC_URL`` (required): external base URL the toolsets are served under,
  e.g. ``https://mcp.example.com``.
- ``HOST`` (default ``127.0.0.1``; the Helm chart sets ``0.0.0.0``).
- ``PORT`` (default ``8000``).
- ``MCP_INDEX_DISCOVERY`` (default ``kubernetes``): where to look for toolsets.
- ``MCP_ECS_CLUSTER``: the cluster to list, required by the ``ecs`` backend.
- ``MCP_TOOLSET_PORT`` (default ``8000``): port to address a toolset on when
  its registration does not name one. ``ecs`` only.
"""

import asyncio
from ipaddress import IPv4Address
from typing import Any, Literal, Self

import httpx
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field, IPvAnyAddress, model_validator
from pydantic_settings import BaseSettings

from mcp_runtime.discovery import (
    DEFAULT_TOOLSET_PORT,
    Discovery,
    EcsDiscovery,
    KubernetesDiscovery,
    ToolsetService,
)

__all__ = [
    "IndexSettings",
    "ToolsetService",
    "build_app",
    "discovery_backend",
    "main",
]


class IndexSettings(BaseSettings):
    """Index configuration, validated from the environment."""

    public_url: str
    host: IPvAnyAddress = IPv4Address("127.0.0.1")
    port: int = Field(default=8000, ge=1, le=65535)
    discovery: Literal["kubernetes", "ecs"] = Field(
        default="kubernetes", validation_alias="MCP_INDEX_DISCOVERY"
    )
    ecs_cluster: str | None = Field(default=None, validation_alias="MCP_ECS_CLUSTER")
    toolset_port: int = Field(
        default=DEFAULT_TOOLSET_PORT,
        ge=1,
        le=65535,
        validation_alias="MCP_TOOLSET_PORT",
    )

    @model_validator(mode="after")
    def _cluster_required_for_ecs(self) -> Self:
        """Fail at startup, not at the first request that finds nothing."""
        if self.discovery == "ecs" and not self.ecs_cluster:
            raise ValueError("MCP_ECS_CLUSTER is required when MCP_INDEX_DISCOVERY=ecs")
        return self


def discovery_backend(settings: IndexSettings) -> Discovery:
    """Build the backend the settings ask for.

    Called at startup so a missing ``[aws]`` extra, an unset region or absent
    credentials surface as the process failing to start, rather than as an
    empty directory served with a 200.
    """
    if settings.discovery == "ecs":
        cluster = settings.ecs_cluster
        if cluster is None:  # unreachable via the environment; the validator caught it
            raise RuntimeError(
                "MCP_ECS_CLUSTER is required when MCP_INDEX_DISCOVERY=ecs"
            )
        backend = EcsDiscovery(cluster, port=settings.toolset_port)
        backend.check()
        return backend
    return KubernetesDiscovery()


class StateDeclarations(BaseModel):
    """What a toolset publishes into session state, and will not author.

    Whether a ``not_authored`` parameter can be satisfied depends on which
    servers a client connects to and what has run, so it is not answerable
    here — only reported.
    """

    #: ``{tool, field, state_key}`` for each ``ToolResult`` data key this
    #: toolset publishes into session state.
    produces: list[dict[str, Any]] = []
    #: ``{tool, parameter}`` for each parameter a model may not write. What a
    #: deployment cannot satisfy on its own: something else has to publish a
    #: value for it first.
    not_authored: list[dict[str, Any]] = []


class ToolsetEntry(BaseModel):
    """One deployed toolset in the directory.

    ``credential_headers`` names the per-user HTTP headers the toolset's
    tools read; clients should send those credentials only to this toolset's
    connection.
    """

    name: str
    url: str
    status: Literal["ok", "unreachable"]
    tools: list[str]
    credential_headers: list[str] = []
    state: StateDeclarations = StateDeclarations()


class Connection(BaseModel):
    """A langchain-mcp-adapters StreamableHttpConnection."""

    transport: Literal["streamable_http"] = "streamable_http"
    url: str


class Index(BaseModel):
    """Directory of every deployed toolset.

    ``connections`` is shaped so an agent can pass it straight to
    ``MultiServerMCPClient``; ``toolsets`` adds per-service status and tool
    names for humans.
    """

    connections: dict[str, Connection]
    toolsets: list[ToolsetEntry]


async def describe(
    client: httpx.AsyncClient, service: ToolsetService, public_url: str
) -> ToolsetEntry:
    """Build one directory entry, asking the toolset's /health for its tools."""
    url = f"{public_url}/{service.toolset}/mcp"
    try:
        response = await client.get(f"{service.base_url}/health")
        response.raise_for_status()
        health = response.json()
    except httpx.HTTPError:
        return ToolsetEntry(
            name=service.toolset, url=url, status="unreachable", tools=[]
        )
    return ToolsetEntry(
        name=service.toolset,
        url=url,
        status="ok",
        tools=health.get("tools", []),
        credential_headers=health.get("credential_headers", []),
        state=StateDeclarations(**(health.get("state") or {})),
    )


def build_app(public_url: str, discovery: Discovery | None = None) -> FastAPI:
    """Build the index app: a toolset directory at ``/``, health at ``/health``."""
    base_url = public_url.rstrip("/")
    backend = discovery if discovery is not None else KubernetesDiscovery()
    app = FastAPI(
        title="mcp-toolsets",
        description="Directory of every deployed MCP toolset service.",
    )

    @app.get("/")
    async def index() -> Index:
        services = await backend.services()
        async with httpx.AsyncClient(timeout=5.0) as client:
            entries = await asyncio.gather(
                *(describe(client, service, base_url) for service in services)
            )
        return Index(
            connections={entry.name: Connection(url=entry.url) for entry in entries},
            toolsets=list(entries),
        )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


def main() -> None:
    """Console entry point (``mcp-index``)."""
    settings = IndexSettings()
    app = build_app(settings.public_url, discovery_backend(settings))
    uvicorn.run(app, host=str(settings.host), port=settings.port)
