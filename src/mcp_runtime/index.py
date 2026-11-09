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
- ``MCP_ECS_TOOLSET_PORT`` (default ``8000``): port to address a toolset on
  when its registration does not name one.
- ``MCP_GATE_*``: sign MCP clients in and gate every toolset behind this
  index (see :mod:`mcp_runtime.gate`). Unset, the toolsets are routed to
  directly and the index only lists them.
"""

import asyncio
from ipaddress import IPv4Address
from typing import TYPE_CHECKING, Any, Literal, Self

import httpx
import uvicorn
from fastapi import FastAPI
from pydantic import (
    BaseModel,
    Field,
    IPvAnyAddress,
    TypeAdapter,
    ValidationError,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from mcp_runtime.discovery import (
    DEFAULT_TOOLSET_PORT,
    Discovery,
    EcsDiscovery,
    KubernetesDiscovery,
    ToolsetService,
)

if TYPE_CHECKING:
    # Imported at use instead: the gate needs the [gate] extra, and an index
    # that does not gate must not import it.
    from mcp_runtime.gate import Credentials, GateSettings

__all__ = [
    "IndexSettings",
    "ToolsetService",
    "build_app",
    "discovery_backend",
    "main",
]


class IndexSettings(BaseSettings):
    """Index configuration, validated from the environment.

    The discovery fields carry explicit aliases so their variables read as a
    family (``MCP_INDEX_DISCOVERY``, ``MCP_ECS_*``) rather than as bare words;
    ``validate_by_name`` keeps the fields constructible by name too, for anyone
    building the index into an app of their own rather than from a shell.
    """

    model_config = SettingsConfigDict(validate_by_name=True)

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
        validation_alias="MCP_ECS_TOOLSET_PORT",
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


class ViewDeclarations(BaseModel):
    """What a toolset declares about one of its ``ui://`` views.

    ``csp`` is the view's ``VIEW_CSP`` entry, as served in its
    ``_meta.ui.csp``: the origins (and ``blob:``/``data:``) a host lets it
    reach. ``None`` means it declared none, and runs under the spec's
    restrictive default. Listed for review, not enforced: the host enforces.
    """

    csp: dict[str, list[str]] | None = None


#: Parses a ``/health``'s ``views``; see :func:`read_views`.
_VIEWS = TypeAdapter(dict[str, ViewDeclarations])


def read_views(views: Any) -> dict[str, ViewDeclarations]:
    """A toolset's ``/health`` ``views``, or ``{}`` if this index can't read it.

    A toolset on a newer runtime may list more about a view than this index has
    a model for, and one on an older runtime lists nothing. Either way the rest
    of its entry still stands, and the other toolsets' entries with it: the
    index answers for every toolset at once, so one entry that fails to parse
    must not turn ``GET /`` into a 500.
    """
    try:
        return _VIEWS.validate_python(views or {})
    except ValidationError:
        return {}


#: Parses a ``/health``'s ``visibility``; see :func:`read_visibility`.
_VISIBILITY = TypeAdapter(dict[str, list[str]])


def read_visibility(visibility: Any) -> dict[str, list[str]]:
    """A toolset's ``/health`` ``visibility``, or ``{}`` if this index can't read it.

    As :func:`read_views`: one entry that fails to parse must not take the
    other toolsets' entries down with it.
    """
    try:
        return _VISIBILITY.validate_python(visibility or {})
    except ValidationError:
        return {}


class ToolsetEntry(BaseModel):
    """One deployed toolset in the directory.

    ``credential_headers`` names the per-user HTTP headers the toolset's
    tools read; clients should send those credentials only to this toolset's
    connection. ``views`` maps each view id the toolset serves to what it
    declares about it; a toolset on a runtime older than the field lists none.
    ``visibility`` maps each tool its ``VISIBILITY`` names to who it is for
    (``"model"``, ``"app"``); a tool it leaves out is for both.
    """

    name: str
    url: str
    status: Literal["ok", "unreachable"]
    tools: list[str]
    credential_headers: list[str] = []
    state: StateDeclarations = StateDeclarations()
    views: dict[str, ViewDeclarations] = {}
    visibility: dict[str, list[str]] = {}


class Connection(BaseModel):
    """How to reach one toolset: a streamable HTTP endpoint."""

    transport: Literal["streamable_http"] = "streamable_http"
    url: str


class Index(BaseModel):
    """Directory of every deployed toolset.

    ``connections`` is shaped so an agent can pass it straight to
    an MCP client; ``toolsets`` adds per-service status and tool
    names for humans.
    """

    connections: dict[str, Connection]
    toolsets: list[ToolsetEntry]


async def describe(
    client: httpx.AsyncClient, service: ToolsetService, public_url: str | None
) -> ToolsetEntry:
    """Build one directory entry, asking the toolset's /health for its tools.

    ``public_url`` ``None`` lists the toolset at its own address instead, for
    a caller inside the platform (see ``build_app``).
    """
    url = (
        f"{public_url}/{service.toolset}/mcp"
        if public_url is not None
        else f"{service.base_url}/mcp"
    )
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
        views=read_views(health.get("views")),
        visibility=read_visibility(health.get("visibility")),
    )


def build_app(
    public_url: str,
    discovery: Discovery | None = None,
    gate: "GateSettings | None" = None,
    credentials: "Credentials | None" = None,
) -> FastAPI:
    """Build the index app: a toolset directory at ``/``, health at ``/health``.

    ``/?network=internal`` lists each toolset at its own address rather than
    under ``public_url``: for an agent running beside the toolsets, which
    reaches them directly and so never meets the gate.

    ``gate`` mounts :mod:`mcp_runtime.gate` on the same app, so the public
    ``/<toolset>/mcp`` routes come here and are signed in; ``credentials``
    adds a toolset's per-user headers to the calls it forwards.
    """
    base_url = public_url.rstrip("/")
    backend = discovery if discovery is not None else KubernetesDiscovery()
    app = FastAPI(
        title="mcp-toolsets",
        description="Directory of every deployed MCP toolset service.",
    )

    @app.get("/")
    async def index(network: Literal["public", "internal"] = "public") -> Index:
        services = await backend.services()
        listed_under = base_url if network == "public" else None
        async with httpx.AsyncClient(timeout=5.0) as client:
            entries = await asyncio.gather(
                *(describe(client, service, listed_under) for service in services)
            )
        return Index(
            connections={entry.name: Connection(url=entry.url) for entry in entries},
            toolsets=list(entries),
        )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    if gate is not None:
        from mcp_runtime.gate import gate_routes

        app.router.routes.extend(gate_routes(gate, base_url, backend, credentials))

    return app


def gate_settings() -> "GateSettings | None":
    """The gate's settings when the environment asks for one, else ``None``."""
    from mcp_runtime.gate import GateSettings, configured

    return GateSettings() if configured() else None


def main() -> None:
    """Console entry point (``mcp-index``)."""
    settings = IndexSettings()
    app = build_app(
        settings.public_url, discovery_backend(settings), gate=gate_settings()
    )
    uvicorn.run(app, host=str(settings.host), port=settings.port)
