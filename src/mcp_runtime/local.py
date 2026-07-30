"""Serve several toolsets together in one local process.

In production every toolset is its own pod/Service, and ``mcp-index`` stitches
them into one directory by discovering Services through the Kubernetes API
(see :mod:`mcp_runtime.index`) behind a shared ingress. Locally there is
neither a cluster nor an ingress, so ``mcp-serve`` — which serves exactly one
toolset per process — cannot present that same "one URL for everything" shape
on its own.

``mcp-serve-local`` recreates it without Kubernetes: it builds each named
toolset's MCP server in this one process and mounts it at ``/<toolset>``
(so ``/<toolset>/mcp`` and ``/<toolset>/health`` match production's ingress
paths exactly), and serves the same directory shape at ``/`` that
``mcp-index`` serves in production. ``mcp-agent``, ``mcp-cli`` and
``MultiServerMCPClient`` all consume that shape already, so they work
unchanged against ``http://localhost:8000/``.

Driven by environment variables:

- ``TOOLSETS`` (optional): comma-separated toolset names, e.g.
  ``hello,stac-explorer``. Defaults to every toolset directory under
  ``TOOLSETS_DIR`` (see :func:`discover_toolsets`).
- ``TOOLSETS_DIR`` (default ``toolsets``): where to look when ``TOOLSETS`` is
  unset, relative to the working directory you run from.
- ``HOST`` (default ``127.0.0.1``).
- ``PORT`` (default ``8000``).

Names are resolved the same way ``mcp-serve`` resolves ``TOOLSET`` — by
importing ``<name>.tools`` — so every toolset must be installed in the
environment you run from. ``TOOLSETS_DIR`` only supplies the *names*; it is a
convenience for the conventional repo layout, not a second discovery
mechanism. Name them explicitly with ``TOOLSETS`` and the directory is never
read at all.
"""

import contextlib
from collections.abc import AsyncIterator
from ipaddress import IPv4Address
from pathlib import Path
from typing import Annotated

import uvicorn
from fastapi import FastAPI
from mcp.server.fastmcp import FastMCP
from pydantic import Field, IPvAnyAddress, field_validator
from pydantic_settings import BaseSettings, NoDecode

from mcp_runtime.index import Connection, Index, ToolsetEntry
from mcp_runtime.server import (
    build_server,
    load_credential_headers,
    load_tools,
    toolset_module_name,
)

DEFAULT_TOOLSETS_DIR = Path("toolsets")


def discover_toolsets(toolsets_dir: Path = DEFAULT_TOOLSETS_DIR) -> list[str]:
    """Every toolset directory under ``toolsets_dir``, sorted by name.

    Used when ``TOOLSETS`` is unset. One directory per toolset, each with its
    own ``pyproject.toml`` — the layout ``mcp-toolset new`` produces. The path
    is resolved against the working directory, so this is for running from a
    consumer repo's root; anywhere else, name the toolsets with ``TOOLSETS``.
    """
    resolved = Path(toolsets_dir).resolve()
    if not resolved.is_dir():
        raise RuntimeError(
            f"can't auto-discover toolsets: {resolved} does not exist "
            "(run from your repo root, set TOOLSETS_DIR, or name them in TOOLSETS)"
        )
    return sorted(
        entry.name
        for entry in resolved.iterdir()
        if entry.is_dir() and (entry / "pyproject.toml").is_file()
    )


class LocalSettings(BaseSettings):
    """Local combined-server configuration, validated from the environment."""

    toolsets: Annotated[list[str], NoDecode] = Field(default_factory=list)
    toolsets_dir: Path = DEFAULT_TOOLSETS_DIR
    host: IPvAnyAddress = IPv4Address("127.0.0.1")
    port: int = Field(default=8000, ge=1, le=65535)

    @field_validator("toolsets", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        if isinstance(value, str):
            return [name.strip() for name in value.split(",") if name.strip()]
        return value


def build_local_app(toolsets: list[str], base_url: str) -> FastAPI:
    """Mount each toolset's MCP server at ``/<toolset>``, plus an index at ``/``.

    Mirrors ``mcp_runtime.index.build_app``'s ``Index``/``ToolsetEntry`` shape,
    but built directly from the in-process servers instead of over HTTP from
    Kubernetes-discovered Services, so it needs no cluster. Every entry is
    ``status="ok"``: a toolset that failed to import raised out of
    :func:`build_server` before it could be listed.
    """
    if not toolsets:
        raise RuntimeError("TOOLSETS must name at least one toolset")
    if len(toolsets) != len(set(toolsets)):
        raise RuntimeError(f"TOOLSETS has duplicate entries: {toolsets}")

    base_url = base_url.rstrip("/")
    servers: list[tuple[str, FastMCP]] = []
    connections: dict[str, Connection] = {}
    entries: list[ToolsetEntry] = []

    for name in toolsets:
        module_name = toolset_module_name(name)
        tools = load_tools(module_name)
        credential_headers = load_credential_headers(module_name)
        server = build_server(name, module_name)
        servers.append((name, server))

        url = f"{base_url}/{name}/mcp"
        connections[name] = Connection(url=url)
        entries.append(
            ToolsetEntry(
                name=name,
                url=url,
                status="ok",
                tools=[tool.name for tool in tools],
                credential_headers=credential_headers,
            )
        )

    # Each server's own Starlette app runs its session manager from its own
    # lifespan (see FastMCP.streamable_http_app), but mounting it as a
    # sub-application means the ASGI "lifespan" scope never reaches it — only
    # the outermost app receives it. Enter every server's session manager
    # here instead, so mounting several together still starts them all.
    sub_apps = [(name, server.streamable_http_app()) for name, server in servers]

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with contextlib.AsyncExitStack() as stack:
            for _, server in servers:
                await stack.enter_async_context(server.session_manager.run())
            yield

    app = FastAPI(
        title="mcp-toolsets (local)",
        description="Local combined server: every requested toolset, mounted together.",
        lifespan=lifespan,
    )
    for name, sub_app in sub_apps:
        app.mount(f"/{name}", sub_app)

    index = Index(connections=connections, toolsets=entries)

    @app.get("/")
    async def root() -> Index:
        return index

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


def main() -> None:
    """Console entry point (``mcp-serve-local``)."""
    settings = LocalSettings()
    toolsets = settings.toolsets or discover_toolsets(settings.toolsets_dir)
    app = build_local_app(toolsets, base_url=f"http://{settings.host}:{settings.port}")
    uvicorn.run(app, host=str(settings.host), port=settings.port)
