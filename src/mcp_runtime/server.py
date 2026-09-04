"""Discover a toolset's LangChain tools and serve them over MCP.

The runtime is driven by environment variables so the same image works for
every toolset:

- ``TOOLSET`` (required): the toolset directory name, e.g. ``dataset-search``.
- ``TOOLSET_MODULE`` (optional): module exporting ``TOOLS``; defaults to the
  convention ``<toolset>.tools`` (``dataset-search`` -> ``dataset_search.tools``).
- ``HOST`` (default ``127.0.0.1``; the Helm chart sets ``0.0.0.0``).
- ``PORT`` (default ``8000``).
- ``MCP_PATH_PREFIX`` (optional): serve under a path rather than at the root,
  e.g. ``/hello`` for ``/hello/mcp`` and ``/hello/health``.

Why a prefix, when an ingress can rewrite one away: not every proxy can. A
Kubernetes Ingress strips ``/<toolset>`` before the request arrives, so the
container only ever sees ``/mcp``. An AWS Application Load Balancer forwards
the path it received and offers no rewrite on a forward action, so serving
many toolsets under one domain there means each one answering on its own path.
Unset — the default — nothing changes.
"""

import importlib
import re
from ipaddress import IPv4Address

from langchain_core.tools import BaseTool
from mcp.server.fastmcp import FastMCP
from pydantic import Field, IPvAnyAddress, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from mcp_runtime.fastmcp_output import to_fastmcp
from mcp_runtime.declarations import state_declarations, with_state_meta
from mcp_runtime.views import load_views, register_views, with_view_meta


#: A prefix is one or more ``/``-separated segments of the characters a
#: toolset name and a URL path comfortably share.
PATH_PREFIX_PATTERN = re.compile(r"^(/[A-Za-z0-9._~-]+)+$")


def normalise_path_prefix(prefix: str) -> str:
    """Accept ``hello``, ``/hello`` or ``/hello/`` and return ``/hello``.

    Empty means "serve at the root", which is what every deployment did before
    this existed. Anything else must be a path, because it is about to be
    concatenated with ``/mcp``: a prefix that is quietly wrong would serve a
    working endpoint at an address nobody is routing to.
    """
    prefix = prefix.strip().rstrip("/")
    if not prefix:
        return ""
    if not prefix.startswith("/"):
        prefix = f"/{prefix}"
    if not PATH_PREFIX_PATTERN.match(prefix):
        raise ValueError(
            f"path prefix must be one or more path segments, got {prefix!r}"
        )
    return prefix


class RuntimeSettings(BaseSettings):
    """Runtime configuration, validated from the environment."""

    model_config = SettingsConfigDict(validate_by_name=True)

    toolset: str
    toolset_module: str | None = None
    host: IPvAnyAddress = IPv4Address("127.0.0.1")
    port: int = Field(default=8000, ge=1, le=65535)
    path_prefix: str = Field(default="", validation_alias="MCP_PATH_PREFIX")

    @field_validator("path_prefix")
    @classmethod
    def _normalise_prefix(cls, prefix: str) -> str:
        return normalise_path_prefix(prefix)


def toolset_module_name(toolset: str) -> str:
    """Derive the tools module from a toolset directory name by convention."""
    return toolset.replace("-", "_") + ".tools"


def load_tools(module_name: str) -> list[BaseTool]:
    """Import a tools module and return its ``TOOLS`` export."""
    module = importlib.import_module(module_name)
    tools = getattr(module, "TOOLS", None)
    if not tools:
        raise RuntimeError(f"module {module_name!r} must export a non-empty TOOLS list")
    for tool in tools:
        if not isinstance(tool, BaseTool):
            raise RuntimeError(
                f"{module_name}.TOOLS entries must be LangChain BaseTool "
                f"instances, got {type(tool).__name__}"
            )
    return list(tools)


def load_credential_headers(module_name: str) -> list[str]:
    """Return a tools module's optional ``CREDENTIAL_HEADERS`` export.

    Names of per-user HTTP headers the toolset's tools read (via
    ``mcp_runtime.credentials``). Advertised in ``/health`` and the index so
    clients attach each credential only to the toolsets that declare it.
    """
    module = importlib.import_module(module_name)
    headers = getattr(module, "CREDENTIAL_HEADERS", [])
    if not isinstance(headers, list) or not all(
        isinstance(header, str) and header for header in headers
    ):
        raise RuntimeError(
            f"{module_name}.CREDENTIAL_HEADERS must be a list of header names"
        )
    return sorted(header.lower() for header in headers)


def credential_instructions(credential_headers: list[str]) -> str | None:
    """Server ``instructions`` telling the model how this server's auth works.

    Credentials ride the transport as HTTP headers (see
    ``mcp_runtime.credentials``) and are never visible to the model. Without a
    hint, a model tends to preemptively refuse a credential-bearing tool or ask
    the user for a key it will never see. Deriving this once from the toolset's
    declared ``CREDENTIAL_HEADERS`` keeps the guidance in a single place —
    server-level, returned in the ``initialize`` result so every host gets it —
    instead of repeating it in each tool's docstring. Returns ``None`` when the
    toolset declares no credential headers (no instructions to add).
    """
    if not credential_headers:
        return None
    names = ", ".join(f"``{header}``" for header in credential_headers)
    return (
        f"Some tools here authenticate the caller via HTTP headers ({names}) "
        "supplied by the client transport, not by you. You never see these "
        "credentials and cannot set them, so do not ask the user for them and do "
        "not refuse to call a tool for lack of a key — just call the tool. If a "
        "credential is genuinely missing, the tool reports that itself."
    )


def build_server(
    toolset: str,
    module_name: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    path_prefix: str = "",
) -> FastMCP:
    """Build a stateless FastMCP server exposing the toolset's TOOLS.

    ``path_prefix`` moves the MCP endpoint under a path (``/hello/mcp``) for a
    proxy that cannot rewrite one away. ``/health`` is served at the prefix
    *and* at the root, because they have different callers: a load balancer's
    health check and the index reach the container directly, below whatever
    routing put the prefix there, while the prefixed one is what the published
    URL shape promises.
    """
    prefix = normalise_path_prefix(path_prefix)
    module_name = module_name or toolset_module_name(toolset)
    tools = load_tools(module_name)
    credential_headers = load_credential_headers(module_name)
    views = load_views(module_name)

    fastmcp_tools = with_view_meta(
        toolset, module_name, [to_fastmcp(tool) for tool in tools], views
    )
    fastmcp_tools = with_state_meta(toolset, tools, fastmcp_tools)

    server = FastMCP(
        name=f"mcp-{toolset}",
        instructions=credential_instructions(credential_headers),
        tools=fastmcp_tools,
        host=host,
        port=port,
        streamable_http_path=f"{prefix}/mcp",
        stateless_http=True,
    )
    register_views(server, toolset, module_name, views)

    tool_names = [tool.name for tool in tools]
    state = state_declarations(toolset, tools)

    async def health(request: Request) -> Response:
        return JSONResponse(
            {
                "status": "ok",
                "tools": tool_names,
                "credential_headers": credential_headers,
                "state": state,
            }
        )

    for path in dict.fromkeys(["/health", f"{prefix}/health"]):
        server.custom_route(path, methods=["GET"])(health)

    return server


def main() -> None:
    """Console entry point (``mcp-serve``)."""
    settings = RuntimeSettings()
    server = build_server(
        toolset=settings.toolset,
        module_name=settings.toolset_module,
        host=str(settings.host),
        port=settings.port,
        path_prefix=settings.path_prefix,
    )
    server.run(transport="streamable-http")
