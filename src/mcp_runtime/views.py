"""Optional per-toolset UI views.

A view is a build-time, self-contained HTML bundle (e.g. a React app built by
Vite with everything inlined) that a UI-capable MCP host renders in a sandboxed
iframe, fed the tool's ``structuredContent``. It follows the **MCP Apps** spec
(``modelcontextprotocol/ext-apps``): the runtime serves each view as a
**resource** ``ui://<toolset>/<view_id>`` (MIME ``text/html;profile=mcp-app``)
and stamps the owning tool's ``_meta`` with that URI — exactly what MCP Apps
hosts (Claude, ChatGPT, Goose, VS Code) read. Nothing here executes at call
time, so the runtime stays pure-Python.

The resources go through the SDK's own ``mcp.server.apps.Apps`` extension, so a
server with views also advertises ``io.modelcontextprotocol/ui`` in its
capabilities, and the wire format follows the SDK as the draft spec moves. The
tools do not: ``Apps.tool`` rebuilds a tool from a function's signature, which
would discard the schemas :func:`~mcp_runtime.bridge.to_mcp_tool` derives. So
:func:`with_view_meta` stamps the same ``_meta.ui.resourceUri`` itself.

A toolset opts in with two things, both validated at ``build_server`` time (a
view naming an unknown tool, or a missing bundle, aborts startup):

- ``VIEWS`` — a ``{tool_name: view_id}`` export in the tools module, and
- a built bundle at ``<package>/views/<view_id>.html``.

Views are **progressive enhancement**: the tool's ``message`` and
``structuredContent`` still stand alone in a plain client. Credentials never
reach the iframe — put pre-signed URLs in the ``ToolResult`` instead of tokens.
"""

import importlib
from pathlib import Path

from mcp.server.apps import Apps
from mcp.server.mcpserver.tools import Tool as MCPTool

# The ``_meta`` an MCP Apps host reads: ``{"ui": {"resourceUri": "ui://..."}}``.
VIEW_META_KEY = "ui"


def view_resource_uri(toolset: str, view_id: str) -> str:
    """The ``ui://`` resource URI a view is served under."""
    return f"ui://{toolset}/{view_id}"


def load_views(module_name: str) -> dict[str, str]:
    """A tools module's optional ``VIEWS`` export (tool name -> view id)."""
    views = getattr(importlib.import_module(module_name), "VIEWS", {})
    if not isinstance(views, dict) or not all(
        isinstance(t, str) and t and isinstance(v, str) and v for t, v in views.items()
    ):
        raise RuntimeError(
            f"{module_name}.VIEWS must be a dict of tool name -> view id"
        )
    return dict(views)


def view_html(module_name: str, view_id: str) -> str:
    """Read a view's built HTML bundle from ``<package>/views/``, or raise."""
    location = importlib.import_module(module_name.rsplit(".", 1)[0]).__file__
    if location is None:
        raise RuntimeError(f"cannot locate the package directory for {module_name!r}")
    path = Path(location).parent / "views" / f"{view_id}.html"
    if not path.is_file():
        raise RuntimeError(
            f"view {view_id!r} has no bundle at {path} — build the toolset's ui/ "
            "(vite build) so <package>/views/<view_id>.html exists"
        )
    return path.read_text(encoding="utf-8")


def with_view_meta(
    toolset: str,
    module_name: str,
    tools: list[MCPTool],
    views: dict[str, str],
) -> list[MCPTool]:
    """Return ``tools`` with each view-bearing tool's ``_meta`` stamped.

    Pure: inputs are left untouched; each view-owning tool is replaced by a copy
    carrying its ``ui://`` resource URI under ``_meta``. Validates first that
    every ``VIEWS`` entry names a real tool with a built bundle, raising (naming
    the offender) otherwise — so a broken view fails ``build_server``.
    """
    names = {tool.name for tool in tools}
    for tool_name, view_id in views.items():
        if tool_name not in names:
            raise RuntimeError(
                f"{module_name}.VIEWS names tool {tool_name!r}, which is not in TOOLS"
            )
        view_html(module_name, view_id)  # raise now if the bundle is missing

    def stamped(tool: MCPTool) -> MCPTool:
        if tool.name not in views:
            return tool
        uri = view_resource_uri(toolset, views[tool.name])
        return tool.model_copy(
            update={"meta": {**(tool.meta or {}), VIEW_META_KEY: {"resourceUri": uri}}}
        )

    return [stamped(tool) for tool in tools]


def views_extension(
    toolset: str, module_name: str, views: dict[str, str]
) -> Apps | None:
    """The Apps extension serving each view as ``ui://<toolset>/<view_id>``.

    ``None`` for a toolset with no views, so it does not advertise a capability
    it has nothing behind.
    """
    if not views:
        return None
    apps = Apps()
    for view_id in sorted(set(views.values())):
        apps.add_html_resource(
            view_resource_uri(toolset, view_id),
            view_html(module_name, view_id),
            name=f"{toolset}-{view_id}",
        )
    return apps
