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

A view that fetches anything from elsewhere also declares where, in an optional
``VIEW_CSP`` export: ``{view_id: {"connectDomains": [...], ...}}``. The spec
has a host give a ``ui://`` resource with no ``_meta.ui.csp`` a restrictive
default (``connect-src 'none'``, no remote scripts, images or fonts) and never
allow an undeclared origin, so without one a map view gets no tiles. The
runtime passes each entry to the SDK as a ``ResourceCsp``, which serves it as
the resource's ``_meta.ui.csp``; see :func:`load_view_csp` for what it checks.

Views are **progressive enhancement**: the tool's ``message`` and
``structuredContent`` still stand alone in a plain client. Credentials never
reach the iframe — put pre-signed URLs in the ``ToolResult`` instead of tokens.
"""

import importlib
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mcp.server.apps import Apps, ResourceCsp
from mcp.server.mcpserver.tools import Tool as MCPTool

# The ``_meta`` an MCP Apps host reads: ``{"ui": {"resourceUri": "ui://..."}}``.
VIEW_META_KEY = "ui"

#: A view's declared CSP as it goes on the wire: ``{"connectDomains": [...]}``.
ViewCsp = dict[str, list[str]]

#: The fields a ``VIEW_CSP`` entry may set, by their wire name. Read off the
#: SDK's ``ResourceCsp`` rather than listed here, so a field the spec and the
#: SDK add (ext-apps#566 proposes a ``workerDomains``) becomes declarable for
#: ``https://`` origins with a dependency bump. It fails closed beyond that: a
#: scheme-source such as ``blob:`` in a new field is refused until someone adds
#: the field to ``_SCHEME_FIELDS`` on purpose. Today: ``connectDomains``,
#: ``resourceDomains``, ``frameDomains`` and ``baseUriDomains``.
CSP_FIELDS: frozenset[str] = frozenset(
    field.alias or name for name, field in ResourceCsp.model_fields.items()
)

#: An origin a view may name: ``https://`` (``wss://`` for a connection), an
#: ASCII host (punycode for anything else), optionally a ``*.`` subdomain
#: wildcard as the spec's own examples use, and optionally a port from 1 to
#: 65535. No path, no query, nothing after the host. Each value is spliced into
#: the host's CSP header as one source, so anything looser lets a toolset write
#: CSP of its own: ``'unsafe-eval'``, a bare ``*``, or a second directive after
#: a ``;``. A wildcard needs two labels after it: ``*.example.com`` is one
#: site's subdomains, ``*.com`` every site in a TLD.
_LABEL = r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?"
_HOST = rf"(?:\*\.{_LABEL}(?:\.{_LABEL})+|{_LABEL}(?:\.{_LABEL})*)"
# Matched with ``fullmatch`` (``$`` would let a trailing newline through), and
# with ``[0-9]``, not ``\d``, which matches every Unicode digit.
_ORIGIN = re.compile(rf"(?P<scheme>https|wss)://{_HOST}(?::(?P<port>[0-9]{{1,5}}))?")

#: Scheme-sources a view may list beside origins. ``blob:`` is the one a map
#: needs: MapLibre GL starts its tile worker from a ``blob:`` URL, and the
#: spec's CSP template has no ``worker-src``, so the worker falls back to
#: ``script-src``, which only ``resourceDomains`` widens. ``data:`` is the
#: inlined-asset counterpart (a font or an image as a data URI). Either one in
#: ``resourceDomains`` goes into ``script-src`` too, not just images and fonts.
SCHEME_SOURCES: frozenset[str] = frozenset({"blob:", "data:"})
#: The fields a scheme-source may go in. A fixed list, unlike ``CSP_FIELDS``:
#: putting ``blob:`` into a directive is a decision, not a dependency bump.
_SCHEME_FIELDS: frozenset[str] = frozenset({"resourceDomains", "connectDomains"})


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


def _csp_source_ok(field: str, value: str) -> bool:
    """Whether ``value`` may be listed under ``field`` of a view's CSP.

    A WebSocket is a connection and nothing else, so ``wss://`` is accepted in
    ``connectDomains`` only (the spec's own example lists one there). A
    scheme-source is something a view loads or fetches, so ``blob:`` and
    ``data:`` go in ``resourceDomains`` and ``connectDomains`` only: in
    ``frameDomains`` or ``baseUriDomains`` they would open a frame or a base
    URI no view needs.
    """
    if value in SCHEME_SOURCES:
        return field in _SCHEME_FIELDS
    origin = _ORIGIN.fullmatch(value)
    if origin is None:
        return False
    port = origin["port"]
    if port is not None and not 1 <= int(port) <= 65535:
        return False
    return origin["scheme"] == "https" or field == "connectDomains"


def check_view_csp(where: str, entry: object) -> ViewCsp:
    """Check one view's CSP entry and return a copy, or raise naming the fault.

    ``where`` names the entry in the message (``tools.VIEW_CSP['map']``).
    :func:`load_view_csp` runs it on every entry before anything is served.
    """
    if not isinstance(entry, dict):
        raise RuntimeError(f"{where} must be a dict of CSP field -> list of sources")
    csp: ViewCsp = {}
    for field, values in entry.items():
        if field not in CSP_FIELDS:
            raise RuntimeError(
                f"{where} has unknown field {field!r}; "
                f"the fields are {', '.join(sorted(CSP_FIELDS))}"
            )
        if not isinstance(values, list):
            raise RuntimeError(f"{where}[{field!r}] must be a list of sources")
        for value in values:
            if not isinstance(value, str) or not _csp_source_ok(field, value):
                raise RuntimeError(
                    f"{where}[{field!r}] has {value!r}, which is not an "
                    "https:// origin (a *. subdomain wildcard over at least "
                    "two labels and a port from 1 to 65535 allowed, no path), "
                    "a wss:// origin in connectDomains, or blob: or data: in "
                    "resourceDomains or connectDomains"
                )
        csp[field] = list(values)
    return csp


def load_view_csp(module_name: str, views: Mapping[str, str]) -> dict[str, ViewCsp]:
    """A tools module's optional ``VIEW_CSP`` export (view id -> CSP), checked.

    Raises, naming the view, field or value at fault, unless:

    - every key is a view id that ``VIEWS`` names (a CSP for a view nobody
      serves is a typo, and would otherwise be dropped without a word);
    - every field is one of ``ResourceCsp``'s, by its wire (camelCase) name;
    - every value is a list of ``https://`` origins, optionally with a ``*.``
      subdomain wildcard and a port, ``wss://`` ones in ``connectDomains``, or
      one of the scheme-sources ``blob:`` and ``data:`` in ``resourceDomains``
      or ``connectDomains``.

    The checks are strict on purpose. A host builds its CSP header from these
    strings, and the spec says it MUST NOT allow an undeclared domain, so what
    a toolset declares here is the most its views can reach (a host may allow
    less). Startup is the one moment someone is reading the output.

    Returns copies, in the wire's own shape: what ``_meta.ui.csp`` carries and
    what ``/health`` lists. A toolset without the export gets ``{}``, and its
    views go out with no ``_meta.ui.csp``, as before.
    """
    declared = getattr(importlib.import_module(module_name), "VIEW_CSP", {})
    where = f"{module_name}.VIEW_CSP"
    if not isinstance(declared, dict):
        raise RuntimeError(f"{where} must be a dict of view id -> CSP")
    view_ids = set(views.values())
    checked: dict[str, ViewCsp] = {}
    for view_id, entry in declared.items():
        here = f"{where}[{view_id!r}]"
        if view_id not in view_ids:
            raise RuntimeError(
                f"{where} names view {view_id!r}, which no VIEWS entry serves"
            )
        checked[view_id] = check_view_csp(here, entry)
    return checked


#: Who a tool can be for, as MCP Apps' ``_meta.ui.visibility`` names them.
VISIBILITY_VALUES = frozenset({"model", "app"})


def load_visibility(module_name: str) -> dict[str, list[str]]:
    """A tools module's optional ``VISIBILITY`` export (tool name -> who it is for).

    Raises, naming the entry at fault, unless it is a dict whose every value is
    a non-empty list of ``"model"`` and ``"app"``. Whether each key is a real
    tool is :func:`with_view_meta`'s check, which has ``TOOLS`` to hand.

    A tool it does not name keeps the spec's default, ``["model", "app"]``,
    and goes out with no ``_meta.ui.visibility``. ``["model"]`` keeps a tool
    from a view's ``tools/call``; ``["app"]`` keeps it out of the agent's tool
    list, so only a view calls it.
    """
    declared = getattr(importlib.import_module(module_name), "VISIBILITY", {})
    where = f"{module_name}.VISIBILITY"
    if not isinstance(declared, dict):
        raise RuntimeError(
            f'{where} must be a dict of tool name -> a non-empty list of "model" and "app"'
        )
    checked: dict[str, list[str]] = {}
    for tool_name, entry in declared.items():
        if (
            not isinstance(tool_name, str)
            or not isinstance(entry, list)
            or not entry
            or not all(isinstance(each, str) for each in entry)
            or not set(entry) <= VISIBILITY_VALUES
        ):
            raise RuntimeError(
                f'{where}[{tool_name!r}] must be a non-empty list of "model" and '
                f'"app", not {entry!r}'
            )
        checked[tool_name] = list(dict.fromkeys(entry))
    return checked


def view_declarations(
    views: Mapping[str, str], csp: Mapping[str, ViewCsp]
) -> dict[str, dict[str, Any]]:
    """What ``/health`` and the index say about each view: ``{view_id: {...}}``.

    Lists the CSP each view declared, or ``None`` for one that declared none
    and so runs under the host's restrictive default. That is the spec's
    "Audit Trail": a deployer reads what a toolset's views may fetch here,
    without reading its code. A dict per view leaves room for the rest of a
    view's ``_meta.ui`` (permissions, a dedicated domain) beside it.
    """
    return {
        view_id: {"csp": csp.get(view_id)} for view_id in sorted(set(views.values()))
    }


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
    visibility: Mapping[str, list[str]] | None = None,
) -> list[MCPTool]:
    """Return ``tools`` with each view-bearing or visibility-declaring tool's ``_meta`` stamped.

    Pure: inputs are left untouched; each tool either export names is replaced
    by a copy carrying one ``_meta.ui`` object: its ``ui://`` resource URI
    (``VIEWS``) and who it is for (``VISIBILITY``, :func:`load_visibility`'s
    output). Validates first that every ``VIEWS`` entry names a real tool with
    a built bundle, and every ``VISIBILITY`` entry a real tool, raising (naming
    the offender) otherwise — so a broken declaration fails ``build_server``.
    """
    visibility = visibility or {}
    names = {tool.name for tool in tools}
    for tool_name, view_id in views.items():
        if tool_name not in names:
            raise RuntimeError(
                f"{module_name}.VIEWS names tool {tool_name!r}, which is not in TOOLS"
            )
        view_html(module_name, view_id)  # raise now if the bundle is missing
    for tool_name in visibility:
        if tool_name not in names:
            raise RuntimeError(
                f"{module_name}.VISIBILITY names tool {tool_name!r}, which is not in TOOLS"
            )

    def stamped(tool: MCPTool) -> MCPTool:
        ui: dict[str, Any] = {}
        if tool.name in views:
            ui["resourceUri"] = view_resource_uri(toolset, views[tool.name])
        if tool.name in visibility:
            ui["visibility"] = list(visibility[tool.name])
        if not ui:
            return tool
        return tool.model_copy(
            update={"meta": {**(tool.meta or {}), VIEW_META_KEY: ui}}
        )

    return [stamped(tool) for tool in tools]


def views_extension(
    toolset: str,
    module_name: str,
    views: dict[str, str],
    csp: Mapping[str, ViewCsp] | None = None,
) -> Apps | None:
    """The Apps extension serving each view as ``ui://<toolset>/<view_id>``.

    ``None`` for a toolset with no views, so it does not advertise a capability
    it has nothing behind. ``csp`` is :func:`load_view_csp`'s output: a view
    with an entry is served with it as ``_meta.ui.csp``, on both the listing
    and the read, and a view without one is served with no ``_meta`` at all.
    """
    csp = csp or {}
    if not views:
        return None
    apps = Apps()
    for view_id in sorted(set(views.values())):
        declared = csp.get(view_id)
        apps.add_html_resource(
            view_resource_uri(toolset, view_id),
            view_html(module_name, view_id),
            name=f"{toolset}-{view_id}",
            # Validated by wire name, so built by alias: ResourceCsp dumps it
            # back by alias, dropping only the fields the view left unset.
            csp=ResourceCsp.model_validate(declared) if declared is not None else None,
        )
    return apps
