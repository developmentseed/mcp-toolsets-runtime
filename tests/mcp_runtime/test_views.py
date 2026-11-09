"""Views: MCP resource serving and tool _meta stamping.

Uses synthetic package + tools modules with an on-disk views/ dir, so the
tests never import a real toolset (which come and go).
"""

import sys
import types
from pathlib import Path

import pytest
from langchain_core.tools import tool
from starlette.testclient import TestClient

from mcp_runtime.bridge import to_mcp_tool
from mcp_runtime.server import build_server
from mcp_runtime.tool_result import ToolResult
from mcp_runtime.views import load_views, load_visibility, view_html, with_view_meta


@tool
def show(text: str) -> ToolResult:
    """Show the text."""
    return ToolResult(message=text)


def toolset_with_views(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    package: str,
    *,
    views: dict[str, str] | None = None,
    bundles: tuple[str, ...] = ("panel",),
    tools: list | None = None,
    view_csp: object = None,
    visibility: object = None,
) -> str:
    """Register a synthetic ``<package>`` + ``<package>.tools`` with a views dir.

    Returns the tools module name. ``bundles`` are the view ids to write HTML
    files for under ``<package>/views/``. ``view_csp``, when given, is set as
    the module's ``VIEW_CSP``; left out, the module has no such export at all,
    which is the case every toolset written before it was is in.
    """
    pkg_dir = tmp_path / package
    (pkg_dir / "views").mkdir(parents=True)
    for view_id in bundles:
        (pkg_dir / "views" / f"{view_id}.html").write_text(f"<h1>{view_id}</h1>")

    pkg = types.ModuleType(package)
    pkg.__file__ = str(pkg_dir / "__init__.py")
    monkeypatch.setitem(sys.modules, package, pkg)

    module_name = f"{package}.tools"
    module = types.ModuleType(module_name)
    module.TOOLS = tools if tools is not None else [show]  # type: ignore[attr-defined]
    module.VIEWS = {"show": "panel"} if views is None else views  # type: ignore[attr-defined]
    if view_csp is not None:
        module.VIEW_CSP = view_csp  # type: ignore[attr-defined]
    if visibility is not None:
        module.VISIBILITY = visibility  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module_name, module)
    return module_name


def test_load_views_rejects_bad_export(monkeypatch, tmp_path):
    module = toolset_with_views(monkeypatch, tmp_path, "bad", views={"show": 3})  # type: ignore[dict-item]
    with pytest.raises(RuntimeError, match="VIEWS"):
        load_views(module)


def test_view_html_reads_bundle(monkeypatch, tmp_path):
    module = toolset_with_views(monkeypatch, tmp_path, "reads")
    assert view_html(module, "panel") == "<h1>panel</h1>"


def test_view_html_missing_bundle(monkeypatch, tmp_path):
    module = toolset_with_views(monkeypatch, tmp_path, "nobundle", bundles=())
    with pytest.raises(RuntimeError, match="no bundle"):
        view_html(module, "panel")


def test_with_view_meta_stamps_and_is_pure(monkeypatch, tmp_path):
    module = toolset_with_views(monkeypatch, tmp_path, "stamp")
    original = to_mcp_tool(show)
    (stamped,) = with_view_meta("stamp", module, [original], {"show": "panel"})
    assert stamped.meta == {"ui": {"resourceUri": "ui://stamp/panel"}}
    assert original.meta is None  # input untouched


def test_with_view_meta_unknown_tool(monkeypatch, tmp_path):
    module = toolset_with_views(
        monkeypatch, tmp_path, "unknown", views={"ghost": "panel"}
    )
    with pytest.raises(RuntimeError, match="ghost"):
        with_view_meta("unknown", module, [to_mcp_tool(show)], {"ghost": "panel"})


def test_with_view_meta_stamps_visibility_beside_the_view(monkeypatch, tmp_path):
    """One ``ui`` object per tool, carrying whichever keys the exports give it."""
    module = toolset_with_views(monkeypatch, tmp_path, "seen", tools=[show, other])
    shown, plain = with_view_meta(
        "seen",
        module,
        [to_mcp_tool(show), to_mcp_tool(other)],
        {"show": "panel"},
        {"show": ["app"], "other": ["model"]},
    )
    assert shown.meta == {
        "ui": {"resourceUri": "ui://seen/panel", "visibility": ["app"]}
    }
    assert plain.meta == {"ui": {"visibility": ["model"]}}


def test_with_view_meta_leaves_an_undeclared_tool_alone(monkeypatch, tmp_path):
    module = toolset_with_views(monkeypatch, tmp_path, "quiet", views={})
    (tool,) = with_view_meta("quiet", module, [to_mcp_tool(show)], {}, {})
    assert tool.meta is None


def test_with_view_meta_refuses_visibility_for_an_unknown_tool(monkeypatch, tmp_path):
    module = toolset_with_views(monkeypatch, tmp_path, "typo")
    with pytest.raises(RuntimeError, match=r"VISIBILITY names tool 'shwo'"):
        with_view_meta(
            "typo", module, [to_mcp_tool(show)], {"show": "panel"}, {"shwo": ["app"]}
        )


def test_load_visibility_without_the_export_is_empty(monkeypatch, tmp_path):
    assert load_visibility(toolset_with_views(monkeypatch, tmp_path, "none")) == {}


@pytest.mark.parametrize(
    "visibility",
    [
        ["app"],
        {"show": "app"},
        {"show": []},
        {"show": ["user"]},
        {"show": ["app", 3]},
        {"show": None},
    ],
)
def test_load_visibility_refuses_a_bad_entry(monkeypatch, tmp_path, visibility):
    module = toolset_with_views(monkeypatch, tmp_path, "bad", visibility=visibility)
    with pytest.raises(RuntimeError, match="VISIBILITY"):
        load_visibility(module)


async def test_build_server_stamps_visibility_and_lists_it(monkeypatch, tmp_path):
    toolset_with_views(
        monkeypatch,
        tmp_path,
        "vis",
        tools=[show, other],
        visibility={"other": ["model"]},
    )
    server = build_server("vis", module_name="vis.tools")
    listed = {tool.name: tool for tool in await server.list_tools()}
    assert listed["other"].meta == {"ui": {"visibility": ["model"]}}
    assert listed["show"].meta == {"ui": {"resourceUri": "ui://vis/panel"}}
    with TestClient(server.streamable_http_app()) as client:
        assert client.get("/health").json()["visibility"] == {"other": ["model"]}


async def test_build_server_registers_view_resource(monkeypatch, tmp_path):
    toolset_with_views(monkeypatch, tmp_path, "srv")
    server = build_server("srv", module_name="srv.tools")

    (listed,) = await server.list_tools()
    assert listed.meta == {"ui": {"resourceUri": "ui://srv/panel"}}

    resources = await server.list_resources()
    assert str(resources[0].uri) == "ui://srv/panel"
    # The MCP Apps profile MIME — a plain text/html is rejected by spec hosts.
    assert resources[0].mime_type == "text/html;profile=mcp-app"
    contents = list(await server.read_resource("ui://srv/panel"))
    assert contents[0].content == "<h1>panel</h1>"


def rpc(server, method: str, params: dict | None = None) -> dict:
    """One JSON-RPC request over streamable HTTP, at the 2026 revision.

    Goes through the served app rather than the server object, so what it
    returns is what a host reads off the wire. A ``resources/read`` names its
    URI in an ``mcp-name`` header as well, which the server checks against the
    body.
    """
    headers = {
        "accept": "application/json, text/event-stream",
        "mcp-protocol-version": "2026-07-28",
        "mcp-method": method,
    }
    if params and "uri" in params:
        headers["mcp-name"] = params["uri"]
    meta = {
        "io.modelcontextprotocol/protocolVersion": "2026-07-28",
        "io.modelcontextprotocol/clientCapabilities": {},
        "io.modelcontextprotocol/clientInfo": {"name": "test", "version": "0"},
    }
    # The loopback host passes mcp 2's DNS-rebinding check.
    with TestClient(
        server.streamable_http_app(), base_url="http://127.0.0.1:8000"
    ) as client:
        response = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": method,
                "params": {**(params or {}), "_meta": meta},
            },
            headers=headers,
        )
    return response.json()["result"]


def advertised_extensions(server) -> dict:
    """The extensions ``server/discover`` advertises, at the 2026 revision.

    The 2025 handshake has no ``extensions`` field, so only a client speaking
    2026-07-28 can learn that a server serves MCP Apps.
    """
    return rpc(server, "server/discover")["capabilities"].get("extensions", {})


async def test_a_toolset_with_views_advertises_mcp_apps(monkeypatch, tmp_path):
    toolset_with_views(monkeypatch, tmp_path, "advertising")
    server = build_server("advertising", module_name="advertising.tools")
    assert advertised_extensions(server) == {"io.modelcontextprotocol/ui": {}}


async def test_a_toolset_without_views_advertises_nothing(monkeypatch, tmp_path):
    toolset_with_views(monkeypatch, tmp_path, "viewless", views={})
    server = build_server("viewless", module_name="viewless.tools")
    assert advertised_extensions(server) == {}


# --- VIEW_CSP ---------------------------------------------------------------

#: What a map view asks for: its basemap host for fetches and for tiles, and
#: ``blob:`` so the map library can start the Web Worker it builds from a blob.
MAP_CSP = {
    "connectDomains": ["https://tiles.openfreemap.org"],
    "resourceDomains": ["https://tiles.openfreemap.org", "blob:"],
}


def read_view(server, uri: str) -> dict:
    """``resources/read`` of one ``ui://`` view, as the host receives it."""
    (content,) = rpc(server, "resources/read", {"uri": uri})["contents"]
    return content


async def test_a_declared_csp_is_served_on_the_view(monkeypatch, tmp_path):
    toolset_with_views(
        monkeypatch,
        tmp_path,
        "mapped",
        views={"show": "map"},
        bundles=("map",),
        view_csp={"map": MAP_CSP},
    )
    server = build_server("mapped", module_name="mapped.tools")

    content = read_view(server, "ui://mapped/map")
    assert content["_meta"]["ui"]["csp"] == MAP_CSP
    assert content["text"] == "<h1>map</h1>"
    # On the listing too, so a host can review it before fetching the bundle.
    (listed,) = rpc(server, "resources/list")["resources"]
    assert listed["_meta"]["ui"]["csp"] == MAP_CSP


async def test_without_view_csp_the_view_has_no_meta(monkeypatch, tmp_path):
    """Today's wire, unchanged: the host applies the spec's restrictive default."""
    toolset_with_views(monkeypatch, tmp_path, "plain")
    server = build_server("plain", module_name="plain.tools")

    assert "_meta" not in read_view(server, "ui://plain/panel")
    (listed,) = rpc(server, "resources/list")["resources"]
    assert "_meta" not in listed


async def test_a_view_without_an_entry_has_no_csp(monkeypatch, tmp_path):
    """A declaration covers its own view, and no other."""
    toolset_with_views(
        monkeypatch,
        tmp_path,
        "partial",
        views={"show": "map", "other": "panel"},
        bundles=("map", "panel"),
        tools=[show, other],
        view_csp={"map": MAP_CSP},
    )
    server = build_server("partial", module_name="partial.tools")
    assert read_view(server, "ui://partial/map")["_meta"]["ui"]["csp"] == MAP_CSP
    assert "_meta" not in read_view(server, "ui://partial/panel")


@tool
def other(text: str) -> ToolResult:
    """Show the text elsewhere."""
    return ToolResult(message=text)


def test_view_csp_for_an_undeclared_view_aborts(monkeypatch, tmp_path):
    toolset_with_views(monkeypatch, tmp_path, "ghostcsp", view_csp={"ghost": MAP_CSP})
    with pytest.raises(RuntimeError, match="VIEW_CSP.*'ghost'.*VIEWS"):
        build_server("ghostcsp", module_name="ghostcsp.tools")


@pytest.mark.parametrize(
    ("entry", "match"),
    [
        # The spec has no worker field (ext-apps#566 proposes one).
        ({"workerSrc": ["blob:"]}, "'workerSrc'"),
        # The SDK's Python name is not the wire's.
        ({"connect_domains": ["https://a.example"]}, "'connect_domains'"),
        # A bare host is no origin: a host would read it as a path.
        ({"connectDomains": ["tiles.openfreemap.org"]}, "'tiles.openfreemap.org'"),
        # Plain HTTP would put a view on a mixed-content footing.
        ({"resourceDomains": ["http://x"]}, "'http://x'"),
        # An origin has no path.
        ({"resourceDomains": ["https://a.example/tiles"]}, "'https://a.example/tiles'"),
        # A wildcard covers subdomains, not the whole web.
        ({"resourceDomains": ["https://*"]}, "'https://\\*'"),
        ({"resourceDomains": ["*"]}, "'\\*'"),
        # A CSP keyword would be spliced straight into the policy.
        ({"connectDomains": ["'unsafe-eval'"]}, "unsafe-eval"),
        # wss:// is a connection, never a static resource.
        ({"resourceDomains": ["wss://a.example"]}, "'wss://a.example'"),
        # A scheme-source is a resource or a connection, never a frame or a base.
        ({"frameDomains": ["blob:"]}, "'blob:'"),
        ({"baseUriDomains": ["data:"]}, "'data:'"),
        ({"connectDomains": "https://a.example"}, "list"),
        ({"connectDomains": [3]}, "3"),
        # A port is ASCII digits: \\d would also take Arabic-Indic ones.
        ({"connectDomains": ["https://a.com:\u0661\u0662\u0663"]}, "a.com"),
        # ... and a real port, 1 to 65535.
        ({"connectDomains": ["https://a.com:65536"]}, "65536"),
        ({"connectDomains": ["https://a.com:0"]}, "a.com:0"),
        ({"connectDomains": ["wss://a.com:99999"]}, "99999"),
        # A wildcard over a whole TLD is every host in it.
        ({"resourceDomains": ["https://*.com"]}, "'https://\\*.com'"),
        ({"connectDomains": ["wss://*.org"]}, "'wss://\\*.org'"),
        # A domain is ASCII (punycode for anything else).
        ({"connectDomains": ["https://exämple.com"]}, "ex"),
    ],
)
def test_a_bad_view_csp_entry_aborts(monkeypatch, tmp_path, entry, match):
    toolset_with_views(monkeypatch, tmp_path, "badcsp", view_csp={"panel": entry})
    with pytest.raises(RuntimeError, match=match) as raised:
        build_server("badcsp", module_name="badcsp.tools")
    assert "VIEW_CSP['panel']" in str(raised.value)


def test_view_csp_must_be_a_dict(monkeypatch, tmp_path):
    toolset_with_views(monkeypatch, tmp_path, "listcsp", view_csp=["panel"])
    with pytest.raises(RuntimeError, match="VIEW_CSP must be a dict"):
        build_server("listcsp", module_name="listcsp.tools")


@pytest.mark.parametrize(
    "value",
    [
        "https://tiles.openfreemap.org",
        "https://*.openfreemap.org",
        "https://localhost:8443",
        "https://a.example:65535",
        "https://a.example:1",
        "https://*.tiles.example.co.uk",
        "https://xn--exmple-cua.com",
        "blob:",
        "data:",
    ],
)
def test_accepted_sources(monkeypatch, tmp_path, value):
    toolset_with_views(
        monkeypatch,
        tmp_path,
        "goodcsp",
        view_csp={"panel": {"resourceDomains": [value]}},
    )
    build_server("goodcsp", module_name="goodcsp.tools")


def test_websockets_are_a_connection(monkeypatch, tmp_path):
    """The spec's own ``connectDomains`` example lists a ``wss://`` origin."""
    toolset_with_views(
        monkeypatch,
        tmp_path,
        "wsscsp",
        view_csp={"panel": {"connectDomains": ["wss://realtime.example.com"]}},
    )
    build_server("wsscsp", module_name="wsscsp.tools")


def test_health_lists_each_views_csp(monkeypatch, tmp_path):
    toolset_with_views(
        monkeypatch,
        tmp_path,
        "audited",
        views={"show": "map", "other": "panel"},
        bundles=("map", "panel"),
        tools=[show, other],
        view_csp={"map": MAP_CSP},
    )
    server = build_server("audited", module_name="audited.tools")
    with TestClient(server.streamable_http_app()) as client:
        health = client.get("/health").json()
    assert health["views"] == {"map": {"csp": MAP_CSP}, "panel": {"csp": None}}


def test_health_of_a_toolset_without_views(monkeypatch, tmp_path):
    toolset_with_views(monkeypatch, tmp_path, "noviews", views={})
    server = build_server("noviews", module_name="noviews.tools")
    with TestClient(server.streamable_http_app()) as client:
        assert client.get("/health").json()["views"] == {}
