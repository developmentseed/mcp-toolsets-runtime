"""Tests for the host helpers: view props, view history, and tool-step input."""

import subprocess
import sys

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool

from mcp_agent.host import (
    remember_views,
    step_input,
    view_props,
    view_uri_for,
)


class _Tool(BaseTool):
    """A stand-in for a bound MCP tool carrying LangChain's ``metadata``."""

    name: str = "preview"
    description: str = "a tool"

    def _run(self) -> str:  # pragma: no cover - never invoked
        return ""


def _tool(name: str, uri: str | None) -> BaseTool:
    meta = {"_meta": {"ui": {"resourceUri": uri}}} if uri else {}
    return _Tool(name=name, metadata=meta)


def _result(call_id: str, name: str, structured: dict | None) -> ToolMessage:
    return ToolMessage(
        content="ok",
        name=name,
        tool_call_id=call_id,
        artifact={"structured_content": structured} if structured is not None else None,
    )


def test_view_uri_for_reads_meta():
    assert view_uri_for(_tool("preview", "ui://demo/map")) == "ui://demo/map"
    assert view_uri_for(_tool("preview", None)) is None
    assert view_uri_for(None) is None


def test_view_props_pairs_result_with_its_tools_bundle():
    tools = {"preview": _tool("preview", "ui://demo/map")}
    outputs = {"c1": _result("c1", "preview", {"lat": 1})}
    views = view_props([], outputs, {"ui://demo/map": "<html>map</html>"}, tools)
    assert views == [{"html": "<html>map</html>", "data": {"lat": 1}}]


def test_view_props_skips_tools_without_a_view_or_bundle():
    tools = {
        "plain": _tool("plain", None),
        "orphan": _tool("orphan", "ui://demo/missing"),
    }
    outputs = {
        "c1": _result("c1", "plain", {"x": 1}),
        "c2": _result("c2", "orphan", {"x": 2}),
    }
    assert view_props([], outputs, {"ui://demo/map": "<html/>"}, tools) == []


def test_view_props_orders_newest_first():
    tools = {"preview": _tool("preview", "ui://demo/map")}
    outputs = {
        "c1": _result("c1", "preview", {"n": 1}),
        "c2": _result("c2", "preview", {"n": 2}),
    }
    views = view_props([], outputs, {"ui://demo/map": "<html/>"}, tools)
    assert [view["data"]["n"] for view in views] == [2, 1]


def test_view_props_names_the_tool_from_the_call_when_the_result_omits_it():
    # Some providers return a ToolMessage with no name; the AIMessage's call
    # carries it, and that is what the view lookup has to key on.
    tools = {"preview": _tool("preview", "ui://demo/map")}
    call = AIMessage(
        content="",
        tool_calls=[{"name": "preview", "args": {}, "id": "c1"}],
    )
    result = ToolMessage(content="ok", tool_call_id="c1", artifact={})
    views = view_props([call], {"c1": result}, {"ui://demo/map": "<html/>"}, tools)
    assert views == [{"html": "<html/>", "data": None}]


def test_view_props_tolerates_a_missing_artifact():
    tools = {"preview": _tool("preview", "ui://demo/map")}
    outputs = {"c1": _result("c1", "preview", None)}
    views = view_props([], outputs, {"ui://demo/map": "<html/>"}, tools)
    assert views == [{"html": "<html/>", "data": None}]


def test_view_props_rebuilds_data_that_capture_moved_into_state():
    # The regression this guards: with session state on, the payload a view is
    # written against is no longer on the message. A view must not be able to
    # tell — it gets the tool's whole return either way.
    tools = {"preview": _tool("preview", "ui://demo/map")}
    result = ToolMessage(
        content="Found 3 datasets.\n\n[state updated: search/geometry]",
        name="preview",
        tool_call_id="c1",
        artifact={
            "structured_content": {"message": "Found 3 datasets.", "count": 3},
            "captured_state": {"geometry": "search/geometry"},
        },
    )
    tool_state = {"search/geometry": {"value": {"type": "FeatureCollection"}}}
    views = view_props(
        [], {"c1": result}, {"ui://demo/map": "<html/>"}, tools, tool_state
    )
    assert views == [
        {
            "html": "<html/>",
            "data": {
                "message": "Found 3 datasets.",
                "count": 3,
                "geometry": {"type": "FeatureCollection"},
            },
        }
    ]


def test_view_props_is_unchanged_with_session_state_off():
    # No captured_state map, no tool_state: the artifact is already whole.
    tools = {"preview": _tool("preview", "ui://demo/map")}
    outputs = {"c1": _result("c1", "preview", {"lat": 1})}
    assert view_props([], outputs, {"ui://demo/map": "<html/>"}, tools, None) == [
        {"html": "<html/>", "data": {"lat": 1}}
    ]


def test_remember_views_keeps_the_newest_turns_only():
    history: dict[str, list[dict]] = {}
    for turn in range(25):
        history = remember_views(history, f"turn-{turn}", [{"html": "<html/>"}])
    assert len(history) == 20
    assert "turn-4" not in history
    assert "turn-5" in history
    assert "turn-24" in history


def test_remember_views_replaces_a_turns_snapshot():
    history = remember_views({}, "turn-1", [{"html": "<a/>"}])
    history = remember_views(history, "turn-1", [{"html": "<b/>"}])
    assert history == {"turn-1": [{"html": "<b/>"}]}


def _received(call_id: str, name: str, receipts: dict) -> ToolMessage:
    return ToolMessage(
        content="ok",
        name=name,
        tool_call_id=call_id,
        artifact={"structured_content": {}, "injected_state": receipts},
    )


AOI = {"type": "FeatureCollection", "features": [{"id": "poly"}]}
STATE = {
    "dataset-search/search_datasets/geometry": {
        "value": AOI,
        "tool": "search_datasets",
        "seq": 1,
    }
}
RECEIPT = {
    "aoi": {"key": "dataset-search/search_datasets/geometry", "tool": "search_datasets"}
}


def test_step_input_says_what_a_handle_resolved_to():
    """`@state:<key>` names a value without describing it. The key stays put —
    the model wrote it — and what it does not say is appended."""
    args = {"geometry": "@state:dataset-search/search_datasets/geometry"}
    handle = {"geometry": RECEIPT["aoi"]}

    shown = step_input(args, _received("4", "describe", handle), STATE)

    assert shown["geometry"] == (
        "@state:dataset-search/search_datasets/geometry · 1 feature(s), 0 vertices · from search_datasets"
    )


def test_step_input_never_prints_a_handles_key_twice():
    """The key the model wrote is the argument; repeating it beside itself
    would be noise."""
    args = {"geometry": "@state:dataset-search/search_datasets/geometry"}
    handle = {"geometry": RECEIPT["aoi"]}

    shown = step_input(args, _received("4", "describe", handle), STATE)

    assert shown["geometry"].count("dataset-search/search_datasets/geometry") == 1
    assert "←" not in shown["geometry"]


def test_step_input_annotates_a_handle_with_no_publisher():
    """``tool`` is optional on a state entry, so the line degrades to the
    shape alone rather than dropping out."""
    args = {"request": "@state:gazet/get_aoi/aoi"}
    handle = {"request": {"key": "gazet/get_aoi/aoi"}}
    state = {"gazet/get_aoi/aoi": {"value": AOI, "seq": 1}}

    shown = step_input(args, _received("5", "submit", handle), state)

    assert shown["request"] == ("@state:gazet/get_aoi/aoi · 1 feature(s), 0 vertices")


def test_step_input_falls_back_when_the_value_is_no_longer_in_state():
    """State is bounded and keys are overwritten; the origin is still true."""
    args = {"aoi": "@state:dataset-search/search_datasets/geometry"}
    shown = step_input(args, _received("2", "clip_raster", RECEIPT), {})

    assert (
        shown["aoi"]
        == "@state:dataset-search/search_datasets/geometry · from search_datasets"
    )


def test_step_input_is_untouched_for_a_tool_that_took_nothing_from_state():
    args = {"city": "Reading"}
    assert step_input(args, _result("3", "weather", {}), args) is args
    assert step_input(args, None, None) is args


def test_the_host_helpers_need_no_ui_framework():
    """Importable from a base install, and inert for a host that is not Chainlit.

    Run in a fresh interpreter: importing chainlit registers its lifecycle
    hooks on the importing process, so a host reading a tool's ``_meta`` must
    not pull it in. In-process this would pass on nothing but import order.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import mcp_agent.host; "
            "raise SystemExit(1 if 'chainlit' in sys.modules else 0)",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_the_web_host_still_re_exports_them():
    """The move keeps the old import path working."""
    from mcp_agent import host, web

    for name in host.__all__ if hasattr(host, "__all__") else web.__all__:
        assert getattr(web, name) is getattr(host, name)
