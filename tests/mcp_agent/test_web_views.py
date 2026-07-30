"""Tests for web.py's view-panel helpers (no Chainlit session required)."""

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool

from mcp_agent.web import remember_views, view_props, view_uri_for


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
