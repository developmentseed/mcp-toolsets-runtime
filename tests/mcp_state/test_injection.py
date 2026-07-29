"""Declared parameters: hidden from the model, filled from session state.

The end-to-end case is the interesting one, so it is the first test: a tool
in one toolset publishes a value, a tool in a *different* toolset on a
*different* server takes it, and the two are matched only by kind.
"""

from typing import Annotated, Any, NotRequired

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool, ToolException, tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt.tool_node import ToolNode

from mcp_runtime.declarations import (
    CONSUMES_META_KEY,
    PRODUCES_META_KEY,
    Kind,
    consumed_kinds,
    output_kinds,
    qualified,
    state_declarations,
    with_state_meta,
)
from mcp_runtime.fastmcp_output import to_fastmcp
from mcp_runtime.kinds import GEOJSON_AREA_OF_INTEREST, GEOJSON_FOOTPRINT
from mcp_runtime.tool_result import ToolError, ToolResult
from mcp_state.injection import bind_injected, resolve
from mcp_state.state import AgentState, StateEntry, merge_tool_state

AOI = {"type": "FeatureCollection", "features": [{"id": "big-polygon"}]}
GEOJSON_SCHEMA = {"type": "object", "properties": {"type": {"type": "string"}}}


def declaration(**overrides: Any) -> dict[str, Any]:
    """One consumed-kind declaration as a server would put it on the wire."""
    return {
        "parameter": "aoi",
        "kind": GEOJSON_AREA_OF_INTEREST,
        "required": True,
        "modelGeneratable": False,
        **overrides,
    }


def remote_tool(
    name: str,
    *,
    properties: dict[str, Any],
    required: list[str],
    consumes: list[dict[str, Any]] | None = None,
    seen: dict[str, Any] | None = None,
) -> StructuredTool:
    """A stand-in for a tool loaded from an MCP server by the adapter.

    Mirrors what ``langchain_mcp_adapters`` builds: a dict ``args_schema``
    taken verbatim from the server's ``inputSchema``, a ``**arguments``
    coroutine, and the server's ``_meta`` preserved under ``metadata``.
    """

    async def call(runtime: Any = None, **arguments: Any) -> Any:
        if seen is not None:
            seen.update(arguments)
        return "called", None

    return StructuredTool(
        name=name,
        description=name,
        args_schema={
            "type": "object",
            "properties": properties,
            "required": required,
        },
        coroutine=call,
        response_format="content_and_artifact",
        metadata={"_meta": {CONSUMES_META_KEY: consumes}} if consumes else None,
    )


async def run_tools(tools: list, state: dict[str, Any]) -> dict[str, Any]:
    """Drive a ToolNode inside a real graph so state injection actually runs."""
    graph = StateGraph(AgentState)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "tools")
    graph.add_edge("tools", END)
    return await graph.compile().ainvoke(state)


def tool_call(name: str, args: dict[str, Any]) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": "1", "type": "tool_call"}],
    )


async def test_value_crosses_toolsets_and_servers_matched_only_by_kind() -> None:
    """A publisher in one toolset fills a consumer in another, via kind alone."""
    seen: dict[str, Any] = {}
    # Served by "raster-ops"; knows nothing about who produces an AOI.
    clip = remote_tool(
        "clip_raster",
        properties={"dataset_id": {"type": "string"}, "aoi": GEOJSON_SCHEMA},
        required=["dataset_id", "aoi"],
        consumes=[declaration()],
        seen=seen,
    )
    bound = bind_injected(clip)

    # The model is never offered `aoi` at all.
    parameters = convert_to_openai_tool(bound)["function"]["parameters"]
    assert "aoi" not in parameters["properties"]
    assert parameters["required"] == ["dataset_id"]

    # Published earlier by "dataset-search", on a different server.
    await run_tools(
        [bound],
        {
            "messages": [tool_call("clip_raster", {"dataset_id": "era5"})],
            "tool_state": {
                qualified("dataset-search", "geometry"): StateEntry(
                    value=AOI, kind=GEOJSON_AREA_OF_INTEREST, tool="search", seq=1
                )
            },
        },
    )
    assert seen == {"dataset_id": "era5", "aoi": AOI}


async def test_a_wrong_kind_is_not_injected() -> None:
    """A footprint never satisfies a parameter asking for an area of interest."""
    found, _ = resolve(
        declaration(),
        {"x/f": StateEntry(value=AOI, kind=GEOJSON_FOOTPRINT, seq=1)},
        None,
    )
    assert found is False


async def test_the_most_recent_entry_of_a_kind_wins() -> None:
    """Two AOIs in state: the one published last is the one in play."""
    older = {"type": "FeatureCollection", "features": ["older"]}
    found, value = resolve(
        declaration(),
        {
            "a/geometry": StateEntry(value=older, kind=GEOJSON_AREA_OF_INTEREST, seq=1),
            "b/geometry": StateEntry(value=AOI, kind=GEOJSON_AREA_OF_INTEREST, seq=7),
        },
        None,
    )
    assert (found, value) == (True, AOI)


async def test_a_value_failing_the_parameter_schema_is_skipped() -> None:
    """Same kind, wrong dialect: passed over rather than sent to the server."""
    found, _ = resolve(
        declaration(),
        {
            "a/geometry": StateEntry(
                value="not-an-object", kind=GEOJSON_AREA_OF_INTEREST
            )
        },
        {"type": "object"},
    )
    assert found is False


async def test_a_missing_required_value_tells_the_model_what_to_do() -> None:
    """The model cannot supply it, so the error has to name the way forward."""
    clip = remote_tool(
        "clip_raster",
        properties={"aoi": GEOJSON_SCHEMA},
        required=["aoi"],
        consumes=[declaration()],
    )
    with pytest.raises(ToolException, match="has published it"):
        await bind_injected(clip).ainvoke({"args": {}, "id": "1", "type": "tool_call"})


async def test_an_explicitly_passed_value_is_never_overridden() -> None:
    """Injection fills a gap; it does not take the call away from the caller."""
    seen: dict[str, Any] = {}
    clip = remote_tool(
        "clip_raster",
        properties={"aoi": GEOJSON_SCHEMA},
        required=["aoi"],
        consumes=[declaration()],
        seen=seen,
    )
    explicit = {"type": "FeatureCollection", "features": ["explicit"]}
    await bind_injected(clip).ainvoke(
        {"args": {"aoi": explicit}, "id": "1", "type": "tool_call"}
    )
    assert seen["aoi"] == explicit


async def test_a_tool_with_nothing_bulk_and_nothing_declared_is_untouched() -> None:
    """No declaration and no parameter worth a handle: return it as it came."""
    plain = remote_tool("search", properties={"q": {"type": "string"}}, required=["q"])
    assert bind_injected(plain) is plain


async def test_a_model_generatable_parameter_stays_visible_when_unpublished() -> None:
    """Nothing publishes the kind, so the model fills it as any MCP client would."""
    clip = remote_tool(
        "clip_raster",
        properties={"bbox": {"type": "array"}},
        required=["bbox"],
        consumes=[declaration(parameter="bbox", modelGeneratable=True)],
    )
    bound = bind_injected(clip, published=frozenset())
    parameters = convert_to_openai_tool(bound)["function"]["parameters"]
    assert "bbox" in parameters["properties"]


async def test_a_non_generatable_parameter_stays_hidden_when_unpublished() -> None:
    """Hiding it is the point: the tool is dead, and wiring reports it as such."""
    clip = remote_tool(
        "clip_raster",
        properties={"aoi": GEOJSON_SCHEMA},
        required=["aoi"],
        consumes=[declaration()],
    )
    bound = bind_injected(clip, published=frozenset())
    parameters = convert_to_openai_tool(bound)["function"]["parameters"]
    assert "aoi" not in parameters["properties"]


def test_merge_stamps_write_order_so_recency_is_knowable() -> None:
    first = merge_tool_state({}, {"a/x": StateEntry(value=1)})
    second = merge_tool_state(first, {"b/y": StateEntry(value=2)})
    assert second["b/y"]["seq"] > second["a/x"]["seq"]


# --- the server-side declaration -----------------------------------------


class SearchResult(ToolResult):
    geometry: NotRequired[Annotated[dict, Kind(GEOJSON_AREA_OF_INTEREST)]]


@tool
async def search(query: str) -> SearchResult | ToolError:
    """Search."""
    return SearchResult(message="found", geometry=AOI)


@tool
async def clip(
    dataset_id: str,
    aoi: Annotated[dict, Kind(GEOJSON_AREA_OF_INTEREST, model_generatable=False)],
) -> ToolResult | ToolError:
    """Clip."""
    return ToolResult(message="clipped")


def test_one_marker_reads_off_both_sides_of_a_signature() -> None:
    """The same tag means "takes" on a parameter and "publishes" on a field."""
    assert consumed_kinds(clip)["aoi"].kind == GEOJSON_AREA_OF_INTEREST
    assert output_kinds(search) == {"geometry": GEOJSON_AREA_OF_INTEREST}


def test_meta_carries_both_halves_to_the_client() -> None:
    stamped = with_state_meta(
        "raster-ops", [search, clip], [to_fastmcp(t) for t in (search, clip)]
    )
    by_name = {t.name: t for t in stamped}
    assert by_name["clip"].meta[CONSUMES_META_KEY] == [
        {
            "parameter": "aoi",
            "kind": GEOJSON_AREA_OF_INTEREST,
            "required": True,
            "modelGeneratable": False,
        }
    ]
    assert by_name["search"].meta[PRODUCES_META_KEY] == [
        {
            "stateKey": "raster-ops/geometry",
            "field": "geometry",
            "kind": GEOJSON_AREA_OF_INTEREST,
        }
    ]


def test_required_is_read_from_the_tools_own_schema() -> None:
    """A Python default is what makes a parameter optional; nothing else says so."""

    @tool
    async def defaulted(
        dataset_id: str,
        aoi: Annotated[dict | None, Kind(GEOJSON_AREA_OF_INTEREST)] = None,
    ) -> ToolResult:
        """Clip."""
        return ToolResult(message="x")

    stamped = with_state_meta("t", [defaulted], [to_fastmcp(defaulted)])
    assert stamped[0].meta[CONSUMES_META_KEY][0]["required"] is False


def test_health_advertises_both_halves_to_a_plain_http_client() -> None:
    """What the index aggregates, without speaking MCP."""
    declared = state_declarations([search, clip])
    assert declared["produces"] == [GEOJSON_AREA_OF_INTEREST]
    assert declared["consumes"] == [
        {
            "tool": "clip",
            "parameter": "aoi",
            "kind": GEOJSON_AREA_OF_INTEREST,
            "required": True,
            "modelGeneratable": False,
        }
    ]


def test_a_tag_on_a_nonexistent_parameter_fails_the_build() -> None:
    @tool
    async def broken(
        x: str, ghost: Annotated[dict, Kind(GEOJSON_FOOTPRINT)]
    ) -> ToolResult:
        """Broken."""
        return ToolResult(message="x")

    broken.args.pop("ghost")
    with pytest.raises(RuntimeError, match="not one of its parameters"):
        with_state_meta("t", [broken], [to_fastmcp(broken)])
