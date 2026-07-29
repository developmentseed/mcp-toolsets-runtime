"""Injected parameters: hidden from the model, filled from session state.

The end-to-end case is the interesting one, so it is the first test: a tool
in one toolset publishes a value, a tool in a *different* toolset on a
*different* server consumes it, and the two are matched only by kind.
"""

from typing import Annotated, Any, NotRequired

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool, ToolException, tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt.tool_node import ToolNode

from mcp_state.injection import bind_injected, resolve
from mcp_state.state import AgentState, StateEntry, merge_tool_state
from mcp_runtime.injected import (
    INJECTED_META_KEY,
    PRODUCES_META_KEY,
    Injected,
    Kind,
    injected_params,
    produced_keys,
    qualified,
    with_injected_meta,
)
from mcp_runtime.kinds import GEOJSON_AREA_OF_INTEREST, GEOJSON_FOOTPRINT
from mcp_runtime.tool_result import ToolError, ToolResult

AOI = {"type": "FeatureCollection", "features": [{"id": "big-polygon"}]}
GEOJSON_SCHEMA = {"type": "object", "properties": {"type": {"type": "string"}}}


def remote_tool(
    name: str,
    *,
    properties: dict[str, Any],
    required: list[str],
    injected: list[dict[str, Any]] | None = None,
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
        metadata={"_meta": {INJECTED_META_KEY: injected}} if injected else None,
    )


async def run_tools(tools: list, state: dict[str, Any]) -> dict[str, Any]:
    """Drive a ToolNode inside a real graph so state injection actually runs."""
    graph = StateGraph(AgentState)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "tools")
    graph.add_edge("tools", END)
    return await graph.compile().ainvoke(state)


async def test_value_crosses_toolsets_and_servers_matched_only_by_kind() -> None:
    """A publisher in one toolset fills a consumer in another, via kind alone."""
    seen: dict[str, Any] = {}
    # Served by "raster-ops"; knows nothing about who produces an AOI.
    clip = remote_tool(
        "clip_raster",
        properties={"dataset_id": {"type": "string"}, "aoi": GEOJSON_SCHEMA},
        required=["dataset_id", "aoi"],
        injected=[
            {
                "parameter": "aoi",
                "kind": GEOJSON_AREA_OF_INTEREST,
                "required": True,
            }
        ],
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
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "clip_raster",
                            "args": {"dataset_id": "era5"},
                            "id": "1",
                            "type": "tool_call",
                        }
                    ],
                )
            ],
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
        {"parameter": "aoi", "kind": GEOJSON_AREA_OF_INTEREST},
        {"x/f": StateEntry(value=AOI, kind=GEOJSON_FOOTPRINT, seq=1)},
        None,
    )
    assert found is False


async def test_the_most_recent_entry_of_a_kind_wins() -> None:
    """Two AOIs in state: the one published last is the one in play."""
    older = {"type": "FeatureCollection", "features": ["older"]}
    found, value = resolve(
        {"parameter": "aoi", "kind": GEOJSON_AREA_OF_INTEREST},
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
        {"parameter": "aoi", "kind": GEOJSON_AREA_OF_INTEREST},
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
        injected=[
            {"parameter": "aoi", "kind": GEOJSON_AREA_OF_INTEREST, "required": True}
        ],
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
        injected=[
            {"parameter": "aoi", "kind": GEOJSON_AREA_OF_INTEREST, "required": True}
        ],
        seen=seen,
    )
    explicit = {"type": "FeatureCollection", "features": ["explicit"]}
    await bind_injected(clip).ainvoke(
        {"args": {"aoi": explicit}, "id": "1", "type": "tool_call"}
    )
    assert seen["aoi"] == explicit


async def test_tools_without_declarations_pass_through_untouched() -> None:
    """Third-party servers declare nothing and must be unaffected."""
    plain = remote_tool("search", properties={"q": {"type": "string"}}, required=["q"])
    assert bind_injected(plain) is plain


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
    aoi: Annotated[dict, Injected(kind=GEOJSON_AREA_OF_INTEREST)],
) -> ToolResult | ToolError:
    """Clip."""
    return ToolResult(message="clipped")


def test_declarations_are_read_off_the_signature() -> None:
    assert injected_params(clip)["aoi"].kind == GEOJSON_AREA_OF_INTEREST
    assert produced_keys(search) == {"geometry": GEOJSON_AREA_OF_INTEREST}


def test_meta_carries_both_halves_to_the_client() -> None:
    from mcp_runtime.fastmcp_output import to_fastmcp

    stamped = with_injected_meta(
        "raster-ops", [search, clip], [to_fastmcp(t) for t in (search, clip)]
    )
    by_name = {t.name: t for t in stamped}
    assert by_name["clip"].meta[INJECTED_META_KEY] == [
        {"parameter": "aoi", "required": True, "kind": GEOJSON_AREA_OF_INTEREST}
    ]
    assert by_name["search"].meta[PRODUCES_META_KEY] == [
        {
            "stateKey": "raster-ops/geometry",
            "field": "geometry",
            "kind": GEOJSON_AREA_OF_INTEREST,
        }
    ]


def test_health_advertises_both_halves_to_a_plain_http_client() -> None:
    """What the index aggregates, without speaking MCP."""
    from mcp_runtime.injected import state_declarations

    declared = state_declarations([search, clip])
    assert declared["produces"] == [GEOJSON_AREA_OF_INTEREST]
    assert declared["injects"] == [
        {
            "tool": "clip",
            "parameter": "aoi",
            "required": True,
            "kind": GEOJSON_AREA_OF_INTEREST,
        }
    ]


def test_optional_injection_needs_a_python_default() -> None:
    """Otherwise the tool's own schema rejects the call it was meant to allow.

    A client omits an unfilled optional parameter entirely, so the tool can
    pick its own value — but only if its schema permits the absence.
    """
    from mcp_runtime.fastmcp_output import to_fastmcp

    @tool
    async def no_default(
        dataset_id: str,
        aoi: Annotated[dict, Injected(kind=GEOJSON_AREA_OF_INTEREST, required=False)],
    ) -> ToolResult:
        """Clip."""
        return ToolResult(message="x")

    with pytest.raises(RuntimeError, match="give the parameter a Python default"):
        with_injected_meta("t", [no_default], [to_fastmcp(no_default)])

    @tool
    async def defaulted(
        dataset_id: str,
        aoi: Annotated[
            dict | None, Injected(kind=GEOJSON_AREA_OF_INTEREST, required=False)
        ] = None,
    ) -> ToolResult:
        """Clip."""
        return ToolResult(message="x")

    with_injected_meta("t", [defaulted], [to_fastmcp(defaulted)])  # no raise


def test_a_declaration_naming_a_nonexistent_parameter_fails_the_build() -> None:
    from mcp_runtime.fastmcp_output import to_fastmcp

    @tool
    async def broken(
        x: str, ghost: Annotated[dict, Injected(kind=GEOJSON_FOOTPRINT)]
    ) -> ToolResult:
        """Broken."""
        return ToolResult(message="x")

    broken.args.pop("ghost")
    with pytest.raises(RuntimeError, match="not one of its parameters"):
        with_injected_meta("t", [broken], [to_fastmcp(broken)])
