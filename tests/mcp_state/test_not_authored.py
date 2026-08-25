"""Parameters a model may not write: narrowed to a handle, and nothing else.

``NotAuthored`` says one thing about one parameter — *the caller must supply a
value that already exists*. It names no type, so unlike ``Kind`` there is no
second toolset that has to agree with anything. What the client does with it is
narrow the parameter's schema until a literal will not fit.
"""

from typing import Annotated, Any

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool, tool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt.tool_node import ToolNode

from mcp_runtime.declarations import (
    NOT_AUTHORED_META_KEY,
    NOT_AUTHORED_NOTE,
    NotAuthored,
    not_authored,
    with_state_meta,
)
from mcp_runtime.fastmcp_output import to_fastmcp
from mcp_runtime.tool_result import ToolResult
from mcp_state.handles import HANDLE_PREFIX, handle_for
from mcp_state.injection import StateRefusal, bind_injected, not_authored_for
from mcp_state.state import AgentState, StateEntry

AOI = {"type": "FeatureCollection", "features": [{"id": "big-polygon"}]}
GEOJSON_SCHEMA = {"type": "object", "properties": {"type": {"type": "string"}}}


def remote_tool(
    name: str = "clip_raster",
    *,
    properties: dict[str, Any] | None = None,
    required: list[str] | None = None,
    meta: dict[str, Any] | None = None,
    seen: dict[str, Any] | None = None,
    defs: dict[str, Any] | None = None,
) -> StructuredTool:
    """A stand-in for a tool loaded from an MCP server by the adapter."""

    async def call(runtime: Any = None, **arguments: Any) -> Any:
        if seen is not None:
            seen.update(arguments)
        return "called", None

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties
        if properties is not None
        else {"dataset_id": {"type": "string"}, "aoi": GEOJSON_SCHEMA},
        "required": required if required is not None else ["dataset_id", "aoi"],
    }
    if defs is not None:
        schema["$defs"] = defs
    return StructuredTool(
        name=name,
        description=name,
        args_schema=schema,
        coroutine=call,
        response_format="content_and_artifact",
        metadata={"_meta": meta} if meta else None,
    )


def narrowed(*parameters: str) -> dict[str, Any]:
    return {NOT_AUTHORED_META_KEY: list(parameters)}


async def run_tools(tools: list, state: dict[str, Any]) -> dict[str, Any]:
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


def stored(**entries: Any) -> dict[str, StateEntry]:
    return {
        key: StateEntry(value=value, tool="search_datasets", seq=index)
        for index, (key, value) in enumerate(entries.items(), start=1)
    }


# --- the server side ------------------------------------------------------


def test_marker_is_read_off_the_signature() -> None:
    @tool
    async def clip_raster(
        dataset_id: str, aoi: Annotated[dict, NotAuthored()]
    ) -> ToolResult:
        """Clip a dataset."""
        return ToolResult(message="clipped")

    assert not_authored(clip_raster) == ["aoi"]


def test_stamped_into_meta_and_onto_the_description() -> None:
    """Both, deliberately: ``_meta`` for us, the description for everyone else."""

    @tool
    async def clip_raster(
        dataset_id: str, aoi: Annotated[dict, NotAuthored()]
    ) -> ToolResult:
        """Clip a dataset."""
        return ToolResult(message="clipped")

    served = with_state_meta("raster-ops", [clip_raster], [to_fastmcp(clip_raster)])[0]

    assert served.meta[NOT_AUTHORED_META_KEY] == ["aoi"]
    assert NOT_AUTHORED_NOTE in served.parameters["properties"]["aoi"]["description"]
    # Untagged parameters are left exactly as they were.
    assert NOT_AUTHORED_NOTE not in str(served.parameters["properties"]["dataset_id"])


def test_tagging_a_non_parameter_fails_the_build() -> None:
    """A typo is caught at ``build_server``, not at connect."""

    async def clip_raster(dataset_id: str, aoi: dict) -> ToolResult:
        """Clip a dataset."""
        return ToolResult(message="clipped")

    clip_raster.__annotations__["ghost"] = Annotated[dict, NotAuthored()]
    built = StructuredTool.from_function(coroutine=clip_raster, name="clip_raster")

    with pytest.raises(RuntimeError, match="ghost"):
        with_state_meta("raster-ops", [built], [])


def test_read_back_from_meta_by_the_client() -> None:
    tool_with = remote_tool(meta=narrowed("aoi"))
    assert not_authored_for(tool_with) == frozenset({"aoi"})
    assert not_authored_for(remote_tool()) == frozenset()


# --- the schema the model sees --------------------------------------------


def test_schema_accepts_a_handle_and_nothing_else() -> None:
    bound = bind_injected(remote_tool(meta=narrowed("aoi")))
    aoi = bound.args_schema["properties"]["aoi"]

    assert aoi["type"] == "string"
    assert aoi["pattern"] == f"^{HANDLE_PREFIX}"
    assert "anyOf" not in aoi
    # Still required — narrowing changes what fits, not whether it is needed.
    assert "aoi" in bound.args_schema["required"]


def test_other_parameters_keep_their_literal_arm() -> None:
    bound = bind_injected(
        remote_tool(
            properties={"aoi": GEOJSON_SCHEMA, "footprint": GEOJSON_SCHEMA},
            required=["aoi"],
            meta=narrowed("aoi"),
        )
    )

    assert "anyOf" in bound.args_schema["properties"]["footprint"]
    assert "anyOf" not in bound.args_schema["properties"]["aoi"]


def test_the_parameter_description_survives_narrowing() -> None:
    """It is the sentence saying *why*, and the only prose a model reads."""
    bound = bind_injected(
        remote_tool(
            properties={"aoi": {**GEOJSON_SCHEMA, "description": "The area."}},
            required=["aoi"],
            meta=narrowed("aoi"),
        )
    )

    assert bound.args_schema["properties"]["aoi"]["description"].startswith("The area.")


def test_orphaned_definitions_are_dropped() -> None:
    """Narrowing removes the only ``$ref``; the definition must go with it."""
    bound = bind_injected(
        remote_tool(
            properties={"aoi": {"$ref": "#/$defs/FeatureCollection"}},
            required=["aoi"],
            defs={
                "FeatureCollection": {
                    "type": "object",
                    "properties": {"features": {"$ref": "#/$defs/Feature"}},
                },
                "Feature": {"type": "object"},
            },
            meta=narrowed("aoi"),
        )
    )

    assert "$defs" not in bound.args_schema


def test_definitions_still_referenced_are_kept() -> None:
    bound = bind_injected(
        remote_tool(
            properties={
                "aoi": {"$ref": "#/$defs/FeatureCollection"},
                "footprint": {"$ref": "#/$defs/FeatureCollection"},
            },
            required=["aoi"],
            defs={"FeatureCollection": {"type": "object"}},
            meta=narrowed("aoi"),
        )
    )

    assert "FeatureCollection" in bound.args_schema["$defs"]


# --- what happens at call time --------------------------------------------


async def test_a_handle_is_substituted_and_the_tool_sees_the_value() -> None:
    seen: dict[str, Any] = {}
    bound = bind_injected(remote_tool(meta=narrowed("aoi"), seen=seen))

    await run_tools(
        [bound],
        {
            "messages": [
                tool_call(
                    "clip_raster",
                    {
                        "dataset_id": "era5",
                        "aoi": handle_for("dataset-search/geometry"),
                    },
                )
            ],
            "tool_state": stored(**{"dataset-search/geometry": AOI}),
        },
    )

    assert seen["aoi"] == AOI


async def test_a_written_value_is_refused_and_the_options_listed() -> None:
    """The schema should have stopped this; the check is what makes it true."""
    seen: dict[str, Any] = {}
    bound = bind_injected(remote_tool(meta=narrowed("aoi"), seen=seen))

    result = await run_tools(
        [bound],
        {
            "messages": [tool_call("clip_raster", {"dataset_id": "era5", "aoi": AOI})],
            "tool_state": stored(**{"dataset-search/geometry": AOI}),
        },
    )

    message = result["messages"][-1]
    assert message.status == "error"
    assert "a value you wrote" in message.text
    assert "dataset-search/geometry" in message.text
    assert seen == {}


async def test_a_required_narrowed_parameter_left_out_is_refused() -> None:
    seen: dict[str, Any] = {}
    bound = bind_injected(remote_tool(meta=narrowed("aoi"), seen=seen))

    result = await run_tools(
        [bound],
        {
            "messages": [tool_call("clip_raster", {"dataset_id": "era5"})],
            "tool_state": {},
        },
    )

    message = result["messages"][-1]
    assert message.status == "error"
    assert "Nothing has been published" in message.text
    assert seen == {}


async def test_an_optional_narrowed_parameter_left_out_is_fine() -> None:
    seen: dict[str, Any] = {}
    bound = bind_injected(
        remote_tool(
            properties={"dataset_id": {"type": "string"}, "aoi": GEOJSON_SCHEMA},
            required=["dataset_id"],
            meta=narrowed("aoi"),
            seen=seen,
        )
    )

    await run_tools(
        [bound],
        {
            "messages": [tool_call("clip_raster", {"dataset_id": "era5"})],
            "tool_state": {},
        },
    )

    assert seen == {"dataset_id": "era5"}


async def test_a_handle_naming_nothing_is_refused_by_the_existing_check() -> None:
    seen: dict[str, Any] = {}
    bound = bind_injected(remote_tool(meta=narrowed("aoi"), seen=seen))

    result = await run_tools(
        [bound],
        {
            "messages": [
                tool_call(
                    "clip_raster",
                    {"dataset_id": "era5", "aoi": handle_for("nope")},
                )
            ],
            "tool_state": stored(**{"dataset-search/geometry": AOI}),
        },
    )

    assert result["messages"][-1].status == "error"
    assert seen == {}


def test_refusals_are_state_refusals() -> None:
    """So a host can tell "this call was blocked" from "the tool failed"."""
    assert issubclass(StateRefusal, Exception)


def test_a_tool_with_only_a_narrowed_parameter_is_still_wrapped() -> None:
    """No Kind declarations anywhere: the narrowing alone has to trigger it."""
    plain = remote_tool(
        properties={"aoi": {"type": "string"}},
        required=["aoi"],
        meta=narrowed("aoi"),
    )

    assert bind_injected(plain).args_schema is not plain.args_schema


def test_the_health_payload_reports_narrowed_parameters() -> None:
    """The index needs it: it names what a deployment cannot satisfy alone."""
    from mcp_runtime.declarations import state_declarations
    from mcp_runtime.index import StateDeclarations

    @tool
    async def clip_raster(
        dataset_id: str, aoi: Annotated[dict, NotAuthored()]
    ) -> ToolResult:
        """Clip a dataset."""
        return ToolResult(message="clipped")

    declared = state_declarations("raster-ops", [clip_raster])
    assert declared["not_authored"] == [{"tool": "clip_raster", "parameter": "aoi"}]
    # And it survives the model the index actually serves, rather than being
    # dropped as an unknown key.
    assert StateDeclarations(**declared).not_authored[0]["parameter"] == "aoi"
