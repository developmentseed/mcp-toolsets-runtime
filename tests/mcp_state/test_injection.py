"""Binding a tool: handles resolved, refusals delivered, everything else left.

The kind-matching path is gone. What binding does now is rewrite a schema so a
model can name a stored value, substitute what it names on the way out, and
refuse — as a *result*, never an exception — the calls it cannot serve.

Narrowing (``NotAuthored``) has its own module, ``test_not_authored.py``.
"""

from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool, ToolException
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt.tool_node import ToolNode

from mcp_runtime.declarations import NOT_AUTHORED_META_KEY
from mcp_state.handles import handle_for
from mcp_state.injection import StateRefusal, bind_all_injected, bind_injected
from mcp_state.state import AgentState, StateEntry, merge_tool_state

AOI = {"type": "FeatureCollection", "features": [{"id": "big-polygon"}]}
GEOJSON_SCHEMA = {"type": "object", "properties": {"type": {"type": "string"}}}


def remote_tool(
    name: str,
    *,
    properties: dict[str, Any],
    required: list[str],
    meta: dict[str, Any] | None = None,
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
        metadata={"_meta": meta} if meta else None,
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


async def test_a_handle_reaches_the_tool_as_the_value_it_named() -> None:
    """The whole mechanism, on a server that declares nothing at all."""
    seen: dict[str, Any] = {}
    clip = remote_tool(
        "clip_raster",
        properties={"dataset_id": {"type": "string"}, "aoi": GEOJSON_SCHEMA},
        required=["dataset_id", "aoi"],
        seen=seen,
    )
    key = "dataset-search/search_datasets/area_of_interest"

    await run_tools(
        [bind_injected(clip)],
        {
            "messages": [
                tool_call("clip_raster", {"dataset_id": "era5", "aoi": handle_for(key)})
            ],
            "tool_state": {key: StateEntry(value=AOI, tool="search_datasets", seq=1)},
        },
    )

    assert seen == {"dataset_id": "era5", "aoi": AOI}


async def test_a_literal_is_passed_through_untouched() -> None:
    """An ordinary parameter still takes an ordinary value."""
    seen: dict[str, Any] = {}
    clip = remote_tool(
        "clip_raster",
        properties={"aoi": GEOJSON_SCHEMA},
        required=["aoi"],
        seen=seen,
    )

    await run_tools(
        [bind_injected(clip)],
        {"messages": [tool_call("clip_raster", {"aoi": AOI})], "tool_state": {}},
    )

    assert seen == {"aoi": AOI}


async def test_a_tool_with_nothing_structured_is_untouched() -> None:
    """Nothing worth a handle: return it as it came, by identity."""
    plain = remote_tool("search", properties={"q": {"type": "string"}}, required=["q"])
    assert bind_injected(plain) is plain


async def test_binding_keeps_the_wrapped_tools_own_response_format() -> None:
    """A tool returning plain content still returns plain content once bound.

    Everything the adapter loads is ``content_and_artifact``, so assuming it on
    the wrapper holds right up until a locally defined tool is bound alongside
    them — and then every call fails unpacking a string as a two-tuple.
    """

    async def call(runtime: Any = None, **arguments: Any) -> Any:
        return f"got {len(arguments['payload'])} key(s)"

    local = StructuredTool(
        name="summarise",
        description="Summarise a payload.",
        args_schema={
            "type": "object",
            "properties": {"payload": {"type": "object"}},
            "required": ["payload"],
        },
        coroutine=call,
        response_format="content",
    )
    bound = bind_injected(local)

    assert bound.response_format == "content"
    result = await run_tools(
        [bound],
        {
            "messages": [tool_call("summarise", {"payload": {"a": 1, "b": 2}})],
            "tool_state": {},
        },
    )
    assert result["messages"][-1].content == "got 2 key(s)"


async def test_a_handle_naming_nothing_is_refused_with_the_options() -> None:
    seen: dict[str, Any] = {}
    clip = remote_tool(
        "clip_raster", properties={"aoi": GEOJSON_SCHEMA}, required=["aoi"], seen=seen
    )
    key = "dataset-search/search_datasets/area_of_interest"

    result = await run_tools(
        [bind_injected(clip)],
        {
            "messages": [tool_call("clip_raster", {"aoi": handle_for("nope")})],
            "tool_state": {key: StateEntry(value=AOI, tool="search_datasets", seq=1)},
        },
    )

    message = result["messages"][-1]
    assert message.status == "error"
    assert "no such key" in message.text
    assert key in message.text
    assert seen == {}


async def test_a_refusal_leaves_the_transcript_answerable() -> None:
    """The reason refusals are results and not exceptions.

    A model batching a publisher and its consumer into one assistant message is
    ordinary behaviour, and LangGraph runs both against the state as it stood
    at the *start* of the step — so the consumer cannot see the publication
    happening beside it and is refused.

    Raised, that ended the run and left an assistant message whose `tool_calls`
    no `ToolMessage` answered, which most providers reject: one such step made
    the thread unusable for every turn after it. Returned, every call has a
    result, the transcript stays well-formed, and the model can read the
    refusal and retry.
    """
    clip = remote_tool(
        "clip_raster",
        properties={"aoi": GEOJSON_SCHEMA},
        required=["aoi"],
        meta={NOT_AUTHORED_META_KEY: ["aoi"]},
    )
    result = await run_tools(
        [bind_injected(clip)],
        {"messages": [tool_call("clip_raster", {})], "tool_state": {}},
    )

    # Every call the assistant message made has an answer, which is the
    # property a provider checks and the raised version broke.
    answered = [message for message in result["messages"] if message.type == "tool"]
    assert [message.tool_call_id for message in answered] == ["1"]
    assert answered[0].status == "error"


async def test_the_wrapped_tools_own_errors_are_left_alone() -> None:
    """Only *our* refusals become results.

    `handle_tool_error` belongs to the whole tool, so switching it on for the
    refusals would also change how the tool underneath fails. A tool that
    declared nothing still raises.
    """

    async def explode(**_: Any) -> Any:
        raise ToolException("the server said no")

    exploding = StructuredTool(
        name="clip_raster",
        description="clip",
        args_schema={
            "type": "object",
            "properties": {"aoi": GEOJSON_SCHEMA},
            "required": ["aoi"],
        },
        coroutine=explode,
    )
    with pytest.raises(ToolException, match="the server said no"):
        await bind_injected(exploding).ainvoke(
            {
                "args": {"aoi": AOI},
                "id": "1",
                "type": "tool_call",
                "state": {"tool_state": {}},
            }
        )


async def test_a_tool_that_handles_its_own_errors_still_does() -> None:
    """An inherited `handle_tool_error` is honoured, not overwritten."""

    async def explode(**_: Any) -> Any:
        raise ToolException("the server said no")

    exploding = StructuredTool(
        name="clip_raster",
        description="clip",
        args_schema={"type": "object", "properties": {"aoi": GEOJSON_SCHEMA}},
        coroutine=explode,
        handle_tool_error="ask again later",
    )
    message = await bind_injected(exploding).ainvoke(
        {"args": {"aoi": AOI}, "id": "1", "type": "tool_call"}
    )

    assert message.content == "ask again later"


def test_a_refusal_is_its_own_exception_type() -> None:
    """So a host can tell "the binding said no" from "the tool failed"."""
    assert issubclass(StateRefusal, ToolException)


def test_bind_all_injected_maps_over_every_tool() -> None:
    tools = [
        remote_tool("a", properties={"q": {"type": "string"}}, required=["q"]),
        remote_tool("b", properties={"payload": {"type": "object"}}, required=[]),
    ]
    bound = bind_all_injected(tools)

    assert bound[0] is tools[0]  # nothing structured, nothing to do
    assert "anyOf" in bound[1].args_schema["properties"]["payload"]


def test_merge_stamps_write_order_so_recency_is_knowable() -> None:
    first = merge_tool_state({}, {"t/a/x": StateEntry(value=1)})
    second = merge_tool_state(first, {"t/b/y": StateEntry(value=2)})
    assert second["t/b/y"]["seq"] > second["t/a/x"]["seq"]
