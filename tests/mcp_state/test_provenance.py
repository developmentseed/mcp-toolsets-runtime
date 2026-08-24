"""What a call was given, recorded on every value it produced.

A tool owns its outputs, and nothing here second-guesses them. What the client
knows for certain is where each *argument* came from — a handle names a stored
value, anything else the model wrote — and that is what a state entry carries.

Recorded, never enforced: no call is refused on the strength of it. It exists
so a value the model authored is *visible* where it is later reused, which is
the one place a transcript cannot help because the call that made it has
scrolled away.
"""

from typing import Any

from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.types import Command

from mcp_runtime.declarations import PRODUCES_META_KEY
from mcp_state.handles import available, handle_for
from mcp_state.middleware import (
    DEFAULT_CAPTURE_BYTES,
    StateCaptureMiddleware,
    call_inputs,
    publications,
)
from mcp_state.state import MODEL_AUTHORED, TOOL_STATE_KEY, StateEntry, authored

AOI_KEY = "dataset-search/search/area_of_interest"
AOI = {"type": "FeatureCollection", "features": [{"id": "poly"}]}


def big(chars: int = DEFAULT_CAPTURE_BYTES + 100) -> str:
    return "x" * chars


def remote_tool(name: str, produces: list[dict[str, Any]] | None = None) -> Any:
    """A stand-in for a tool the adapter loaded from an MCP server."""

    async def call(**arguments: Any) -> Any:
        return "called", None

    return StructuredTool(
        name=name,
        description=name,
        args_schema={"type": "object", "properties": {}},
        coroutine=call,
        metadata={"_meta": {PRODUCES_META_KEY: produces}} if produces else None,
    )


async def capture(
    middleware: StateCaptureMiddleware,
    tool_name: str,
    payload: dict[str, Any],
    arguments: dict[str, Any] | None = None,
) -> ToolMessage | Command[Any]:
    """Run one tool return through the middleware, as one call's answer."""
    message = ToolMessage(
        content="raw",
        name=tool_name,
        tool_call_id="1",
        artifact={"structured_content": payload},
    )

    async def handler(_request: Any) -> ToolMessage:
        return message

    request = ToolCallRequest(
        tool_call={
            "name": tool_name,
            "args": arguments or {},
            "id": "1",
            "type": "tool_call",
        },
        tool=None,
        state={},
        runtime=None,  # type: ignore[arg-type]
    )
    return await middleware.awrap_tool_call(request, handler)


# --- reading one call's arguments ----------------------------------------


def test_a_handle_resolves_to_the_key_it_named() -> None:
    assert call_inputs({"aoi": handle_for(AOI_KEY)}) == {"aoi": AOI_KEY}


def test_anything_else_is_the_models_own() -> None:
    assert call_inputs({"dataset_id": "chirps", "n": 3, "box": [1, 2]}) == {
        "dataset_id": MODEL_AUTHORED,
        "n": MODEL_AUTHORED,
        "box": MODEL_AUTHORED,
    }


def test_a_call_with_no_arguments_records_nothing() -> None:
    """Absent is not the same claim as empty, so it stays absent."""
    assert call_inputs({}) == {}


def test_nothing_but_names_is_stored() -> None:
    """The record must stay cheap however large the call was."""
    written = call_inputs({"payload": {"huge": big()}})
    assert written == {"payload": MODEL_AUTHORED}
    assert big() not in str(written)


# --- what capture writes --------------------------------------------------


PUBLISHES = [
    {"stateKey": "raster-ops/clip/geometry", "field": "geometry"},
    {"stateKey": "raster-ops/clip/dataset", "field": "dataset"},
]


async def test_an_entry_records_what_its_producing_call_was_given() -> None:
    middleware = StateCaptureMiddleware(publications([remote_tool("clip", PUBLISHES)]))

    result = await capture(
        middleware,
        "clip",
        {"message": "clipped", "geometry": AOI},
        {"aoi": handle_for(AOI_KEY), "dataset_id": "chirps"},
    )

    assert isinstance(result, Command)
    entry = result.update[TOOL_STATE_KEY]["raster-ops/clip/geometry"]
    assert entry["inputs"] == {"aoi": AOI_KEY, "dataset_id": MODEL_AUTHORED}


async def test_every_field_of_one_return_carries_the_same_record() -> None:
    """It describes the *call*. Which output derives from which input is the
    tool's business, and the tool does not say."""
    middleware = StateCaptureMiddleware(publications([remote_tool("clip", PUBLISHES)]))

    result = await capture(
        middleware,
        "clip",
        {"message": "clipped", "geometry": AOI, "dataset": "chirps"},
        {"aoi": handle_for(AOI_KEY), "dataset_id": "chirps"},
    )

    assert isinstance(result, Command)
    written = result.update[TOOL_STATE_KEY]
    assert (
        written["raster-ops/clip/geometry"]["inputs"]
        == written["raster-ops/clip/dataset"]["inputs"]
    )


async def test_a_no_argument_call_leaves_the_field_off() -> None:
    middleware = StateCaptureMiddleware(publications([remote_tool("clip", PUBLISHES)]))

    result = await capture(middleware, "clip", {"message": "ok", "geometry": AOI})

    assert isinstance(result, Command)
    assert "inputs" not in result.update[TOOL_STATE_KEY]["raster-ops/clip/geometry"]


async def test_a_size_captured_value_records_it_too() -> None:
    """A server that declared nothing still had its call's arguments read."""
    middleware = StateCaptureMiddleware(publications([remote_tool("terrain")]))

    result = await capture(
        middleware, "terrain", {"message": "sampled", "samples": big()}, {"n": 400}
    )

    assert isinstance(result, Command)
    entry = result.update[TOOL_STATE_KEY]["terrain/samples"]
    assert entry["inputs"] == {"n": MODEL_AUTHORED}


# --- the laundering case, which is the point ------------------------------


async def test_a_value_the_tool_merely_echoed_is_visible_as_the_models() -> None:
    """The hole `NotAuthored` alone leaves open.

    A model writes a bounding box, passes it to a tool with an ordinary
    parameter, and the tool returns it. Captured, it wears that tool's name and
    is otherwise indistinguishable from one the tool computed. The record of
    what the call was given is what tells them apart.
    """
    middleware = StateCaptureMiddleware(
        publications(
            [
                remote_tool(
                    "get_aoi", [{"stateKey": "gazet/get_aoi/bbox", "field": "bbox"}]
                )
            ]
        )
    )
    invented = [12.4, 55.6, 12.7, 55.8]

    result = await capture(
        middleware, "get_aoi", {"message": "ok", "bbox": invented}, {"bbox": invented}
    )

    assert isinstance(result, Command)
    entry = result.update[TOOL_STATE_KEY]["gazet/get_aoi/bbox"]
    assert entry["tool"] == "get_aoi"  # still says a tool published it
    assert authored(entry) == ["bbox"]  # and now also says the model wrote it


async def test_a_derived_value_says_nothing_of_the_sort() -> None:
    """The same tool called with a place name: the output is its own work."""
    middleware = StateCaptureMiddleware(
        publications(
            [
                remote_tool(
                    "get_aoi", [{"stateKey": "gazet/get_aoi/bbox", "field": "bbox"}]
                )
            ]
        )
    )

    result = await capture(
        middleware,
        "get_aoi",
        {"message": "ok", "bbox": [12.4, 55.6, 12.7, 55.8]},
        {"place": "Copenhagen"},
    )

    assert isinstance(result, Command)
    # `place` was model-authored, and that is what the record says — the client
    # never claims the *output* is or is not derived.
    assert authored(result.update[TOOL_STATE_KEY]["gazet/get_aoi/bbox"]) == ["place"]


async def test_a_model_authored_value_is_still_accepted() -> None:
    """Recorded, not enforced.

    Refusing here was considered and rejected: it turns visibility into a
    guarantee at the price of a tool going uncallable whenever its only
    producer was itself called with a model-authored argument, and that lands
    on a user with no way to clear it. A wrong value they can see beats a right
    one they cannot obtain.
    """
    from langchain_core.messages import AIMessage
    from langgraph.graph import END, START, StateGraph
    from langgraph.prebuilt.tool_node import ToolNode

    from mcp_runtime.declarations import NOT_AUTHORED_META_KEY
    from mcp_state.injection import bind_injected
    from mcp_state.state import AgentState

    seen: dict[str, Any] = {}

    async def call(runtime: Any = None, **arguments: Any) -> Any:
        seen.update(arguments)
        return "called", None

    narrowed = StructuredTool(
        name="submit",
        description="submit",
        args_schema={
            "type": "object",
            "properties": {"area": {"type": "array"}},
            "required": ["area"],
        },
        coroutine=call,
        response_format="content_and_artifact",
        metadata={"_meta": {NOT_AUTHORED_META_KEY: ["area"]}},
    )
    # Laundered: the model wrote this bbox, a tool echoed it, and here it is.
    laundered = StateEntry(
        value=[12.4, 55.6, 12.7, 55.8],
        tool="get_aoi",
        seq=1,
        inputs={"bbox": MODEL_AUTHORED},
    )

    graph = StateGraph(AgentState)
    graph.add_node("tools", ToolNode([bind_injected(narrowed)]))
    graph.add_edge(START, "tools")
    graph.add_edge("tools", END)
    await graph.compile().ainvoke(
        {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "submit",
                            "args": {"area": handle_for("gazet/get_aoi/bbox")},
                            "id": "1",
                            "type": "tool_call",
                        }
                    ],
                )
            ],
            "tool_state": {"gazet/get_aoi/bbox": laundered},
        }
    )

    assert seen["area"] == [12.4, 55.6, 12.7, 55.8]


# --- reading it back ------------------------------------------------------


def test_authored_names_only_the_models_own_and_sorts_them() -> None:
    entry = StateEntry(
        value=1, inputs={"z": MODEL_AUTHORED, "aoi": AOI_KEY, "a": MODEL_AUTHORED}
    )
    assert authored(entry) == ["a", "z"]


def test_authored_is_empty_where_nothing_was_recorded() -> None:
    assert authored(StateEntry(value=1)) == []
    assert authored(None) == []


def test_provenance_is_a_chain_the_reader_may_walk() -> None:
    """Each recorded input names either the model or another key, so the
    history is a walk over facts rather than a propagated flag."""
    state = {
        AOI_KEY: StateEntry(value=AOI, tool="search", seq=1, inputs={"query": "model"}),
        "raster-ops/clip/geometry": StateEntry(
            value=AOI,
            tool="clip",
            seq=2,
            inputs={"aoi": AOI_KEY, "dataset_id": MODEL_AUTHORED},
        ),
    }

    # One level: the clip rests on a model-authored dataset_id.
    assert authored(state["raster-ops/clip/geometry"]) == ["dataset_id"]
    # And the reader can follow `aoi` to the entry beneath it, which is its own
    # fact rather than one this one inherited.
    assert authored(state[state["raster-ops/clip/geometry"]["inputs"]["aoi"]]) == [
        "query"
    ]


def test_the_listing_names_what_the_model_wrote() -> None:
    listed = available(
        {
            "gazet/get_aoi/bbox": StateEntry(
                value=[1, 2, 3, 4], tool="get_aoi", seq=1, inputs={"bbox": "model"}
            )
        }
    )
    assert listed == [
        "@state:gazet/get_aoi/bbox — 4 item(s), from get_aoi (you wrote: bbox)"
    ]


def test_a_call_that_drew_on_state_names_nothing_it_also_wrote() -> None:
    """The note warns that a value has no tool-found input behind it, so a
    call that had one says nothing — including about the scalars beside it.

    Filtering the arguments instead would invert it: this call would carry six
    names and a value invented from one argument would carry one, marking the
    trustworthy value as the more suspect of the two.
    """
    listed = available(
        {
            "cds/submit_request/job": StateEntry(
                value={"job_id": "x"},
                tool="submit_request",
                seq=1,
                inputs={
                    "area": AOI_KEY,
                    "dataset": MODEL_AUTHORED,
                    "variable": MODEL_AUTHORED,
                    "year": MODEL_AUTHORED,
                    "month": MODEL_AUTHORED,
                    "day": MODEL_AUTHORED,
                    "time": MODEL_AUTHORED,
                },
            )
        }
    )

    assert listed == [
        "@state:cds/submit_request/job — object with 1 key(s), from submit_request"
    ]


def test_a_wholly_invented_call_is_summarised_past_a_few_arguments() -> None:
    """Naming them is the signal while there are few enough to read; past that
    the names are what stops the rest of the line being read."""
    listed = available(
        {
            "search/run/results": StateEntry(
                value=[1],
                tool="run",
                seq=1,
                inputs=dict.fromkeys("abcd", MODEL_AUTHORED),
            )
        }
    )

    assert listed == [
        "@state:search/run/results — 1 item(s), from run "
        "(you wrote every argument: 4 of them)"
    ]


def test_the_listing_stays_quiet_where_there_is_nothing_to_say() -> None:
    """A parameter filled from state is the unremarkable case, and saying so
    would cost tokens on every line of every refusal."""
    listed = available(
        {
            "raster-ops/clip/geometry": StateEntry(
                value=[1], tool="clip", seq=1, inputs={"aoi": AOI_KEY}
            )
        }
    )
    assert listed == ["@state:raster-ops/clip/geometry — 1 item(s), from clip"]
