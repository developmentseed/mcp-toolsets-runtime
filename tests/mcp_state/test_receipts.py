"""What a tool received from state, recorded and reported.

The mirror of capture. ``test_capture.py`` covers the write direction; this is
the read direction — the record a filled parameter leaves behind, and the note
the model is shown in its place.
"""

from typing import Any

from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.types import Command

from mcp_runtime.declarations import NOT_AUTHORED_META_KEY, PRODUCES_META_KEY
from mcp_state.handles import dereference, dereference_with_receipts, handle_for
from mcp_state.injection import bind_injected
from mcp_state.middleware import StateCaptureMiddleware, publications
from mcp_state.receipts import (
    INJECTED_ARTIFACT_KEY,
    Receipt,
    describe_receipt,
    receipts_of,
)
from mcp_state.state import StateEntry
from tests.mcp_state.test_injection import run_tools, tool_call

AOI = {"type": "FeatureCollection", "features": [{"id": "big-polygon"}]}
KEY = "dataset-search/search/geometry"
PUBLISHED = {KEY: StateEntry(value=AOI, tool="search")}


def consumer(
    name: str = "clip_raster",
    *,
    properties: dict[str, Any] | None = None,
    narrowed: list[str] | None = None,
    returns: dict[str, Any] | None = None,
    response_format: str = "content_and_artifact",
) -> StructuredTool:
    """A tool from a server, optionally declaring what it takes from state."""

    async def call(runtime: Any = None, **arguments: Any) -> Any:
        if response_format == "content":
            return "called"
        return "called", ({"structured_content": returns} if returns else None)

    properties = properties or {"aoi": {"type": "object"}}
    return StructuredTool(
        name=name,
        description=name,
        args_schema={
            "type": "object",
            "properties": properties,
            "required": sorted(properties),
        },
        coroutine=call,
        response_format=response_format,  # type: ignore[arg-type]
        metadata=({"_meta": {NOT_AUTHORED_META_KEY: narrowed}} if narrowed else None),
    )


async def run(tool: StructuredTool, arguments: dict[str, Any]) -> ToolMessage:
    """Drive one bound tool through a real graph and return its message."""
    state = await run_tools(
        [tool],
        {"messages": [tool_call(tool.name, arguments)], "tool_state": dict(PUBLISHED)},
    )
    return state["messages"][-1]


# --- the record ----------------------------------------------------------


async def test_a_handle_records_where_its_value_came_from() -> None:
    """The key is in the transcript; who published it is not, so this is."""
    bound = bind_injected(consumer())

    message = await run(bound, {"aoi": handle_for(KEY)})

    assert receipts_of(message.artifact) == {"aoi": Receipt(key=KEY, tool="search")}


async def test_an_explicitly_passed_value_earns_no_receipt() -> None:
    """Nothing came out of state, so there is nothing to record."""
    bound = bind_injected(consumer())

    message = await run(bound, {"aoi": {"type": "FeatureCollection", "features": []}})

    assert receipts_of(message.artifact) == {}


async def test_a_tool_that_takes_nothing_from_state_is_left_alone() -> None:
    """No receipts key at all, rather than an empty one on every message."""
    bound = bind_injected(consumer("describe", properties={"g": {"type": "object"}}))

    message = await run(bound, {"g": {"type": "Polygon"}})

    assert INJECTED_ARTIFACT_KEY not in (message.artifact or {})


async def test_the_wrapped_tools_return_shape_is_never_changed() -> None:
    """A ``content``-only tool has no artifact, so it keeps returning content.

    Its receipts go unrecorded. Changing the shape a tool declared would break
    it outright, which is a worse trade than losing the record.
    """
    bound = bind_injected(consumer(response_format="content"))

    message = await run(bound, {"aoi": handle_for(KEY)})

    assert message.content == "called"
    assert message.artifact is None


async def test_a_content_only_pair_is_not_mistaken_for_an_artifact_slot() -> None:
    """A two-element return is a shape; ``response_format`` is what gives it meaning.

    Only a ``content_and_artifact`` tool has a second slot to write into. Going
    by shape alone would overwrite the second half of an ordinary tool's result
    with a receipt.
    """

    async def call(runtime: Any = None, **arguments: Any) -> Any:
        # A dict second element: shape alone cannot tell this from an artifact.
        return "bounds", {"north": 12.4, "south": 3.1}

    tool = StructuredTool(
        name="bounds",
        description="bounds",
        args_schema={
            "type": "object",
            "properties": {"aoi": {"type": "object"}},
            "required": ["aoi"],
        },
        coroutine=call,
        response_format="content",
        metadata={"_meta": {NOT_AUTHORED_META_KEY: ["aoi"]}},
    )

    message = await run(bind_injected(tool), {"aoi": handle_for(KEY)})

    assert "12.4" in str(message.content) and "3.1" in str(message.content)
    assert INJECTED_ARTIFACT_KEY not in str(message.content)
    assert message.artifact is None


async def test_an_artifact_that_is_not_a_mapping_is_left_as_it_is() -> None:
    """``content_and_artifact`` says there is a second slot, not what is in it.

    Everything the MCP adapter builds is a dict, but a locally defined tool may
    put anything there — and merging a receipt into a list raises.
    """

    async def call(runtime: Any = None, **arguments: Any) -> Any:
        return "called", ["a", "b"]

    tool = StructuredTool(
        name="listy",
        description="listy",
        args_schema={
            "type": "object",
            "properties": {"aoi": {"type": "object"}},
            "required": ["aoi"],
        },
        coroutine=call,
        response_format="content_and_artifact",
        metadata={"_meta": {NOT_AUTHORED_META_KEY: ["aoi"]}},
    )

    message = await run(bind_injected(tool), {"aoi": handle_for(KEY)})

    assert message.artifact == ["a", "b"]


async def test_a_missing_handle_leaves_no_receipt_behind() -> None:
    """It was never resolved, so nothing was supplied to record."""
    arguments, receipts = dereference_with_receipts(
        {"g": handle_for("nope/missing")}, PUBLISHED
    )
    assert arguments == {"g": handle_for("nope/missing")}
    assert receipts == {}


def test_dereference_still_answers_with_just_the_arguments() -> None:
    """The narrow form stays, for callers that only want the substitution."""
    assert dereference({"g": handle_for(KEY)}, PUBLISHED) == {"g": AOI}


# --- what the model is told ----------------------------------------------


async def capture_of(message: ToolMessage) -> ToolMessage:
    """Put a tool message through capture and return the rewritten one."""
    middleware = StateCaptureMiddleware(publications([]))
    request = ToolCallRequest(
        tool_call={"name": "clip_raster", "args": {}, "id": "1", "type": "tool_call"},
        tool=consumer(),
        state={},
        runtime=None,  # type: ignore[arg-type]
    )

    async def handler(_: ToolCallRequest) -> ToolMessage:
        return message

    result = await middleware.awrap_tool_call(request, handler)
    return result.update["messages"][0] if isinstance(result, Command) else result


async def test_nothing_is_echoed_back_to_the_model() -> None:
    """The model wrote the key itself; repeating it would buy nothing.

    The record still exists on the artifact, which is where a host reads it.
    """
    bound = bind_injected(consumer(returns={"message": "clipped 3 rasters"}))

    message = await capture_of(await run(bound, {"aoi": handle_for(KEY)}))

    assert message.content == "clipped 3 rasters"
    assert "state used" not in str(message.content)


async def test_a_consumer_returning_nothing_structured_still_records() -> None:
    """Capture has nothing to do here, and the receipt survives it anyway."""
    bound = bind_injected(consumer())

    message = await capture_of(await run(bound, {"aoi": handle_for(KEY)}))

    assert receipts_of(message.artifact)["aoi"]["key"] == KEY


async def test_a_third_party_return_capture_leaves_alone_still_records() -> None:
    """Structured, but no ``message`` and nothing big enough to capture.

    Capture has no reason to touch this message — and a tool on an untouched
    third-party server is exactly that shape.
    """
    bound = bind_injected(consumer(returns={"vertices": 2000}))

    message = await capture_of(await run(bound, {"aoi": handle_for(KEY)}))

    assert message.artifact["structured_content"] == {"vertices": 2000}
    assert receipts_of(message.artifact)["aoi"]["key"] == KEY


async def test_both_directions_are_reported_on_one_message() -> None:
    """A tool that takes from state and publishes to it says so, in order."""
    tool = consumer(
        "reproject",
        narrowed=["aoi"],
        returns={"message": "reprojected", "geometry": AOI},
    )
    tool.metadata = {
        "_meta": {
            NOT_AUTHORED_META_KEY: ["aoi"],
            PRODUCES_META_KEY: [
                {"stateKey": "raster-ops/reproject/geometry", "field": "geometry"}
            ],
        }
    }
    middleware = StateCaptureMiddleware(publications([tool]))
    request = ToolCallRequest(
        tool_call={"name": "reproject", "args": {}, "id": "1", "type": "tool_call"},
        tool=tool,
        state={},
        runtime=None,  # type: ignore[arg-type]
    )
    incoming = await run(bind_injected(tool), {"aoi": handle_for(KEY)})

    async def handler(_: ToolCallRequest) -> ToolMessage:
        return incoming

    result = await middleware.awrap_tool_call(request, handler)
    assert isinstance(result, Command)
    message = result.update["messages"][0]

    assert "[state updated: raster-ops/reproject/geometry" in message.content
    # Rewriting the artifact must not lose the receipt a UI host reads.
    assert receipts_of(message.artifact)["aoi"]["key"] == KEY


# --- formatting ----------------------------------------------------------


def test_a_receipt_without_a_publisher_still_names_the_key() -> None:
    """``tool`` is optional on a state entry, so it is optional here."""
    assert describe_receipt("aoi", Receipt(key=KEY)) == f"aoi ← {KEY}"


def test_a_receipt_with_a_publisher_names_it() -> None:
    assert describe_receipt("aoi", Receipt(key=KEY, tool="search")) == (
        f"aoi ← {KEY}, published by search"
    )


def test_a_message_that_never_saw_state_reads_as_empty() -> None:
    """Callable on every tool result, so a host need not branch."""
    assert receipts_of(None) == {}
    assert receipts_of({"structured_content": {}}) == {}
    assert receipts_of("not an artifact") == {}
    assert receipts_of({INJECTED_ARTIFACT_KEY: {"aoi": {"no": "key"}}}) == {}
