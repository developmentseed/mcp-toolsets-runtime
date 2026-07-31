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

from mcp_runtime.declarations import CONSUMES_META_KEY, PRODUCES_META_KEY
from mcp_runtime.kinds import GEOJSON_AREA_OF_INTEREST
from mcp_state.handles import dereference, dereference_with_receipts, handle_for
from mcp_state.injection import bind_injected
from mcp_state.middleware import StateCaptureMiddleware, publications
from mcp_state.receipts import (
    BY_DECLARATION,
    BY_HANDLE,
    INJECTED_ARTIFACT_KEY,
    Receipt,
    breadcrumb,
    describe_receipt,
    receipts_of,
    supplied,
)
from mcp_state.state import StateEntry
from tests.mcp_state.test_injection import declaration, run_tools, tool_call

AOI = {"type": "FeatureCollection", "features": [{"id": "big-polygon"}]}
KEY = "dataset-search/geometry"
PUBLISHED = {KEY: StateEntry(value=AOI, kind=GEOJSON_AREA_OF_INTEREST, tool="search")}


def consumer(
    name: str = "clip_raster",
    *,
    properties: dict[str, Any] | None = None,
    consumes: list[dict[str, Any]] | None = None,
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
        metadata={"_meta": {CONSUMES_META_KEY: consumes}} if consumes else None,
    )


async def run(tool: StructuredTool, arguments: dict[str, Any]) -> ToolMessage:
    """Drive one bound tool through a real graph and return its message."""
    state = await run_tools(
        [tool],
        {"messages": [tool_call(tool.name, arguments)], "tool_state": dict(PUBLISHED)},
    )
    return state["messages"][-1]


# --- the record ----------------------------------------------------------


async def test_a_filled_parameter_records_where_its_value_came_from() -> None:
    """The whole point: which stored value the tool ran against, and whose."""
    bound = bind_injected(consumer(consumes=[declaration()]))

    message = await run(bound, {})

    assert receipts_of(message.artifact) == {
        "aoi": {
            "key": KEY,
            "via": BY_DECLARATION,
            "kind": GEOJSON_AREA_OF_INTEREST,
            "tool": "search",
        }
    }


async def test_a_handle_is_recorded_too_and_says_so() -> None:
    """Both paths land in the same place; ``via`` is what tells them apart."""
    bound = bind_injected(consumer("describe", properties={"g": {"type": "object"}}))

    message = await run(bound, {"g": handle_for(KEY)})

    assert receipts_of(message.artifact)["g"]["via"] == BY_HANDLE


async def test_an_explicitly_passed_value_earns_no_receipt() -> None:
    """Nothing came from state, so there is nothing to trace."""
    bound = bind_injected(consumer(consumes=[declaration()]))

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
    bound = bind_injected(consumer(consumes=[declaration()], response_format="content"))

    message = await run(bound, {})

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
        metadata={"_meta": {CONSUMES_META_KEY: [declaration()]}},
    )

    message = await run(bind_injected(tool), {})

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
        metadata={"_meta": {CONSUMES_META_KEY: [declaration()]}},
    )

    message = await run(bind_injected(tool), {})

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


async def test_the_model_is_told_which_stored_value_it_was_given() -> None:
    """Without this the model cannot describe, or correct, what it ran against."""
    bound = bind_injected(
        consumer(consumes=[declaration()], returns={"message": "clipped 3 rasters"})
    )

    message = await capture_of(await run(bound, {}))

    assert message.content == (
        "clipped 3 rasters\n\n[state used: aoi ← dataset-search/geometry, "
        "published by search]"
    )


async def test_a_consumer_returning_nothing_structured_still_reports() -> None:
    """Capture has nothing to do here, so it used to return before saying so."""
    bound = bind_injected(consumer(consumes=[declaration()]))

    message = await capture_of(await run(bound, {}))

    assert "[state used: aoi ← dataset-search/geometry" in message.content


async def test_a_third_party_return_capture_leaves_alone_still_reports() -> None:
    """Structured, but no ``message`` and nothing big enough to capture.

    Capture has no reason to touch this message, so it is the second of the two
    paths that return before saying anything — and a tool on an untouched
    third-party server is exactly the shape that takes it.
    """
    bound = bind_injected(
        consumer(consumes=[declaration()], returns={"vertices": 2000})
    )

    message = await capture_of(await run(bound, {}))

    # The result itself is untouched — only the note is added below it.
    assert message.content == (
        "called\n\n[state used: aoi ← dataset-search/geometry, published by search]"
    )
    assert message.artifact["structured_content"] == {"vertices": 2000}


async def test_a_handle_is_not_echoed_back_to_the_model() -> None:
    """The model wrote the key itself; repeating it buys nothing."""
    bound = bind_injected(consumer("describe", properties={"g": {"type": "object"}}))

    message = await capture_of(await run(bound, {"g": handle_for(KEY)}))

    assert "state used" not in str(message.content)
    assert receipts_of(message.artifact)["g"]["via"] == BY_HANDLE


async def test_both_directions_are_reported_on_one_message() -> None:
    """A tool that takes from state and publishes to it says so, in order."""
    tool = consumer(
        "reproject",
        consumes=[declaration()],
        returns={"message": "reprojected", "geometry": AOI},
    )
    tool.metadata = {
        "_meta": {
            CONSUMES_META_KEY: [declaration()],
            PRODUCES_META_KEY: [
                {
                    "stateKey": "raster-ops/geometry",
                    "field": "geometry",
                    "kind": GEOJSON_AREA_OF_INTEREST,
                }
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
    incoming = await run(bind_injected(tool), {})

    async def handler(_: ToolCallRequest) -> ToolMessage:
        return incoming

    result = await middleware.awrap_tool_call(request, handler)
    assert isinstance(result, Command)
    message = result.update["messages"][0]

    used = message.content.index("[state used:")
    written = message.content.index("[state updated:")
    assert used < written  # inputs before outputs
    # Rewriting the artifact must not lose the receipt a UI host reads.
    assert receipts_of(message.artifact)["aoi"]["key"] == KEY


# --- formatting ----------------------------------------------------------


def test_a_receipt_without_a_publisher_still_names_the_key() -> None:
    """``tool`` is optional on a state entry, so it is optional here."""
    assert (
        describe_receipt("aoi", Receipt(key=KEY, via=BY_DECLARATION)) == f"aoi ← {KEY}"
    )


def test_nothing_declared_means_no_note_at_all() -> None:
    assert breadcrumb({}) is None
    assert breadcrumb({"g": Receipt(key=KEY, via=BY_HANDLE)}) is None


def test_several_filled_parameters_are_listed_in_one_note() -> None:
    note = breadcrumb(
        {
            "aoi": Receipt(key=KEY, via=BY_DECLARATION, tool="search"),
            "bbox": Receipt(key="a/bbox", via=BY_DECLARATION),
        }
    )
    assert note == f"[state used: aoi ← {KEY}, published by search; bbox ← a/bbox]"


def test_a_host_is_shown_only_what_the_arguments_do_not_already_say() -> None:
    """A handle is in the arguments the model wrote; a declared fill is not."""
    receipts = {
        "aoi": Receipt(key=KEY, via=BY_DECLARATION),
        "g": Receipt(key=KEY, via=BY_HANDLE),
    }
    assert list(supplied(receipts, {"g": handle_for(KEY), "id": "era5"})) == ["aoi"]


def test_a_message_that_never_saw_state_reads_as_empty() -> None:
    """Callable on every tool result, so a host need not branch."""
    assert receipts_of(None) == {}
    assert receipts_of({"structured_content": {}}) == {}
    assert receipts_of("not an artifact") == {}
    assert receipts_of({INJECTED_ARTIFACT_KEY: {"aoi": {"no": "key"}}}) == {}
