"""Capture and inspection: what a tool publishes, and how it is read back.

The two directions out of ``tool_state`` are covered together because they
have to agree on one thing — the qualified key — and disagreeing silently is
the failure worth guarding.
"""

from typing import Any

from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.types import Command

from mcp_runtime.injected import PRODUCES_META_KEY
from mcp_runtime.kinds import GEOJSON_AREA_OF_INTEREST
from mcp_state.inspect import read_state_key
from mcp_state.middleware import (
    StateCaptureMiddleware,
    published_keys,
    state_keys,
)
from mcp_state.state import TOOL_STATE_KEY, StateEntry

AOI = {"type": "FeatureCollection", "features": [{"id": "polygon"}]}

PUBLISHES_GEOMETRY = [
    {
        "stateKey": "dataset-search/geometry",
        "field": "geometry",
        "kind": GEOJSON_AREA_OF_INTEREST,
    }
]


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
    middleware: StateCaptureMiddleware, tool_name: str, payload: dict[str, Any]
) -> ToolMessage | Command[Any]:
    """Run one tool return through the middleware."""
    message = ToolMessage(
        content="raw",
        name=tool_name,
        tool_call_id="1",
        artifact={"structured_content": payload},
    )

    async def handler(_request: Any) -> ToolMessage:
        return message

    request = ToolCallRequest(
        tool_call={"name": tool_name, "args": {}, "id": "1", "type": "tool_call"},
        tool=None,
        state={},
        runtime=None,  # type: ignore[arg-type]
    )
    return await middleware.awrap_tool_call(request, handler)


async def test_a_declared_key_lands_under_its_qualified_name_with_its_kind() -> None:
    """The write carries everything injection later resolves on."""
    middleware = StateCaptureMiddleware(
        published_keys([remote_tool("search", PUBLISHES_GEOMETRY)])
    )
    result = await capture(middleware, "search", {"message": "found", "geometry": AOI})
    assert isinstance(result, Command)
    entry = result.update[TOOL_STATE_KEY]["dataset-search/geometry"]
    assert entry["value"] == AOI
    assert entry["kind"] == GEOJSON_AREA_OF_INTEREST
    assert entry["tool"] == "search"


async def test_the_payload_leaves_the_transcript_for_a_breadcrumb() -> None:
    """The point of capture: the model gets the message, not the megabytes."""
    middleware = StateCaptureMiddleware(
        published_keys([remote_tool("search", PUBLISHES_GEOMETRY)])
    )
    result = await capture(middleware, "search", {"message": "found", "geometry": AOI})
    assert isinstance(result, Command)
    (captured,) = result.update["messages"]
    assert "polygon" not in captured.content
    assert "dataset-search/geometry" in captured.content
    assert captured.artifact is None


async def test_an_undeclared_server_publishes_nothing() -> None:
    """A third-party return that merely looks like a ToolResult is left alone."""
    middleware = StateCaptureMiddleware(published_keys([remote_tool("other")]))
    result = await capture(middleware, "other", {"message": "hi", "geometry": AOI})
    assert isinstance(result, ToolMessage)
    assert result.content == "hi"


async def test_a_secret_shaped_field_is_never_stored_however_it_is_declared() -> None:
    """The backstop for a toolset that should not have declared it at all."""
    middleware = StateCaptureMiddleware(
        published_keys(
            [
                remote_tool(
                    "auth",
                    [{"stateKey": "auth/api_key", "field": "api_key", "kind": None}],
                )
            ]
        )
    )
    result = await capture(middleware, "auth", {"message": "ok", "api_key": "sk-live"})
    assert isinstance(result, ToolMessage)
    assert "sk-live" not in result.content


def test_inspect_reads_through_the_envelope() -> None:
    """A stored value is read as its value, not as its StateEntry wrapper."""
    state = {
        TOOL_STATE_KEY: {
            "dataset-search/geometry": StateEntry(
                value=AOI, kind=GEOJSON_AREA_OF_INTEREST, tool="search", seq=1
            )
        }
    }
    read = read_state_key("dataset-search/geometry", state)
    assert "FeatureCollection" in read
    assert "seq" not in read
    assert "kind" not in read


def test_inspect_and_capture_agree_on_the_key() -> None:
    """The keys the model may name are exactly the ones tools publish."""
    published = published_keys([remote_tool("search", PUBLISHES_GEOMETRY)])
    allowed = state_keys(published)
    assert allowed == {"dataset-search/geometry"}

    state = {
        TOOL_STATE_KEY: {
            "dataset-search/geometry": StateEntry(value=AOI, seq=1),
            "sneaky/undeclared": StateEntry(value="hidden", seq=2),
        }
    }
    listing = read_state_key("*", state, allowed_keys=allowed)
    assert "dataset-search/geometry" in listing
    assert "sneaky/undeclared" not in listing
