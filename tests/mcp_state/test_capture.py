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

from mcp_runtime.declarations import PRODUCES_META_KEY
from mcp_state.inspect import make_inspect_state, read_state_key
from mcp_state.middleware import (
    CAPTURED_ARTIFACT_KEY,
    StateCaptureMiddleware,
    _breadcrumb,
    publications,
    restore_structured,
    state_keys,
)
from mcp_state.prompt import SESSION_STATE_PROMPT
from mcp_state.state import (
    MAX_TOOL_STATE_BYTES,
    TOOL_STATE_KEY,
    StateEntry,
    merge_tool_state,
)

AOI = {"type": "FeatureCollection", "features": [{"id": "polygon"}]}

PUBLISHES_GEOMETRY = [
    {"stateKey": "dataset-search/search/geometry", "field": "geometry"}
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


async def test_a_declared_key_lands_under_its_qualified_name() -> None:
    """The write carries everything a later reader needs to make sense of it."""
    middleware = StateCaptureMiddleware(
        publications([remote_tool("search", PUBLISHES_GEOMETRY)])
    )
    result = await capture(middleware, "search", {"message": "found", "geometry": AOI})
    assert isinstance(result, Command)
    entry = result.update[TOOL_STATE_KEY]["dataset-search/search/geometry"]
    assert entry["value"] == AOI
    assert entry["tool"] == "search"


async def test_the_payload_leaves_the_transcript_for_a_breadcrumb() -> None:
    """The point of capture: the model gets the message, not the megabytes."""
    middleware = StateCaptureMiddleware(
        publications([remote_tool("search", PUBLISHES_GEOMETRY)])
    )
    result = await capture(middleware, "search", {"message": "found", "geometry": AOI})
    assert isinstance(result, Command)
    (captured,) = result.update["messages"]
    assert "polygon" not in captured.content
    assert "dataset-search/search/geometry" in captured.content
    # The payload leaves the artifact as well as the content, replaced by a note
    # of where it went. The artifact never reached the model either way; this is
    # so a UI host can put the value back (test_restore_* below).
    assert captured.artifact["structured_content"] == {"message": "found"}
    assert captured.artifact[CAPTURED_ARTIFACT_KEY] == {
        "geometry": "dataset-search/search/geometry"
    }


async def test_a_ui_host_rebuilds_the_whole_return_from_message_and_state() -> None:
    """A view is written against the tool's return, not against what capture left."""
    middleware = StateCaptureMiddleware(
        publications([remote_tool("search", PUBLISHES_GEOMETRY)])
    )
    payload = {"message": "found", "geometry": AOI, "count": 3}
    result = await capture(middleware, "search", payload)
    assert isinstance(result, Command)
    (captured,) = result.update["messages"]
    restored = restore_structured(captured.artifact, result.update[TOOL_STATE_KEY])
    assert restored == payload  # including the small field capture left behind


async def test_restore_is_a_no_op_on_an_uncaptured_message() -> None:
    """So a host calls it on every result rather than branching on capture."""
    artifact = {"structured_content": {"message": "hi", "count": 3}}
    assert restore_structured(artifact, None) == {"message": "hi", "count": 3}
    assert restore_structured(None, None) is None


async def test_restore_omits_a_key_that_is_no_longer_in_state() -> None:
    """A bounded/pruned state must degrade to a partial view, not a KeyError."""
    artifact = {
        "structured_content": {"message": "found"},
        CAPTURED_ARTIFACT_KEY: {"geometry": "dataset-search/search/geometry"},
    }
    assert restore_structured(artifact, {}) == {"message": "found"}


async def test_a_secret_shaped_field_survives_on_neither_side() -> None:
    """The backstop has to cover the artifact too, or a UI host would receive it."""
    middleware = StateCaptureMiddleware(
        publications([remote_tool("search", PUBLISHES_GEOMETRY)])
    )
    result = await capture(
        middleware, "search", {"message": "found", "geometry": AOI, "api_key": "s3cret"}
    )
    assert isinstance(result, Command)
    (captured,) = result.update["messages"]
    assert "s3cret" not in captured.content
    assert "api_key" not in captured.artifact["structured_content"]
    assert not any("s3cret" in str(entry) for entry in result.update[TOOL_STATE_KEY])
    assert restore_structured(captured.artifact, result.update[TOOL_STATE_KEY]) == {
        "message": "found",
        "geometry": AOI,
    }


async def test_a_small_undeclared_value_stays_in_the_transcript() -> None:
    """Below the threshold there is nothing to save, so capture stays out of it."""
    middleware = StateCaptureMiddleware(publications([remote_tool("other")]))
    result = await capture(middleware, "other", {"message": "hi", "geometry": AOI})
    assert isinstance(result, ToolMessage)
    assert result.content == "hi"


# --- capture without any declaration at all -------------------------------


def big_geometry(vertices: int = 400) -> dict[str, Any]:
    """A FeatureCollection comfortably over the capture threshold."""
    ring = [[-3.0 + index / vertices, 51.0] for index in range(vertices)]
    return {
        "type": "FeatureCollection",
        "features": [{"geometry": {"type": "Polygon", "coordinates": [ring]}}],
    }


async def test_a_large_undeclared_value_is_captured_on_size_alone() -> None:
    """The claim that makes third-party servers work: no declaration needed."""
    middleware = StateCaptureMiddleware(publications([remote_tool("terrain")]))
    result = await capture(
        middleware, "terrain", {"message": "sampled", "coverage": big_geometry()}
    )
    assert isinstance(result, Command)
    entry = result.update[TOOL_STATE_KEY]["terrain/coverage"]
    assert entry["tool"] == "terrain"


async def test_an_undeclared_capture_keyed_by_its_server_when_known() -> None:
    """So the model reads one key shape whatever produced the value."""
    middleware = StateCaptureMiddleware(
        publications([remote_tool("terrain")]), owners={"terrain": "terrain-ops"}
    )
    result = await capture(
        middleware, "terrain", {"message": "sampled", "coverage": big_geometry()}
    )
    assert isinstance(result, Command)
    assert "terrain-ops/terrain/coverage" in result.update[TOOL_STATE_KEY]


async def test_undeclared_capture_can_be_switched_off() -> None:
    """A deployment that wants capture strictly as declared can have it."""
    middleware = StateCaptureMiddleware(
        publications([remote_tool("terrain")]), capture_undeclared=None
    )
    result = await capture(
        middleware, "terrain", {"message": "sampled", "coverage": big_geometry()}
    )
    assert isinstance(result, ToolMessage)


async def test_a_foreign_return_with_no_message_is_summarised_not_blanked() -> None:
    """Nothing said what to tell the model, so keep whatever was small enough."""
    middleware = StateCaptureMiddleware(publications([remote_tool("terrain")]))
    result = await capture(
        middleware, "terrain", {"region": "Severn", "coverage": big_geometry()}
    )
    assert isinstance(result, Command)
    (captured,) = result.update["messages"]
    assert "Severn" in captured.content
    assert "terrain/coverage" in captured.content
    assert "51.0" not in captured.content


async def test_a_secret_shaped_field_is_never_stored_however_it_is_declared() -> None:
    """The backstop for a toolset that should not have declared it at all."""
    middleware = StateCaptureMiddleware(
        publications(
            [
                remote_tool(
                    "auth",
                    [{"stateKey": "auth/login/api_key", "field": "api_key"}],
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
            "dataset-search/search/geometry": StateEntry(
                value=AOI, tool="search", seq=1
            )
        }
    }
    read = read_state_key("dataset-search/search/geometry", state)
    assert "FeatureCollection" in read
    assert "seq" not in read
    assert "tool" not in read


def test_inspect_and_capture_agree_on_the_key() -> None:
    """Every stored key is readable, declared or not.

    Capture writes a breadcrumb naming the key it just stored and telling the
    model to read it. A value captured *by size* has no declaration — that is
    the whole point of the undeclared path — so filtering reads by declaration
    would make that breadcrumb a lie for exactly those values.
    """
    published = publications([remote_tool("search", PUBLISHES_GEOMETRY)])
    allowed = state_keys(published)
    assert allowed == {"dataset-search/search/geometry"}

    state = {
        TOOL_STATE_KEY: {
            "dataset-search/search/geometry": StateEntry(value=AOI, seq=1),
            "foreign/samples": StateEntry(value=[1, 2, 3], seq=2),
        }
    }
    listing = read_state_key("*", state, allowed_keys=allowed)
    assert "dataset-search/search/geometry" in listing
    assert "foreign/samples" in listing
    assert "[1, 2, 3]" in read_state_key("foreign/samples", state, allowed_keys=allowed)


def test_a_handle_is_read_as_the_key_inside_it() -> None:
    """Observed four times in one session on mistral-small.

    The key is the argument here, so `@state:foo` cannot mean anything but
    `foo`. Refusing it produces a "no such key" answer that names the key the
    caller asked for, which reads as the value being gone.
    """
    state = {
        TOOL_STATE_KEY: {"dataset-search/search/geometry": StateEntry(value=AOI, seq=1)}
    }

    handled = read_state_key("@state:dataset-search/search/geometry", state)
    assert handled == read_state_key("dataset-search/search/geometry", state)
    assert "FeatureCollection" in handled


def test_a_handle_to_a_key_that_is_not_there_still_reports_the_bare_key() -> None:
    """The miss has to name what was looked for, or the model cannot correct it."""
    missing = read_state_key("@state:nobody/knows", {})
    assert '"unknown_or_empty_key": "nobody/knows"' in missing


def test_the_breadcrumb_names_the_keys_and_does_not_teach() -> None:
    """News about this call, and nothing that was true before it.

    How to use a key is a standing rule, so it is told once in the prompt
    rather than re-paid for on every capture — and a capture is the message a
    long run accumulates most of.
    """
    note = _breadcrumb(["dataset-search/search/geometry"])

    assert note == "[state updated: dataset-search/search/geometry]"


def test_the_prompt_is_where_using_a_key_is_taught() -> None:
    """The other half of the trade the breadcrumb makes.

    The note stopped saying how a key is used on the strength of this being
    said once at the top. Trim the prompt to nothing and a model would be told
    in neither place — which nothing else here would notice, since every test
    of a handle passes one directly rather than asking a model to write one.
    """
    # A read takes the bare key...
    assert "bare key" in SESSION_STATE_PROMPT
    assert "inspect_state" in SESSION_STATE_PROMPT
    # ...and a handle goes only where a schema accepts one.
    assert "@state:<key>" in SESSION_STATE_PROMPT
    assert "schema accepts it" in SESSION_STATE_PROMPT


def test_the_overwrite_note_hands_over_a_turn_number_and_nothing_else() -> None:
    """It says what happened. What to do about it is the argument's own job."""
    note = _breadcrumb(["dataset-search/search/geometry"], {2: [
        "dataset-search/search/geometry"
    ]})

    assert "replaces what dataset-search/search/geometry held at turn 2" in note
    assert "inspect_state" not in note


def test_inspect_state_documents_where_a_turn_number_comes_from() -> None:
    """The one place left that says what to do with the turn a note named.

    A description is what the model has in front of it at the moment it could
    act, so all three surfaces that produce a turn number are named in it. Two
    are quoted as this package emits them and the third is described, which is
    what the asserts below pin.
    """
    described = make_inspect_state(set()).description

    assert "replaces what" in described  # the capture breadcrumb
    assert "several turns wrote it" in described  # a read of the key
    assert "written in N turns" in described  # a refusal's listing
    # And the one thing a model gets wrong by default: answering anyway.
    assert "no longer retained" in described


def test_a_declared_key_not_yet_published_says_so() -> None:
    """Distinct from an unknown key: the answer is "run the producer", not "give up"."""
    allowed = state_keys(publications([remote_tool("search", PUBLISHES_GEOMETRY)]))
    missing = read_state_key("dataset-search/search/geometry", {}, allowed_keys=allowed)
    assert "has not published it yet" in missing

    unknown = read_state_key("nobody/knows", {}, allowed_keys=allowed)
    assert "unknown_or_empty_key" in unknown
    assert "has not published it yet" not in unknown


def test_state_is_bounded_and_keeps_the_newest() -> None:
    """Nothing else bounds this namespace, and capture writes on every call."""
    big = "x" * 400_000  # 20 of these exceed the 8 MB budget
    state: dict[str, StateEntry] = {}
    for turn in range(30):
        state = merge_tool_state(state, {f"tool/value-{turn}": StateEntry(value=big)})

    assert len(state) < 30, "unbounded: a long session would grow until it died"
    assert "tool/value-29" in state, "the most recent write must always survive"
    assert "tool/value-0" not in state, "the oldest write is the one to drop"
    # What survives is a contiguous run of the newest writes, in seq order.
    kept = sorted(int(key.rsplit("-", 1)[1]) for key in state)
    assert kept == list(range(kept[0], 30))


def test_a_single_oversized_value_is_still_stored() -> None:
    """Evicting it would leave the tool that just ran with nothing to show."""
    huge = StateEntry(value="x" * (MAX_TOOL_STATE_BYTES * 2))
    state = merge_tool_state({"tool/small": StateEntry(value="s")}, {"tool/huge": huge})
    assert "tool/huge" in state
