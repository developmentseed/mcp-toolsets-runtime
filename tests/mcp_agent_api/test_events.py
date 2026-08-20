"""``agui_events``: one turn mapped onto AG-UI.

The agent here is the real one from ``tests.mcp_agent.test_streaming`` — a real
graph, real tools, real session state — so what is asserted is the whole path
from a tool publishing a geometry to the wire event a browser would receive.
"""

import json
from typing import Any

from ag_ui.core import UserMessage
from ag_ui.encoder import EventEncoder
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.checkpoint.memory import InMemorySaver

from mcp_agent.main import with_session_state

from mcp_agent.streaming import stream_turn
from mcp_state.wiring import Unsatisfiable
from mcp_agent_api.events import (
    ANSWER_CITATIONS,
    MCP_VIEW,
    STATE_CONSUMED,
    STATE_NAMESPACE,
    STATE_PUBLISHED,
    TOOLS_WITHHELD,
    agui_events,
    state_metadata,
    state_patch,
)
from tests.mcp_agent.test_streaming import (
    AOI,
    STATE_KEY,
    StreamingScriptedModel,
    _agent,
    _tool_call,
)

VIEW_URI = "ui://raster-ops/map"


def _viewed(name: str) -> BaseTool:
    """A tool declaring a ``ui://`` bundle, as an MCP toolset's would."""

    async def call(**arguments: Any) -> str:
        return "ok"

    return StructuredTool(
        name=name,
        description=name,
        args_schema={"type": "object", "properties": {}},
        coroutine=call,
        metadata={"_meta": {"ui": {"resourceUri": VIEW_URI}}},
    )


def _withheld(name: str) -> Unsatisfiable:
    """One dropped tool, as ``partition_usable`` reports it.

    ``BuiltAgent.withheld`` is a list of these, not of names — a test passing
    strings would exercise a caller that does not exist.
    """
    return Unsatisfiable(
        tool=name,
        parameter="aoi",
        wants="geojson.AreaOfInterest",
        required=True,
        model_generatable=False,
    )


async def _events(agent: Any = None, **kwargs: Any) -> list:
    turn = stream_turn(agent or _agent(), "clip chirps", "t1")
    return [
        event
        async for event in agui_events(turn, thread_id="t1", run_id="r1", **kwargs)
    ]


def _types(events: list) -> list[str]:
    return [event.type.value for event in events]


def _applied(events: list, state: dict[str, Any] | None = None) -> dict[str, Any]:
    """Every ``STATE_DELTA`` applied in order, the way a client applies them.

    Twelve lines because that is all our patches need: one ``add`` of the whole
    namespace, then ``add`` and ``remove`` of single keys under it. Asserting on
    the result rather than on the operations is what makes these tests about the
    state a client ends up holding.
    """
    document = dict(state or {})
    for event in events:
        if event.type.value != "STATE_DELTA":
            continue
        for operation in event.delta:
            head, _, tail = operation["path"].lstrip("/").partition("/")
            key = tail.replace("~1", "/").replace("~0", "~")
            if not tail:
                document[head] = operation["value"]
            elif operation["op"] == "remove":
                document[head].pop(key, None)
            else:
                document[head][key] = operation["value"]
    return document


def _deltas(events: list) -> list:
    return [event for event in events if event.type.value == "STATE_DELTA"]


def _activities(events: list) -> dict[str, Any]:
    return {
        event.activity_type: event.content
        for event in events
        if event.type.value == "ACTIVITY_SNAPSHOT"
    }


async def test_the_run_is_framed_and_nothing_precedes_it():
    """The client's verifier rejects a stream whose first event is anything but
    RUN_STARTED — including an activity announcing withheld tools."""
    events = await _events(withheld=[_withheld("submit_request")])

    assert _types(events)[0] == "RUN_STARTED"
    assert _types(events)[-1] == "RUN_FINISHED"
    assert _types(events)[1] == "ACTIVITY_SNAPSHOT"


async def test_withheld_tools_are_announced_once():
    events = await _events(
        withheld=[_withheld("submit_request"), _withheld("download")]
    )

    content = _activities(events)[TOOLS_WITHHELD]
    # Each declaration in full: a client can name the parameter and the kind,
    # not just the tool it lost.
    assert [item["tool"] for item in content["tools"]] == [
        "submit_request",
        "download",
    ]
    assert content["tools"][0]["wants"] == "geojson.AreaOfInterest"
    assert content["tools"][0]["parameter"] == "aoi"
    assert "submit_request" in content["display"]
    assert _types(events).count("ACTIVITY_SNAPSHOT") == len(_activities(events))


async def test_nothing_is_announced_when_no_tool_was_withheld():
    events = await _events()

    assert TOOLS_WITHHELD not in _activities(events)


async def test_a_tool_call_is_a_full_lifecycle():
    events = await _events()

    kinds = _types(events)
    assert kinds.count("TOOL_CALL_START") == 2
    assert kinds.count("TOOL_CALL_ARGS") == 2
    assert kinds.count("TOOL_CALL_END") == 2
    assert kinds.count("TOOL_CALL_RESULT") == 2
    starts = [e for e in events if e.type.value == "TOOL_CALL_START"]
    assert [event.tool_call_name for event in starts] == ["search", "clip"]


async def test_arguments_go_out_as_json():
    events = await _events()

    args = [e for e in events if e.type.value == "TOOL_CALL_ARGS"]
    assert json.loads(args[0].delta) == {}


async def test_a_receipt_rides_an_activity_beside_its_tool_call():
    events = await _events()

    received = _activities(events)[STATE_CONSUMED]["received"]
    assert received["aoi"]["key"] == STATE_KEY
    assert received["aoi"]["via"] == "declaration"
    assert received["aoi"]["tool"] == "search"


async def test_the_receipt_carries_the_line_the_chainlit_host_shows():
    """A minimal client prints ``display`` and is done. It is ``step_input``'s
    output, so the wire and the bundled UI cannot drift apart."""
    events = await _events()

    display = _activities(events)[STATE_CONSUMED]["received"]["aoi"]["display"]
    assert display == (
        f"← {STATE_KEY} · geojson.AreaOfInterest · 1 feature(s), 0 vertices · from search"
    )


async def test_the_display_describes_the_value_a_previous_tool_published():
    """The shape in that line comes from session state, which only holds the
    geometry because an earlier tool put it there — so the receipt for a later
    tool depends on state accumulated across the turn, not on its own artifact.
    """
    events = await _events()

    display = _activities(events)[STATE_CONSUMED]["received"]["aoi"]["display"]
    assert "1 feature(s)" in display
    assert len(AOI["features"]) == 1


async def test_what_a_tool_published_gets_its_own_activity():
    events = await _events()

    published = _activities(events)[STATE_PUBLISHED]
    assert published["published"] == {"geometry": STATE_KEY}
    assert published["tool"] == "search"


async def test_history_closes_the_run_rather_than_opening_it():
    """A client renders its question the moment it is typed, and a snapshot is
    applied by dropping every local message it does not name. Sent up front,
    before this turn is checkpointed, it would take the question off the screen
    and leave the answer under nothing."""

    async def history() -> list[UserMessage]:
        return [UserMessage(id="u0", content="the question")]

    types = _types(await _events(history=history))

    assert types.count("MESSAGES_SNAPSHOT") == 1
    assert types[-2:] == ["MESSAGES_SNAPSHOT", "RUN_FINISHED"]


async def test_no_history_callable_sends_no_snapshot():
    """The pure layer stays usable by a consumer with no checkpointer to read."""
    assert "MESSAGES_SNAPSHOT" not in _types(await _events())


async def test_a_failed_turn_snapshots_nothing():
    """A turn that raised left the thread mid-write. Telling a client to adopt
    that as the transcript would hand it a state the server may itself drop."""

    async def boom() -> Any:
        raise RuntimeError("nope")
        yield  # pragma: no cover - never reached

    async def history() -> list[UserMessage]:  # pragma: no cover - not called
        raise AssertionError("history must not be read after a failure")

    types = _types(
        [
            event
            async for event in agui_events(
                boom(), thread_id="t1", run_id="r1", history=history
            )
        ]
    )

    assert types == ["RUN_STARTED", "RUN_ERROR"]


async def test_the_answer_is_labelled_with_the_id_the_thread_will_keep():
    """The whole reason a closing snapshot reconciles instead of rebuilding: the
    message this stream created and the message a readback returns are the same
    message, so a client keeps its copy in place."""
    agent = _agent()
    events = [
        event
        async for event in agui_events(
            stream_turn(agent, "clip chirps", "t1"), thread_id="t1", run_id="r1"
        )
    ]
    opened = [e for e in events if e.type.value == "TEXT_MESSAGE_START"]
    assert opened

    state = await agent.aget_state({"configurable": {"thread_id": "t1"}})
    stored = {
        message.id
        for message in state.values["messages"]
        if type(message).__name__ == "AIMessage"
    }
    assert {event.message_id for event in opened} <= stored


def test_a_state_key_is_escaped_before_it_becomes_a_pointer():
    """State keys are ``toolset/name`` and ``/`` is the pointer's own separator.
    Unescaped, ``gazet/candidates`` would address a member of a ``gazet`` object
    that does not exist, and the whole patch would be rejected."""
    operations = state_patch({}, {"gazet/candidates": {"bytes": 1}, "a~b": {}})

    assert [operation["path"] for operation in operations] == [
        f"/{STATE_NAMESPACE}/gazet~1candidates",
        f"/{STATE_NAMESPACE}/a~0b",
    ]


def test_state_that_did_not_move_says_nothing():
    """A tool writing one key of six should not re-announce the other five, and
    a turn whose state is unchanged should send no event at all."""
    before = {"a/one": {"bytes": 1}, "b/two": {"bytes": 2}}

    assert state_patch(before, before) == []
    assert state_patch(before, {**before, "b/two": {"bytes": 9}}) == [
        {"op": "add", "path": f"/{STATE_NAMESPACE}/b~1two", "value": {"bytes": 9}}
    ]


def test_a_key_that_left_state_is_removed_from_the_client():
    """Eviction is the only way a key leaves. Left in place it would name a
    value the state route now answers 404 for."""
    operations = state_patch({"a/one": {"bytes": 1}}, {})

    assert operations == [{"op": "remove", "path": f"/{STATE_NAMESPACE}/a~1one"}]


async def test_the_client_keeps_its_own_state_through_a_whole_run():
    """AG-UI's ``state`` is shared with the client, not ours to own. Every
    operation names a path inside our namespace, so a client's own keys survive
    a run untouched — which a snapshot, replacing the object, cannot promise."""
    events = await _events()

    mine = {"selectedLayer": "era5", "theme": "dark"}
    after = _applied(events, {**mine, "unrelated": {"deep": [1, 2]}})
    assert {key: after[key] for key in mine} == mine
    assert after["unrelated"] == {"deep": [1, 2]}
    assert STATE_KEY in after[STATE_NAMESPACE]


async def test_a_runs_first_delta_establishes_the_namespace_whole():
    """One ``add`` of the whole namespace, which cannot fail on an object and is
    what gives the per-key operations after it something to address. It is also
    the resynchronisation point: a client that lost track gets the map back at
    the top of every run."""
    events = await _events()

    first = _deltas(events)[0].delta
    assert [operation["op"] for operation in first] == ["add"]
    assert first[0]["path"] == f"/{STATE_NAMESPACE}"
    assert STATE_KEY in first[0]["value"]


async def test_a_second_runs_first_delta_names_the_whole_thread():
    """The namespace goes out whole at the top of each run, so it has to *be*
    whole. A run announcing only what this turn wrote would tell a client the
    thread had lost everything an earlier turn published."""
    agent = _agent(
        [
            _tool_call("search", "c1"),
            _tool_call("clip", "c2"),
            AIMessage(content="done"),
            # The second turn calls only the consumer, so it writes nothing of
            # its own and everything it announces came from the first.
            _tool_call("clip", "c3"),
            AIMessage(content="done"),
        ]
    )
    await _events(agent)
    events = [
        event
        async for event in agui_events(
            stream_turn(agent, "again", "t1"), thread_id="t1", run_id="r2"
        )
    ]

    assert STATE_KEY in _deltas(events)[0].delta[0]["value"]


async def test_state_deltas_carry_metadata_and_never_the_value():
    events = await _events()

    deltas = _deltas(events)
    assert deltas, "a tool published, so state changed"
    entry = _applied(events)[STATE_NAMESPACE][STATE_KEY]
    assert entry["kind"] == "geojson.AreaOfInterest"
    assert entry["tool"] == "search"
    assert entry["bytes"] > 0
    assert "value" not in entry
    assert "FeatureCollection" not in json.dumps([e.delta for e in deltas])


async def test_the_last_state_delta_is_the_merged_one():
    """Write order is assigned by the state reducer, so an entry taken from a
    mid-turn update has no ``seq`` — it is omitted rather than sent as null, and
    the delta closing the turn carries the real one."""
    events = await _events()

    deltas = _deltas(events)
    assert "seq" not in deltas[0].delta[0]["value"][STATE_KEY]
    assert _applied(events)[STATE_NAMESPACE][STATE_KEY]["seq"] == 1
    assert _types(events).index("STATE_DELTA") < _types(events).index(
        "TEXT_MESSAGE_START"
    )


async def test_the_final_delta_lands_before_the_run_closes():
    events = await _events()

    kinds = _types(events)
    assert kinds[-1] == "RUN_FINISHED"
    assert kinds[-2] == "STATE_DELTA"


async def test_a_view_is_announced_with_its_uri_not_its_bundle():
    """A bundle can be ~300 kB and would otherwise repeat every turn; the client
    fetches it once from the view route and caches it."""
    agent = _agent()
    events = await _events(agent, tools={"search": _viewed("search")})

    view = _activities(events)[MCP_VIEW]
    assert view["uri"] == VIEW_URI
    assert view["tool"] == "search"


async def test_a_tool_without_a_view_announces_none():
    events = await _events(tools={"search": _viewed("search")} and {})

    assert MCP_VIEW not in _activities(events)


async def test_a_call_with_nothing_to_render_announces_no_view():
    """A call that failed leaves no structured content — and neither does one
    that simply returned text. There is nothing for the bundle to draw either
    way, and announcing a view anyway gives a client a panel it can only render
    empty, which next to a retry of the same tool reads as a duplicate."""
    viewed = _viewed("show")
    agent, _ = with_session_state(
        StreamingScriptedModel(
            script=[_tool_call("show", "c1"), AIMessage(content="done")]
        ),
        [viewed],
        InMemorySaver(),
    )

    events = await _events(agent, tools={"show": viewed})

    assert _types(events).count("TOOL_CALL_RESULT") == 1, "the tool did run"
    assert MCP_VIEW not in _activities(events)


async def test_a_view_stays_beside_its_own_call():
    """Held for the state its tool wrote — but no further. A view released only
    at the answer would sit after every later tool call, which is the position
    the whole ordering rule exists to prevent."""
    events = await _events(
        _agent(), tools={"search": _viewed("search"), "clip": _viewed("clip")}
    )

    positions = [
        (event.type.value, getattr(event, "activity_type", None)) for event in events
    ]
    first_view = positions.index((("ACTIVITY_SNAPSHOT"), MCP_VIEW))
    second_call = [
        index for index, (kind, _) in enumerate(positions) if kind == "TOOL_CALL_START"
    ][1]
    assert first_view < second_call


async def test_a_view_carries_the_data_its_bundle_renders():
    """A view is written against its tool's own return, so it gets that return
    whole — including the fields capture moved into state, which is most of the
    reason it wanted them."""
    events = await _events(_agent(), tools={"search": _viewed("search")})

    view = _activities(events)[MCP_VIEW]
    assert view["data"]["message"] == "found 3"
    assert view["data"]["geometry"] == AOI


async def test_a_view_waits_for_the_state_its_own_tool_just_wrote():
    """The publishing tool's writes reach state on the event *after* the one
    that announced them. A view built and sent in the same breath is handed the
    fields small enough to have stayed on the message and silently loses the
    ones it exists to draw."""
    events = await _events(_agent(), tools={"search": _viewed("search")})

    kinds = _types(events)
    view = next(
        position
        for position, event in enumerate(events)
        if getattr(event, "activity_type", None) == MCP_VIEW
    )
    assert kinds[:view].count("STATE_DELTA") == 1
    # Still before the answer opens, which is the rule that matters to a client.
    assert "TEXT_MESSAGE_START" not in kinds[:view]


async def test_a_view_on_a_tool_that_published_nothing_is_still_sent():
    """``clip`` consumes and publishes, so nothing follows it on the state
    channel. With no write to wait for, the only thing that can release the
    view is the answer opening — or the turn ending without one."""
    events = await _events(_agent(), tools={"clip": _viewed("clip")})

    kinds = _types(events)
    view = next(
        position
        for position, event in enumerate(events)
        if getattr(event, "activity_type", None) == MCP_VIEW
    )
    assert _activities(events)[MCP_VIEW]["tool"] == "clip"
    assert view > kinds.index("TOOL_CALL_RESULT")
    assert view < kinds.index("TEXT_MESSAGE_START")


async def test_the_answer_streams_as_text_events():
    events = await _events()

    deltas = [e.delta for e in events if e.type.value == "TEXT_MESSAGE_CONTENT"]
    assert "".join(deltas) == "Clipped chirps to your area."
    assert _types(events).count("TEXT_MESSAGE_START") == 1
    assert _types(events).count("TEXT_MESSAGE_END") == 1


async def test_every_activity_for_a_call_precedes_the_answers_message():
    """The sharp constraint: a message's position is fixed when it is created,
    so an activity emitted after TEXT_MESSAGE_START renders *after* the answer
    however early it was sent."""
    events = await _events()

    kinds = _types(events)
    opened = kinds.index("TEXT_MESSAGE_START")
    for position, kind in enumerate(kinds):
        if kind == "ACTIVITY_SNAPSHOT":
            activity = events[position].activity_type
            if activity != ANSWER_CITATIONS:
                assert position < opened, f"{activity} would render after the answer"


async def test_citations_come_after_the_answer_because_that_is_where_they_belong():
    from langchain_core.messages import AIMessage

    agent = _agent(
        [
            AIMessage(
                content=[
                    {"type": "text", "text": "ERA5 covers it. "},
                    {
                        "type": "text",
                        "text": "See the catalogue.",
                        "reference": {"reference_ids": ["era5"]},
                    },
                ]
            )
        ]
    )
    events = await _events(agent)

    kinds = _types(events)
    content = _activities(events)[ANSWER_CITATIONS]
    assert content["ids"] == ["era5"]
    assert content["display"] == "Sources: era5"
    # A JSON object, never a bare list: ActivityMessage.content is a mapping.
    assert isinstance(content, dict)
    assert kinds.index("TEXT_MESSAGE_END") < len(kinds) - 2


async def test_no_activity_deltas_are_emitted():
    """Deltas fail silently client-side — an orphan is dropped without error —
    so every activity here is a snapshot complete in itself."""
    events = await _events(withheld=[_withheld("x")])

    assert "ACTIVITY_DELTA" not in _types(events)


async def test_a_failing_turn_ends_in_run_error_not_silence():
    async def explodes():
        raise RuntimeError("the toolset went away")
        yield  # pragma: no cover - never reached

    events = [
        event async for event in agui_events(explodes(), thread_id="t1", run_id="r1")
    ]

    assert _types(events) == ["RUN_STARTED", "RUN_ERROR"]
    assert events[-1].message == "the toolset went away"


async def test_each_new_message_gets_its_own_id():
    """Only the events that *create* a message: an answer's START/CONTENT/END
    share one id by design. Reusing an id across two creations would overwrite
    the first message, since a snapshot replaces by default."""
    events = await _events(withheld=[_withheld("x")])
    creates = {"ACTIVITY_SNAPSHOT", "TOOL_CALL_RESULT", "TEXT_MESSAGE_START"}

    ids = [event.message_id for event in events if event.type.value in creates]

    assert len(ids) == len(set(ids))
    assert len(ids) >= 4  # withheld, two tool results, the answer


def test_state_metadata_survives_a_value_that_will_not_serialise():
    class Opaque:
        __slots__ = ()

    metadata = state_metadata({"k": {"value": Opaque(), "kind": None, "tool": "t"}})
    assert metadata["k"]["tool"] == "t"
    assert metadata["k"]["bytes"] is not None  # default=str still measures it


async def test_the_whole_run_encodes_as_sse():
    """Every event has to survive the encoder — a payload that will not
    serialise fails at the socket, mid-run, where a client cannot recover."""
    events = await _events(
        withheld=[_withheld("x")], tools={"search": _viewed("search")}
    )
    encoder = EventEncoder()

    wire = "".join(encoder.encode(event) for event in events)

    assert wire.count("data: ") == len(events)
    for line in wire.splitlines():
        if line.startswith("data: "):
            json.loads(line.removeprefix("data: "))
    # camelCase on the wire, whatever the Python field names are.
    assert '"messageId"' in wire
    assert '"activityType"' in wire
    assert '"toolCallId"' in wire
