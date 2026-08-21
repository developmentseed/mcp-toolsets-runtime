"""One turn, as AG-UI events.

A pure async generator: :func:`~mcp_agent.streaming.stream_turn` in, AG-UI
events out. It imports no FastAPI and opens no socket, so a consumer driving the
agent over a websocket, a queue or its own framework can use it as-is;
:mod:`mcp_agent_api.routes` is only one caller.

AG-UI natively covers the answer, token streaming, tool calls and state. What it
has no vocabulary for is the two things this runtime exists to make visible —
**where a tool's arguments came from** and **which view renders its result** —
so both ride ``ACTIVITY_*``. An activity *is* a message in AG-UI's model, so a
client rendering messages in order shows a receipt in the right place with no
correlation code, which a ``CUSTOM`` event out of band could not do.

Every activity carries both a structured payload and a ``display`` string. A
minimal client prints ``display``; a bespoke one styles the fields. The string
is :func:`mcp_agent.host.step_input`'s, so the wire says exactly what the
bundled Chainlit host shows.

Four rules of the protocol shape this loop, each verified against
``@ag-ui/client``'s own verifier and message-applying pipeline rather than read
off the specification:

1. **Nothing may precede ``RUN_STARTED``** — the verifier rejects the stream.
2. **A message's position is fixed when it is first created, not completed.** An
   activity emitted between ``TEXT_MESSAGE_START`` and ``TEXT_MESSAGE_END`` is
   accepted, but the assistant message already exists, so the activity renders
   *after* the answer. Everything belonging to a tool call is therefore emitted
   before the answer's text message opens. Citations are the deliberate
   exception: they belong after the answer and are only known then.
3. **Deltas fail silently.** An ``ACTIVITY_DELTA`` for an unknown ``messageId``
   is dropped without error, and a state patch that fails to apply is a console
   warning and no more. Every message event here is therefore a snapshot,
   complete in itself. State is the one delta, because there the alternative is
   worse: a snapshot replaces the whole ``state`` object, including the keys the
   client keeps in it. It is made safe by construction — each run opens by
   adding :data:`STATE_NAMESPACE` whole, which cannot fail on an object and
   leaves a run's later per-key operations something to address.
4. **Activity content must be a JSON object.** ``ActivityMessage.content`` is a
   mapping, so a bare list would not survive being read back — citations go out
   as ``{"ids": [...]}``, never as an array.
"""

import json
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from typing import Any

from ag_ui.core import (
    ActivitySnapshotEvent,
    BaseEvent,
    Message,
    MessagesSnapshotEvent,
    RunErrorEvent,
    RunFinishedEvent,
    RunStartedEvent,
    StateDeltaEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallResultEvent,
    ToolCallStartEvent,
)
from langchain_core.tools import BaseTool

from mcp_agent.host import step_input, view_uri_for
from mcp_agent.streaming import (
    AnswerChunk,
    StateChanged,
    ToolFinished,
    ToolStarted,
    TurnEvent,
    TurnFinished,
)
from mcp_state import restore_structured
from mcp_state.state import StateEntry

#: ``activityType`` values this module emits. A client switches on these; they
#: are part of the wire contract, so they are named rather than inlined.
STATE_CONSUMED = "state.consumed"
STATE_PUBLISHED = "state.published"
MCP_VIEW = "mcp.view"
ANSWER_CITATIONS = "answer.citations"

#: Key inside AG-UI's ``state`` object under which session-state metadata is
#: carried, and the only path this server ever writes.
#:
#: The object is the *client's* as much as ours — the protocol has it hold
#: whatever a client and agent share. Writing our map at the root leaves a
#: client nowhere to keep its own keys; under a namespace both fit.
STATE_NAMESPACE = "toolState"


def _pointer(key: str) -> str:
    """A ``tool_state`` key as a JSON Pointer under :data:`STATE_NAMESPACE`.

    Keys are ``toolset/name`` and ``/`` is the pointer's own separator, so an
    unescaped ``gazet/candidates`` would address a ``candidates`` member of a
    ``gazet`` object that does not exist. RFC 6901 escaping, ``~`` before ``/``
    because the second escape introduces a ``~``.
    """
    return f"/{STATE_NAMESPACE}/" + key.replace("~", "~0").replace("/", "~1")


def state_patch(
    previous: Mapping[str, Any] | None, current: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """JSON Patch operations taking a client's state from *previous* to *current*.

    Every operation is confined to :data:`STATE_NAMESPACE`, which is what makes
    the state object shareable: a client's own keys sit beside ours and are
    never named, so nothing this server sends can disturb them.

    ``previous`` is ``None`` before anything has been announced on a run, and
    the whole namespace goes out as one ``add``. ``add`` on an object member
    replaces or creates that member and leaves its siblings alone, so it both
    establishes the namespace a later per-key operation needs and resynchronises
    it wholesale — the one thing a delta stream needs to be able to do.

    Nothing is emitted for state that has not moved: the empty list means the
    caller sends no event at all.
    """
    if previous is None:
        return [{"op": "add", "path": f"/{STATE_NAMESPACE}", "value": dict(current)}]
    removed = [
        {"op": "remove", "path": _pointer(key)}
        for key in previous
        if key not in current
    ]
    changed = [
        {"op": "add", "path": _pointer(key), "value": entry}
        for key, entry in current.items()
        if previous.get(key) != entry
    ]
    return removed + changed


def state_metadata(state: Mapping[str, StateEntry] | None) -> dict[str, Any]:
    """``tool_state`` with the values replaced by what describes them.

    A stored value is in state precisely because it is too big for a transcript,
    and a state snapshot on every tool call is no better a place for it. A
    frontend that wants the geometry fetches it from the state route; what it
    needs here is enough to decide whether to.

    ``seq`` is omitted rather than sent as null when it is not yet known. It is
    assigned by the state reducer when the write is merged, so an entry taken
    from a mid-turn update does not carry one — and a client ordering by it
    would be sorting nulls. The turn's closing update is built from the merged
    state and does carry it. ``inputs`` is omitted the same way, and for the
    same reason: absent means the producing call took no arguments, which is
    not the same claim as an empty object.
    """
    return {
        key: {
            "tool": entry.get("tool"),
            "bytes": _rough_size(entry.get("value")),
            **({} if entry.get("seq") is None else {"seq": entry["seq"]}),
            **({} if not entry.get("inputs") else {"inputs": entry["inputs"]}),
        }
        for key, entry in (state or {}).items()
    }


def _rough_size(value: Any) -> int | None:
    """Serialised size, or ``None`` when the value will not serialise.

    Indicative rather than exact — it is a hint for a client deciding whether to
    fetch, not an allocation.
    """
    try:
        return len(json.dumps(value, default=str))
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None


def _activity(
    message_id: str, activity_type: str, content: dict[str, Any]
) -> BaseEvent:
    return ActivitySnapshotEvent(
        message_id=message_id, activity_type=activity_type, content=content
    )


def consumed_content(
    arguments: Mapping[str, Any],
    finished: ToolFinished,
    state: Mapping[str, StateEntry] | None,
) -> dict[str, Any] | None:
    """The ``state.consumed`` payload, or ``None`` when nothing was supplied.

    Shared by the live stream and by a thread readback, so a receipt rendered
    now and the same receipt rendered after a reload cannot say different
    things. ``arguments`` are the model's, from the call this answers.
    """
    if not finished.received:
        return None
    rendered = step_input(dict(arguments), finished, dict(state or {}))
    return {
        "toolCallId": finished.id,
        "tool": finished.name,
        "received": {
            parameter: {**receipt, "display": str(rendered.get(parameter, ""))}
            for parameter, receipt in finished.received.items()
        },
    }


def published_content(finished: ToolFinished) -> dict[str, Any] | None:
    """The ``state.published`` payload, or ``None`` when the tool wrote nothing."""
    if not finished.published:
        return None
    return {
        "toolCallId": finished.id,
        "tool": finished.name,
        "published": dict(finished.published),
        "display": "→ " + ", ".join(sorted(finished.published.values())),
    }


def view_content(
    finished: ToolFinished, uri: str, state: Mapping[str, StateEntry] | None
) -> dict[str, Any] | None:
    """The ``mcp.view`` payload, or ``None`` when there is nothing to render.

    A view is written against its tool's own return, which
    :func:`~mcp_state.restore_structured` rebuilds from the artifact plus
    whatever capture moved into state. ``None`` is what a tool call that raised
    leaves behind: no structured content, so no view — announcing one anyway
    gives a client a panel it can only draw empty, beside the error explaining
    why.

    **The state has to be the state after that tool's own writes landed.** They
    reach the graph on the event *after* the tool finishes, which is why the
    live stream holds a view back rather than filling it where it is built.
    """
    data = restore_structured(finished.artifact, dict(state or {}))
    if data is None:
        return None
    return {
        "toolCallId": finished.id,
        "tool": finished.name,
        "uri": uri,
        "display": f"view: {uri}",
        "data": data,
    }


def citations_content(citations: Sequence[str]) -> dict[str, Any] | None:
    """The ``answer.citations`` payload, or ``None`` when the answer cited none.

    A mapping rather than a bare list: ``ActivityMessage.content`` is an object,
    so an array would not survive being read back.
    """
    if not citations:
        return None
    ids = list(citations)
    return {"ids": ids, "display": "Sources: " + ", ".join(ids)}


async def agui_events(
    turn: AsyncIterator[TurnEvent],
    *,
    thread_id: str,
    run_id: str,
    tools: Mapping[str, BaseTool] | None = None,
    history: Callable[[], Awaitable[Sequence[Message]]] | None = None,
) -> AsyncIterator[BaseEvent]:
    """Map one turn onto AG-UI, in the order a client can render.

    ``tools`` is the agent's bound tools by name, read only for the ``ui://``
    bundle a tool declares.

    ``history`` is awaited once the turn has finished, and what it returns goes
    out as a ``MESSAGES_SNAPSHOT`` before ``RUN_FINISHED``. History here is the
    server's — a client posts its whole array and only the trailing user
    message is read — so one that edited its own copy is otherwise wrong with
    nothing on the wire to say so.

    It is deliberately the *end* of the run and not the start. A client renders
    its question the moment it is typed, and a snapshot is applied by dropping
    every local message the snapshot does not name: sent up front, before this
    turn is checkpointed, it would take the question off the screen and leave
    the answer under nothing. At the end everything this turn produced is in
    the thread, and — because the ids match, see ``AnswerChunk.id`` — the
    client reconciles in place rather than rebuilding its list.

    Omit it and no snapshot is sent, which is what a consumer driving this
    without a checkpointer wants.

    Exceptions from the turn become ``RUN_ERROR`` and end the stream: a client
    that opened an SSE connection gets told, rather than watching it close.
    """
    yield RunStartedEvent(thread_id=thread_id, run_id=run_id)

    calls: dict[str, ToolStarted] = {}
    state: dict[str, StateEntry] = {}
    #: The open answer message, or None when none is open. Holding the id
    #: rather than a flag means every close has the id it needs.
    open_message: str | None = None
    #: Views waiting for the state their tool just wrote, as
    #: ``(message id, tool, uri)``. See :func:`view_content` for why they
    #: cannot go out where they are built.
    deferred: list[tuple[str, ToolFinished, str]] = []
    sequence = 0

    #: The metadata map this run has told the client about, or None before it
    #: has said anything. What a delta is measured against.
    announced: dict[str, Any] | None = None

    def state_event(source: Mapping[str, StateEntry] | None) -> StateDeltaEvent | None:
        """One ``STATE_DELTA``, or None when nothing about state has moved."""
        nonlocal announced
        current = state_metadata(source)
        operations = state_patch(announced, current)
        announced = current
        return StateDeltaEvent(delta=operations) if operations else None

    def next_id(prefix: str) -> str:
        nonlocal sequence
        sequence += 1
        return f"{prefix}_{run_id}_{sequence}"

    def ready() -> list[BaseEvent]:
        """Take the deferred views, filled, dropping any with nothing to draw."""
        taken, deferred[:] = list(deferred), []
        return [
            _activity(message_id, MCP_VIEW, content)
            for message_id, finished, uri in taken
            if (content := view_content(finished, uri, state)) is not None
        ]

    try:
        async for event in turn:
            match event:
                case ToolStarted():
                    # Anything still held belongs to the *previous* call, and
                    # this is the last moment it can be sent and still sit
                    # beside it rather than after everything that follows.
                    for view in ready():
                        yield view
                    # A tool call cannot open while a text message is: the
                    # verifier rejects it, and the answer so far is complete.
                    if open_message is not None:
                        yield TextMessageEndEvent(message_id=open_message)
                        open_message = None
                    calls[event.id] = event
                    yield ToolCallStartEvent(
                        tool_call_id=event.id,
                        tool_call_name=event.name,
                        parent_message_id=event.message_id,
                    )
                    yield ToolCallArgsEvent(
                        tool_call_id=event.id,
                        delta=json.dumps(event.arguments, default=str),
                    )
                    yield ToolCallEndEvent(tool_call_id=event.id)

                case ToolFinished():
                    yield ToolCallResultEvent(
                        message_id=event.message_id or next_id("tool"),
                        tool_call_id=event.id,
                        content=event.content,
                    )
                    started = calls.get(event.id)
                    if consumed := consumed_content(
                        started.arguments if started else {}, event, state
                    ):
                        yield _activity(next_id("act"), STATE_CONSUMED, consumed)
                    if published := published_content(event):
                        yield _activity(next_id("act"), STATE_PUBLISHED, published)
                    if uri := view_uri_for((tools or {}).get(event.name)):
                        deferred.append((next_id("act"), event, uri))

                case StateChanged():
                    state = dict(event.state)
                    if delta := state_event(state):
                        yield delta
                    # The write that was missing is in now, so a view held
                    # back at its tool can be filled and sent.
                    for view in ready():
                        yield view

                case AnswerChunk():
                    if open_message is None:
                        # A tool that published nothing produced no state
                        # change, so its view is still waiting. Last chance:
                        # once this message exists, an activity renders after
                        # the answer instead of beside its call.
                        for view in ready():
                            yield view
                        # The provider's own id where there is one, so the
                        # message this stream created and the message a
                        # readback returns are the same message. A snapshot
                        # then reconciles a client's copy in place instead of
                        # dropping it and re-appending the server's.
                        open_message = event.id or next_id("msg")
                        yield TextMessageStartEvent(message_id=open_message)
                    yield TextMessageContentEvent(
                        message_id=open_message, delta=event.text
                    )

                case TurnFinished():
                    # A turn that ended without an answer still owes the
                    # client any view it has been holding.
                    for view in ready():
                        yield view
                    if open_message is not None:
                        yield TextMessageEndEvent(message_id=open_message)
                        open_message = None
                    # The merged state, which the mid-turn deltas are not: they
                    # are assembled from what each node wrote, before the
                    # reducer has assigned write order. Usually this closing
                    # delta says only that each entry now has its `seq`.
                    if event.result.sidecar:
                        if delta := state_event(event.result.sidecar):
                            yield delta
                    if cited := citations_content(event.result.citations):
                        yield _activity(next_id("act"), ANSWER_CITATIONS, cited)
    except Exception as error:  # noqa: BLE001 - the client is owed a reason
        # Close an open message first: RUN_ERROR while a text message is
        # still open is rejected by the client's verifier.
        if open_message is not None:
            yield TextMessageEndEvent(message_id=open_message)
        yield RunErrorEvent(message=str(error) or type(error).__name__)
        return

    if history is not None:
        # Not inside the `try`: a failed turn leaves the thread mid-write, and
        # a snapshot of that would tell a client to adopt a transcript the
        # server may itself discard. A run that errored says so and stops.
        yield MessagesSnapshotEvent(messages=list(await history()))
    yield RunFinishedEvent(thread_id=thread_id, run_id=run_id)
