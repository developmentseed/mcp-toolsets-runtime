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
   is dropped without error, and a patch that fails to apply is a console
   warning. Only snapshots are emitted here, each complete in itself.
4. **Activity content must be a JSON object.** ``ActivityMessage.content`` is a
   mapping, so a bare list would not survive being read back — citations go out
   as ``{"ids": [...]}``, never as an array.
"""

import json
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import asdict
from typing import Any

from ag_ui.core import (
    ActivitySnapshotEvent,
    BaseEvent,
    RunErrorEvent,
    RunFinishedEvent,
    RunStartedEvent,
    StateSnapshotEvent,
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
from mcp_state.wiring import Unsatisfiable

#: ``activityType`` values this module emits. A client switches on these; they
#: are part of the wire contract, so they are named rather than inlined.
TOOLS_WITHHELD = "tools.withheld"
STATE_CONSUMED = "state.consumed"
STATE_PUBLISHED = "state.published"
MCP_VIEW = "mcp.view"
ANSWER_CITATIONS = "answer.citations"


def state_metadata(state: Mapping[str, StateEntry] | None) -> dict[str, Any]:
    """``tool_state`` with the values replaced by what describes them.

    A stored value is in state precisely because it is too big for a transcript,
    and a state snapshot on every tool call is no better a place for it. A
    frontend that wants the geometry fetches it from the state route; what it
    needs here is enough to decide whether to.

    ``seq`` is omitted rather than sent as null when it is not yet known. It is
    assigned by the state reducer when the write is merged, so an entry taken
    from a mid-turn update does not carry one — and a client ordering by it
    would be sorting nulls. The snapshot at the end of the turn is built from
    the merged state and does carry it. ``kind`` is always present, because
    there ``None`` is a fact: the value is untyped.
    """
    return {
        key: {
            "kind": entry.get("kind"),
            "tool": entry.get("tool"),
            "bytes": _rough_size(entry.get("value")),
            **({} if entry.get("seq") is None else {"seq": entry["seq"]}),
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


def _received(
    call: ToolStarted | None,
    finished: ToolFinished,
    state: Mapping[str, StateEntry] | None,
) -> dict[str, Any]:
    """Receipts for one call, each with the line a simple client can print.

    ``step_input`` renders the arguments as the Chainlit host shows them, so the
    rendered form on the wire and the rendered form in the bundled UI cannot
    drift apart. The receipt's own fields go out beside it: a client branches on
    ``via``, never on the string.
    """
    arguments = call.arguments if call else {}
    rendered = step_input(dict(arguments), finished, dict(state or {}))
    return {
        parameter: {**receipt, "display": str(rendered.get(parameter, ""))}
        for parameter, receipt in finished.received.items()
    }


def _activity(
    message_id: str, activity_type: str, content: dict[str, Any]
) -> BaseEvent:
    return ActivitySnapshotEvent(
        message_id=message_id, activity_type=activity_type, content=content
    )


def _view(message_id: str, finished: ToolFinished, uri: str) -> BaseEvent:
    """A view activity, minus the data — see :func:`_filled`.

    Built where the tool finishes, so its ``messageId`` keeps the run's
    ordering, but not sent from there. A view is written against its tool's own
    return, and :func:`~mcp_state.restore_structured` rebuilds that from the
    artifact plus whatever capture moved into state — and this tool's own
    writes reach state on the *next* event. Filling it here would hand a view
    the fields small enough to have stayed behind and silently drop the ones
    it exists to render.
    """
    return _activity(
        message_id,
        MCP_VIEW,
        {
            "toolCallId": finished.id,
            "tool": finished.name,
            "uri": uri,
            # Carried so the fill below can reach it; stripped before sending.
            "_artifact": finished.artifact,
            "display": f"view: {uri}",
        },
    )


def _filled(view: BaseEvent, state: Mapping[str, StateEntry] | None) -> BaseEvent:
    """The same view activity with the data its bundle renders.

    A view cannot draw a geometry it was not given, so this is the one place
    the wire carries a payload rather than a description of one — the trade
    the ``ui://`` contract makes, and why a view's own tool decides how much
    it returns.
    """
    content = dict(getattr(view, "content", {}) or {})
    artifact = content.pop("_artifact", None)
    content["data"] = restore_structured(artifact, dict(state or {}))
    return _activity(str(getattr(view, "message_id", "")), MCP_VIEW, content)


async def agui_events(
    turn: AsyncIterator[TurnEvent],
    *,
    thread_id: str,
    run_id: str,
    tools: Mapping[str, BaseTool] | None = None,
    withheld: Sequence[Unsatisfiable] = (),
) -> AsyncIterator[BaseEvent]:
    """Map one turn onto AG-UI, in the order a client can render.

    ``tools`` is the agent's bound tools by name, read only for the ``ui://``
    bundle a tool declares; ``withheld`` is ``BuiltAgent.withheld``, announced
    once at the top of the run so a client can explain a capability it does not
    have rather than appearing to ignore the request.

    Exceptions from the turn become ``RUN_ERROR`` and end the stream: a client
    that opened an SSE connection gets told, rather than watching it close.
    """
    yield RunStartedEvent(thread_id=thread_id, run_id=run_id)

    calls: dict[str, ToolStarted] = {}
    state: dict[str, StateEntry] = {}
    #: The open answer message, or None when none is open. Holding the id
    #: rather than a flag means every close has the id it needs.
    open_message: str | None = None
    #: View activities waiting for the state their tool just wrote. See
    #: :func:`_view` for why they cannot go out where they are built.
    deferred: list[BaseEvent] = []
    sequence = 0

    def next_id(prefix: str) -> str:
        nonlocal sequence
        sequence += 1
        return f"{prefix}_{run_id}_{sequence}"

    def held() -> list[BaseEvent]:
        """Take the deferred activities, leaving none behind."""
        taken, deferred[:] = list(deferred), []
        return taken

    try:
        if withheld:
            yield _activity(
                next_id("act"),
                TOOLS_WITHHELD,
                {
                    # Each declaration in full, so a client can say which
                    # parameter went unsatisfied and what kind it wanted;
                    # `asdict` because the wire carries JSON, not dataclasses.
                    "tools": [asdict(item) for item in withheld],
                    "display": f"{len(withheld)} tool(s) unavailable: "
                    + ", ".join(str(item) for item in withheld),
                },
            )

        async for event in turn:
            match event:
                case ToolStarted():
                    # A tool call cannot open while a text message is: the
                    # verifier rejects it, and the answer so far is complete.
                    if open_message is not None:
                        yield TextMessageEndEvent(message_id=open_message)
                        open_message = None
                    calls[event.id] = event
                    yield ToolCallStartEvent(
                        tool_call_id=event.id, tool_call_name=event.name
                    )
                    yield ToolCallArgsEvent(
                        tool_call_id=event.id,
                        delta=json.dumps(event.arguments, default=str),
                    )
                    yield ToolCallEndEvent(tool_call_id=event.id)

                case ToolFinished():
                    yield ToolCallResultEvent(
                        message_id=next_id("tool"),
                        tool_call_id=event.id,
                        content=event.content,
                    )
                    if event.received:
                        yield _activity(
                            next_id("act"),
                            STATE_CONSUMED,
                            {
                                "toolCallId": event.id,
                                "tool": event.name,
                                "received": _received(
                                    calls.get(event.id), event, state
                                ),
                            },
                        )
                    if event.published:
                        yield _activity(
                            next_id("act"),
                            STATE_PUBLISHED,
                            {
                                "toolCallId": event.id,
                                "tool": event.name,
                                "published": dict(event.published),
                                "display": "→ "
                                + ", ".join(sorted(event.published.values())),
                            },
                        )
                    if uri := view_uri_for((tools or {}).get(event.name)):
                        deferred.append(_view(next_id("act"), event, uri))

                case StateChanged():
                    state = dict(event.state)
                    yield StateSnapshotEvent(snapshot=state_metadata(state))
                    # The write that was missing is in now, so a view held
                    # back at its tool can be filled and sent.
                    for view in held():
                        yield _filled(view, state)

                case AnswerChunk():
                    if open_message is None:
                        # A tool that published nothing produced no state
                        # change, so its view is still waiting. Last chance:
                        # once this message exists, an activity renders after
                        # the answer instead of beside its call.
                        for view in held():
                            yield _filled(view, state)
                        open_message = next_id("msg")
                        yield TextMessageStartEvent(message_id=open_message)
                    yield TextMessageContentEvent(
                        message_id=open_message, delta=event.text
                    )

                case TurnFinished():
                    # A turn that ended without an answer still owes the
                    # client any view it has been holding.
                    for view in held():
                        yield _filled(view, state)
                    if open_message is not None:
                        yield TextMessageEndEvent(message_id=open_message)
                        open_message = None
                    # The merged state, which the mid-turn snapshots are not:
                    # they are assembled from what each node wrote, before the
                    # reducer has assigned write order.
                    if event.result.sidecar:
                        yield StateSnapshotEvent(
                            snapshot=state_metadata(event.result.sidecar)
                        )
                    if event.result.citations:
                        yield _activity(
                            next_id("act"),
                            ANSWER_CITATIONS,
                            {
                                "ids": list(event.result.citations),
                                "display": "Sources: "
                                + ", ".join(event.result.citations),
                            },
                        )
    except Exception as error:  # noqa: BLE001 - the client is owed a reason
        # Close an open message first: RUN_ERROR while a text message is
        # still open is rejected by the client's verifier.
        if open_message is not None:
            yield TextMessageEndEvent(message_id=open_message)
        yield RunErrorEvent(message=str(error) or type(error).__name__)
        return

    yield RunFinishedEvent(thread_id=thread_id, run_id=run_id)
