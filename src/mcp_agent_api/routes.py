"""The agent as HTTP routes, mountable into an application of your own.

An ``APIRouter`` and nothing else: no app, no CORS, no lifespan, no settings.
A deployment with its own FastAPI application — its own auth, its own
middleware, its own everything — mounts this and keeps all of that.
:mod:`mcp_agent_api.app` is the other end, for a deployment that wants the
whole service handed over.

Five routes, which is what the wire in :mod:`mcp_agent_api.events` implies:

``POST /runs``
    One turn, streamed as Server-Sent Events. The whole conversation is here.
``GET /threads/{thread_id}``
    The thread's messages, so a page reload restores it.
``GET /threads/{thread_id}/turns``
    The thread's turns, and what session state held at the end of each. The
    state channel is cumulative, so this is what says which keys a *particular*
    turn added.
``GET /threads/{thread_id}/state/{key}``
    One session-state value in full. The wire carries only ``{kind, tool,
    bytes}`` per key, so this is where a client that decided it wants the
    38 kB geometry comes to get it. ``?turn=N`` serves it as of that turn
    rather than as of now.
``GET /views/{toolset}/{view}``
    The HTML for a ``ui://`` bundle a tool declared. A bundle can be hundreds
    of kilobytes and does not change within a deployment, so it is fetched
    once and cached rather than repeated on every turn that renders it.

**The agent arrives through a callable, not as an argument.**
:func:`~mcp_agent.main.build_agent` is async and connects to MCP servers, so it
runs in a lifespan — after the router has been built and mounted. ``provider``
is called per request and may return anything with ``.agent``,
``.connections``, ``.tools``, ``.withheld`` and ``.required``; a caller holding
a built agent already passes ``lambda: built``. Reading it by attribute is not
fastidiousness: the runtime's :class:`~mcp_agent.main.BuiltAgent` and dss's
carry those five fields in different orders, and unpacking one as the other
yields ``required`` where ``withheld`` belongs.

**What wraps a run is the host's.** Tracing callbacks, a correlation id on the
outbound MCP calls, per-request metadata — none of that belongs here, and all
of it has to be in force *while the turn runs* rather than while the handler
is on the stack. ``turn_context`` is the one seam for it: a context manager
entered inside the stream, yielding the turn's runnable config. Without one,
nothing changes.

**History is the server's.** Only the trailing user message of a request is
read; the rest of ``messages`` is ignored, and the checkpointer's transcript is
the truth. That diverges from AG-UI's client-is-authoritative convention on
purpose — the values session state exists to keep out of the model's context
would otherwise have to live in the browser and be posted back every turn.

**A thread id is the only credential on a state read.** ``GET
/threads/{id}/state/{key}`` returns whatever that thread published, to whoever
names the thread. That is what lets an anonymous conversation draw its own map,
and it means the id must be treated as a secret: it leaks through logs,
referrers and browser history like any other URL component.
"""

import asyncio
import json
from collections.abc import AsyncIterator, Callable, Iterable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from typing import Any, Protocol, cast

from ag_ui.core import (
    AssistantMessage,
    Event,
    FunctionCall,
    Message,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from ag_ui.encoder import EventEncoder
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from mcp_agent.host import view_bundles
from mcp_agent.main import (
    new_thread_id,
    resolve_credentials,
    user_credentials,
)
from mcp_agent.streaming import stream_turn
from mcp_agent_api.events import agui_events, state_metadata
from mcp_agent_api.history import Turn, turns_of
from mcp_state.state import TOOL_STATE_KEY, StateEntry
from mcp_state.wiring import Unsatisfiable

#: URI scheme and layout of a view resource: ``ui://<toolset>/<view>``.
VIEW_URI = "ui://{toolset}/{view}"


class EventStreamResponse(StreamingResponse):
    """A ``StreamingResponse`` that knows what it is.

    Naming a ``media_type`` is what puts ``/runs``'s schema under
    ``text/event-stream`` in the generated OpenAPI. Without it FastAPI falls
    back to ``application/json``, which is the one thing a turn never is. The
    handler builds its own response, so this only affects the document.
    """

    media_type = "text/event-stream"


class Built(Protocol):
    """What a built agent has to expose for these routes to serve it.

    Declared as read-only properties so a ``NamedTuple`` (the runtime's
    :class:`~mcp_agent.main.BuiltAgent`), a frozen dataclass and a plain object
    all satisfy it.
    """

    @property
    def agent(self) -> Any: ...

    @property
    def connections(self) -> dict[str, Any]: ...

    @property
    def tools(self) -> list[BaseTool]: ...

    @property
    def withheld(self) -> Sequence[Unsatisfiable]: ...

    @property
    def required(self) -> dict[str, list[str]] | None: ...


#: Called per request for the agent to serve. Raising is how a caller says
#: "not built yet"; see :func:`create_router`.
Provider = Callable[[], Built]

#: Wraps one run, given the request and the ids it was assigned. Entered
#: *inside* the SSE generator, so context variables it sets are in force while
#: the turn runs; what it yields becomes ``stream_turn``'s runnable config.
#: See :func:`create_router`.
TurnContext = Callable[
    [Request, str, str], AbstractContextManager[dict[str, Any] | None]
]


class RunRequest(BaseModel):
    """What a client posts to ``/runs``.

    A permissive reading of AG-UI's ``RunAgentInput``: that model requires
    ``state``, ``tools``, ``context`` and ``forwardedProps`` as well, and this
    endpoint uses none of them, so demanding them would make a hand-written
    client harder to write for no gain. A full ``RunAgentInput`` from
    ``@ag-ui/client`` posts cleanly against it.

    ``messages`` is AG-UI's own discriminated union, which includes the
    ``activity`` and ``reasoning`` roles — so a client echoing back a history
    containing the activity messages *this server* generated still validates.

    The protocol requires an ``id`` on every message, and so do we by using its
    models. Nothing here reads it: history is the server's, only the trailing
    user message's text is taken, and the id a client posts is discarded rather
    than stored. It is required because the protocol says so — a client
    assembling ``TEXT_MESSAGE_*`` deltas needs ids on the messages it receives —
    and any string will do, a fresh uuid per message being the obvious choice.
    """

    model_config = ConfigDict(populate_by_name=True)

    thread_id: str | None = Field(default=None, alias="threadId")
    run_id: str | None = Field(default=None, alias="runId")
    messages: list[Message] = Field(default_factory=list)


class StateEntryInfo(BaseModel):
    """What the wire says about one stored value, without the value itself.

    The value is in session state precisely because it is too big for a
    transcript, so what travels is enough to decide whether to fetch it from
    ``GET /threads/{id}/state/{key}``.
    """

    kind: str | None = Field(
        description="The `Kind` the publishing tool tagged this value with. "
        "`null` is a fact rather than a gap: the value is untyped."
    )
    tool: str | None = Field(description="The tool that published it.")
    bytes: int = Field(
        description="Rough serialised size, for deciding whether to fetch it."
    )
    seq: int | None = Field(
        default=None,
        description="Publication order, assigned when the write is merged. "
        "**Omitted** rather than sent as `null` before then, so a client "
        "sorting by it is never sorting nulls.",
    )


class ThreadResponse(BaseModel):
    """``GET /threads/{thread_id}`` — a conversation, for restoring it."""

    model_config = ConfigDict(populate_by_name=True)

    thread_id: str = Field(alias="threadId")
    messages: list[Message] = Field(
        description="The transcript, oldest first — including the `activity` "
        "messages this server generates alongside the standard roles."
    )
    state: dict[str, StateEntryInfo] = Field(
        description="Session state as of now, keyed `<toolset>/<field>`."
    )


class TurnInfo(BaseModel):
    """One turn of a thread, and what state held when it finished."""

    model_config = ConfigDict(populate_by_name=True)

    turn: int = Field(description="1-based, counted off the user's messages.")
    question: str
    checkpoint_id: str | None = Field(alias="checkpointId")
    state: dict[str, StateEntryInfo] = Field(
        description="State at the **end of this turn** — the state channel is "
        "cumulative, so comparing consecutive turns is what says which keys a "
        "particular turn added."
    )


class TurnsResponse(BaseModel):
    """``GET /threads/{thread_id}/turns`` — the turns still retained."""

    model_config = ConfigDict(populate_by_name=True)

    thread_id: str = Field(alias="threadId")
    turns: int = Field(description="How many turns are still retained.")
    total: int = Field(
        description="How many the thread has had. Greater than `turns` means "
        "the checkpointer evicted the difference — a fact about the "
        "deployment's retention, not about the request."
    )
    history: list[TurnInfo]


class StateValueResponse(BaseModel):
    """``GET /threads/{thread_id}/state/{key}`` — one published value in full."""

    key: str
    kind: str | None
    tool: str | None
    seq: int | None
    turn: int | None = Field(
        description="Echoed back from `?turn=N`, so a client holding several "
        "panels cannot mistake one turn's value for another's. `null` when "
        "the value was read as of now."
    )
    value: Any = Field(description="The payload the event stream left out.")


def latest_user_text(messages: Iterable[Mapping[str, Any]]) -> str:
    """The text of the last user message, flattened out of its content parts.

    AG-UI content is a string or a list of typed parts; a client sending an
    image alongside its question still has a question in there, so the text
    parts are joined rather than the whole thing rejected.
    """
    for message in reversed(list(messages)):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
    return ""


def _as_message(message: BaseMessage, fallback_id: str) -> Message | None:
    """One LangChain message as its AG-UI counterpart, or ``None`` to drop it.

    ``.text`` rather than ``.content``: a provider's structured content is a
    list of blocks, which ``str()`` would render as a Python repr.
    """
    identifier = str(getattr(message, "id", None) or fallback_id)
    match getattr(message, "type", None):
        case "human":
            return UserMessage(id=identifier, content=message.text)
        case "system":
            return SystemMessage(id=identifier, content=message.text)
        case "tool":
            return ToolMessage(
                id=identifier,
                content=message.text,
                tool_call_id=str(getattr(message, "tool_call_id", "") or ""),
            )
        case "ai":
            calls = [
                ToolCall(
                    id=str(call.get("id") or ""),
                    function=FunctionCall(
                        name=str(call.get("name") or ""),
                        arguments=json.dumps(call.get("args") or {}, default=str),
                    ),
                )
                for call in getattr(message, "tool_calls", None) or []
            ]
            return AssistantMessage(
                id=identifier,
                content=message.text or None,
                tool_calls=calls or None,
            )
    return None


def thread_messages(history: Iterable[BaseMessage]) -> list[Message]:
    """A thread's transcript as AG-UI messages, in order.

    Past turns' activities are not rebuilt. Doing so would mean re-deriving
    every historical turn's receipts and re-resolving its view props at the
    moment a page is trying to load; the state and view routes are how a client
    recovers a past turn's geometry or view.
    """
    return [
        converted
        for position, message in enumerate(history)
        if (converted := _as_message(message, f"msg_{position}")) is not None
    ]


class ViewCache:
    """The deployment's ``ui://`` bundles, read once.

    Bundles are static for the life of a deployment and large enough that
    re-reading them per request would be the slowest thing these routes do.
    The lock is what makes it once rather than once per concurrent request on a
    cold start.
    """

    def __init__(self, provider: Provider) -> None:
        self._provider = provider
        self._bundles: dict[str, str] | None = None
        self._lock = asyncio.Lock()

    async def html(self, uri: str) -> str | None:
        async with self._lock:
            if self._bundles is None:
                built = self._provider()
                self._bundles = await view_bundles(built.connections, built.required)
        return self._bundles.get(uri)


def credentials_for(
    headers: Mapping[str, str], required: dict[str, list[str]] | None
) -> dict[str, str]:
    """Per-request credential headers, from the request then the environment.

    Only headers a connected toolset actually declared are taken off the
    request. Forwarding everything would put this request's ``Authorization``
    — which belongs to *this* API, not to a toolset — onto an outbound MCP
    call, and the MCP authorization spec is explicit that a server must not
    transit a token addressed to somebody else.

    A deployment that sets the fallback environment variable gives every caller
    that account. That is right for local development and for a single shared
    service credential, and wrong for per-user access — such a deployment must
    leave the variable unset and let each request carry its own header.
    """
    declared = {name.lower() for names in (required or {}).values() for name in names}
    supplied = {
        name: value for name, value in headers.items() if name.lower() in declared
    }
    return resolve_credentials(required, supplied)


def create_router(
    provider: Provider,
    *,
    prefix: str = "",
    turn_context: TurnContext | None = None,
) -> APIRouter:
    """Routes serving the agent ``provider`` returns.

    ``provider`` is called on each request rather than once here, so an agent
    rebuilt behind it — on a model change, on a reconnect — is picked up
    without remounting. It may raise to signal that the agent is not ready;
    that surfaces as ``503``, which is the honest answer while a lifespan is
    still connecting to MCP servers.

    ``turn_context`` is where a host puts what it needs around each run:
    tracing callbacks, a correlation id, per-request metadata. It is a context
    manager rather than a plain config factory because the two things a host
    wants here differ in kind — a config is a *value* passed to the turn, while
    a correlation id read by an httpx hook at request time is a **context
    variable**, and that has to be set for the duration rather than handed
    over. Entered beside :func:`~mcp_agent.main.user_credentials` and for the
    same reason: by the time the first tool is called, the request handler has
    long returned, so anything scoped to the handler's stack is already gone.
    Whatever it yields — ``None`` is fine — becomes the turn's runnable config.
    """
    router = APIRouter(prefix=prefix)
    views = ViewCache(provider)

    def built() -> Built:
        try:
            return provider()
        except Exception as error:  # noqa: BLE001 - a caller's "not ready yet"
            raise HTTPException(503, f"agent unavailable: {error}") from error

    async def thread_values(thread_id: str) -> dict[str, Any]:
        """A thread's checkpointed values, or ``404`` if it holds nothing.

        An unknown thread and an empty one are indistinguishable in the
        checkpointer, and a thread that ran a turn always has messages — so
        empty means the id is wrong, which is worth saying rather than
        answering a mistyped id with an empty conversation.
        """
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = await built().agent.aget_state(cast(Any, config))
        values = dict(getattr(snapshot, "values", None) or {})
        if not values.get("messages"):
            raise HTTPException(404, f"no thread {thread_id!r}")
        return values

    @router.post(
        "/runs",
        response_class=EventStreamResponse,
        responses={
            200: {
                # AG-UI's own Event union: 33 members, discriminated on `type`.
                # Passing the model is also what registers every event schema
                # into components, so the oneOf's refs resolve.
                "model": Event,
                "description": (
                    "One turn, as Server-Sent Events. Each frame is a `data:` "
                    "line carrying an AG-UI event: `RUN_STARTED` first (with "
                    "both ids), then `TEXT_MESSAGE_*` for the answer, "
                    "`TOOL_CALL_*` per tool, `STATE_SNAPSHOT`, and "
                    "`ACTIVITY_SNAPSHOT` for what AG-UI has no vocabulary for "
                    "— where a tool's arguments came from and which `ui://` "
                    "view renders its result. Ends with `RUN_FINISHED`, or "
                    "`RUN_ERROR` if the turn failed after the stream opened. "
                    "The schema below is every event the protocol defines; this "
                    "server emits the subset named above."
                ),
            }
        },
    )
    async def create_run(body: RunRequest, request: Request) -> StreamingResponse:
        """Run one turn, streamed as AG-UI Server-Sent Events.

        A client that omitted ``threadId`` learns the one it was given from
        ``RUN_STARTED``, which carries both ids and is always the first event.

        The agent is resolved before the response begins so that "not ready"
        is a status code. Once the stream is open the only way to report a
        failure is ``RUN_ERROR``, which :func:`~mcp_agent_api.events.agui_events`
        already does for anything the turn raises.
        """
        agent = built()
        # Dumped back to mappings for the helper, which reads AG-UI content
        # parts and is shared with callers that never had models.
        question = latest_user_text(
            message.model_dump(by_alias=True) for message in body.messages
        )
        if not question:
            # A turn on an empty string would run the model, cost a call and
            # answer nothing. The client dropped its own question.
            raise HTTPException(422, "no user message to run")
        thread_id = body.thread_id or new_thread_id()
        run_id = body.run_id or new_thread_id()
        credentials = credentials_for(request.headers, agent.required)
        encoder = EventEncoder(accept=request.headers.get("accept", ""))

        async def frames() -> AsyncIterator[str]:
            # Both are entered inside the generator, so they are in force while
            # the turn runs rather than only while the handler is on the stack
            # — by the time the first tool is called, this handler has long
            # returned.
            around: AbstractContextManager[dict[str, Any] | None] = (
                turn_context(request, thread_id, run_id)
                if turn_context
                else nullcontext(None)
            )
            with user_credentials(credentials or None), around as config:
                turn = stream_turn(agent.agent, question, thread_id, config)
                async for event in agui_events(
                    turn,
                    thread_id=thread_id,
                    run_id=run_id,
                    tools={tool.name: tool for tool in agent.tools},
                    withheld=agent.withheld,
                ):
                    yield encoder.encode(event)

        return StreamingResponse(
            frames(),
            media_type=encoder.get_content_type(),
            # Nothing between here and the browser may buffer a turn into one
            # delivery: a stream that arrives whole at the end is a slow
            # non-streaming response.
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    @router.get("/threads/{thread_id}", responses={200: {"model": ThreadResponse}})
    async def read_thread(thread_id: str) -> dict[str, Any]:
        """The thread's messages, for a client restoring a conversation."""
        values = await thread_values(thread_id)
        return {
            "threadId": thread_id,
            "messages": [
                message.model_dump(by_alias=True, exclude_none=True)
                for message in thread_messages(values["messages"])
            ],
            # The same shape the turn's STATE_SNAPSHOT carries, so a restored
            # thread knows what is in state without a second convention.
            "state": state_metadata(values.get(TOOL_STATE_KEY)),
        }

    async def retained(thread_id: str, n: int) -> Turn:
        """Turn ``n`` of a thread, or the right kind of failure.

        Three outcomes, and they are genuinely different things: a turn the
        thread never had is a client mistake, a turn the checkpointer no longer
        keeps is a fact about the deployment, and either beats implying the
        value never existed.
        """
        await thread_values(thread_id)  # 404s an unknown thread, as ever.
        history = await turns_of(built().agent, thread_id)
        if found := history.find(n):
            return found
        if n > history.total or n < 1:
            raise HTTPException(
                404,
                f"no turn {n} on thread {thread_id!r}: it has had {history.total}",
            )
        raise HTTPException(
            410,
            f"turn {n} of thread {thread_id!r} is no longer retained — the "
            f"checkpointer keeps {len(history.turns)} of its {history.total} "
            "turns. The value existed; it has been pruned.",
        )

    @router.get("/threads/{thread_id}/turns", responses={200: {"model": TurnsResponse}})
    async def read_turns(thread_id: str) -> dict[str, Any]:
        """The thread's turns, and what session state held at the end of each.

        The state channel is cumulative — a snapshot names every key the thread
        holds — so this is what a client needs to say "these two keys are what
        *this* turn added", and to offer a value as of a turn rather than as of
        now.
        """
        await thread_values(thread_id)
        history = await turns_of(built().agent, thread_id)
        return {
            "threadId": thread_id,
            # Both counts, always: a client cannot tell eviction from a short
            # conversation without them, and the difference is the deployment's
            # retention rather than anything it did wrong.
            "turns": len(history.turns),
            "total": history.total,
            "history": [
                {
                    "turn": turn.n,
                    "question": turn.question,
                    "checkpointId": turn.checkpoint_id,
                    # The same shape every STATE_SNAPSHOT carries, so a turn's
                    # state needs no second convention to read.
                    "state": state_metadata(turn.state),
                }
                for turn in history.turns
            ],
        }

    @router.get(
        "/threads/{thread_id}/state/{key:path}",
        responses={200: {"model": StateValueResponse}},
    )
    async def read_state(
        thread_id: str, key: str, turn: int | None = None
    ) -> dict[str, Any]:
        """One published value in full — the payload the wire left out.

        ``{key:path}`` because state keys are qualified with the publishing
        toolset (``dataset-search/geometry``) and that slash is part of the
        key, not a path separator.

        ``?turn=N`` serves the value **as it stood at the end of turn N**
        rather than now. Without it a key a later turn overwrote reads back as
        the later value, which is right for "what is in state" and wrong for
        "what did this turn run on".
        """
        if turn is None:
            values = await thread_values(thread_id)
            state = values.get(TOOL_STATE_KEY) or {}
        else:
            state = (await retained(thread_id, turn)).state

        entry: StateEntry | None = state.get(key)
        if entry is None:
            raise HTTPException(
                404,
                f"no state {key!r} on thread {thread_id!r}"
                + ("" if turn is None else f" at turn {turn}"),
            )
        return {
            "key": key,
            "kind": entry.get("kind"),
            "tool": entry.get("tool"),
            "seq": entry.get("seq"),
            # Echoed so a client holding several panels cannot mistake one
            # turn's value for another's.
            "turn": turn,
            "value": entry.get("value"),
        }

    @router.get(
        "/views/{toolset}/{view}",
        response_class=HTMLResponse,
        responses={
            200: {
                "description": (
                    "The view's self-contained HTML bundle. Hundreds of "
                    "kilobytes and fixed for the life of a deployment, so "
                    "fetch it once and cache it."
                ),
                "content": {"text/html": {"schema": {"type": "string"}}},
            }
        },
    )
    async def read_view(toolset: str, view: str) -> HTMLResponse:
        """The HTML bundle behind a ``mcp.view`` activity's URI.

        The URI's two components are path segments rather than the whole
        ``ui://toolset/view`` string, which would have to be escaped into one.
        """
        html = await views.html(VIEW_URI.format(toolset=toolset, view=view))
        if html is None:
            raise HTTPException(404, f"no view {toolset}/{view}")
        return HTMLResponse(html)

    return router
