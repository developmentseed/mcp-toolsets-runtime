"""``create_router``: the agent over HTTP.

Driven through a real ASGI transport rather than by calling the handlers, so
the SSE framing, the path converters and the status codes are all exercised as
a browser would meet them. Behind it is the same real agent the streaming and
events suites use — a real graph, real tools, real session state.
"""

import asyncio
import json
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import httpx
import pytest
from fastapi import FastAPI, Request
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver

from mcp_agent.main import BuiltAgent, _credentials, with_session_state
from mcp_agent_api.events import TOOLS_WITHHELD
from mcp_agent_api.routes import (
    Built,
    ViewCache,
    create_router,
    credentials_for,
    latest_user_text,
    thread_messages,
)
from mcp_state.wiring import Unsatisfiable
from tests.mcp_agent.test_streaming import (
    AOI,
    STATE_KEY,
    StreamingScriptedModel,
    _agent,
    _consumer,
    _publisher,
)

VIEW_URI = "ui://raster-ops/map"
VIEW_HTML = "<!doctype html><p>a map</p>"


def _built(**overrides: Any) -> BuiltAgent:
    """A built agent, as the runtime's own ``build_agent`` returns one.

    A ``NamedTuple``, deliberately: its fields are read-only properties, which
    is what the ``Built`` protocol has to accept for the real thing to satisfy
    it.
    """
    fields: dict[str, Any] = {
        "agent": _agent(),
        "connections": {},
        "tools": [_publisher(), _consumer()],
        "withheld": [],
        "required": None,
    }
    return BuiltAgent(**{**fields, **overrides})


def _client(built: Built | None = None, **kwargs: Any) -> httpx.AsyncClient:
    # One agent for the life of the client, not one per request: the default
    # checkpointer is in-process, so a provider handing back a fresh agent each
    # time would give every request its own empty set of threads.
    agent = built if built is not None else _built()
    app = FastAPI()
    app.include_router(create_router(lambda: agent, **kwargs))
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://api"
    )


def _ask(text: str = "clip chirps", **body: Any) -> dict[str, Any]:
    return {"messages": [{"id": "u1", "role": "user", "content": text}], **body}


def _frames(payload: str) -> list[dict[str, Any]]:
    """The SSE body parsed back into events, the way a client's reader does."""
    return [
        json.loads(block.removeprefix("data: "))
        for block in payload.split("\n\n")
        if block.startswith("data: ")
    ]


async def _run(client: httpx.AsyncClient, **body: Any) -> list[dict[str, Any]]:
    response = await client.post("/runs", json=_ask(**body))
    assert response.status_code == 200
    return _frames(response.text)


# --- POST /runs ------------------------------------------------------------


async def test_a_run_is_a_stream_of_ag_ui_frames():
    async with _client() as client:
        events = await _run(client, threadId="t1")

    assert events[0]["type"] == "RUN_STARTED"
    assert events[-1]["type"] == "RUN_FINISHED"
    # The wire is camelCase, and it is the encoder that makes it so — a handler
    # returning model_dump() would send snake_case and no client would read it.
    assert events[0]["threadId"] == "t1"
    assert "TOOL_CALL_START" in [event["type"] for event in events]


async def test_the_response_is_an_event_stream():
    async with _client() as client:
        response = await client.post("/runs", json=_ask())

    assert response.headers["content-type"].startswith("text/event-stream")
    # A proxy buffering the turn into one delivery makes this a slow
    # non-streaming response, which is the one thing it must not be.
    assert response.headers["x-accel-buffering"] == "no"


async def test_the_frames_leave_one_at_a_time():
    """Asserted at the ASGI layer because that is the last place the boundaries
    exist: httpx's in-memory transport concatenates the body parts, so a handler
    that assembled the whole turn and returned it once would look identical
    through the client above."""
    app = FastAPI()
    app.include_router(create_router(lambda: _built()))
    body = json.dumps(_ask()).encode()
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/runs",
        "raw_path": b"/runs",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"host", b"api"), (b"content-type", b"application/json")],
        "client": ("127.0.0.1", 1234),
        "server": ("api", 80),
    }
    written: list[bytes] = []
    delivered = False
    never = asyncio.Event()

    async def receive() -> dict[str, Any]:
        # After the request body, block. Starlette watches this channel for a
        # client disconnect for as long as the response runs, and cancels the
        # watcher when it finishes; a receive that returned immediately would
        # spin that watcher forever.
        nonlocal delivered
        if delivered:
            await never.wait()
        delivered = True
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        if message["type"] == "http.response.body" and message.get("body"):
            written.append(message["body"])

    await app(scope, receive, send)

    assert len(written) > 1
    assert written[0].startswith(b'data: {"type":"RUN_STARTED"')
    assert written[-1].startswith(b'data: {"type":"RUN_FINISHED"')


async def test_a_client_without_a_thread_id_is_told_the_one_it_got():
    """There is no other channel for it: the body is consumed as a stream, so
    the ids have to arrive in the first event."""
    async with _client() as client:
        events = await _run(client)
        thread_id = events[0]["threadId"]
        assert thread_id and events[0]["runId"]
        assert (await client.get(f"/threads/{thread_id}")).status_code == 200


async def test_withheld_tools_reach_the_client():
    """The first caller to pass ``BuiltAgent.withheld`` through for real. It is
    a list of ``Unsatisfiable``, and rendering it as though it were a list of
    names raises inside the generator and becomes a RUN_ERROR."""
    withheld = [
        Unsatisfiable(
            tool="submit_request",
            parameter="aoi",
            wants="geojson.AreaOfInterest",
            required=True,
            model_generatable=False,
        )
    ]

    async with _client(_built(withheld=withheld)) as client:
        events = await _run(client)

    announced = [
        event for event in events if event.get("activityType") == TOOLS_WITHHELD
    ]
    assert [item["tool"] for item in announced[0]["content"]["tools"]] == [
        "submit_request"
    ]
    assert "RUN_ERROR" not in [event["type"] for event in events]


async def test_a_turn_that_fails_is_reported_on_the_stream():
    """Once the response has begun there is no status code left to send."""
    async with _client(_built(agent=_agent(script=[]))) as client:
        events = await _run(client)

    assert events[-1]["type"] == "RUN_ERROR"


async def test_an_agent_that_is_not_ready_is_a_status_code():
    """Resolved before the response opens, so a lifespan still connecting to
    its MCP servers answers 503 rather than an empty stream."""

    def not_yet() -> Built:
        raise RuntimeError("still connecting")

    app = FastAPI()
    app.include_router(create_router(not_yet))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://api"
    ) as client:
        response = await client.post("/runs", json=_ask())

    assert response.status_code == 503
    assert "still connecting" in response.json()["detail"]


async def test_a_request_with_no_question_is_rejected():
    """Running the model on an empty string costs a call and answers nothing."""
    async with _client() as client:
        response = await client.post("/runs", json={"messages": []})

    assert response.status_code == 422


async def test_a_credential_header_is_in_force_while_the_tool_runs():
    """The handler has long returned by the time the first tool is called, so
    the context has to be entered inside the stream rather than around it."""
    seen: list[dict[str, str] | None] = []

    async def call() -> str:
        seen.append(_credentials.get())
        return "ok"

    peek = StructuredTool(
        name="peek",
        description="peek",
        args_schema={"type": "object", "properties": {}},
        coroutine=call,
    )
    agent, _ = with_session_state(
        StreamingScriptedModel(
            script=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "peek", "args": {}, "id": "c1", "type": "tool_call"}
                    ],
                ),
                AIMessage(content="done"),
            ]
        ),
        [peek],
        InMemorySaver(),
    )
    built = _built(agent=agent, tools=[peek], required={"cds": ["x-cds-token"]})

    async with _client(built) as client:
        await client.post("/runs", json=_ask(), headers={"x-cds-token": "the user's"})

    assert seen == [{"x-cds-token": "the user's"}]


async def test_the_turn_context_wraps_the_run_and_is_told_its_ids():
    """It is handed the request and both ids, and it is left on the way out —
    a wrapper that never exits leaks whatever it opened, once per run."""
    seen: list[tuple[str, str, str]] = []
    exited: list[bool] = []

    @contextmanager
    def around(request: Request, thread_id: str, run_id: str) -> Iterator[None]:
        seen.append((request.url.path, thread_id, run_id))
        try:
            yield None
        finally:
            exited.append(True)

    async with _client(turn_context=around) as client:
        events = await _run(client, threadId="t9")

    assert seen == [("/runs", "t9", events[0]["runId"])]
    assert exited == [True]


async def test_the_turn_context_yields_the_turn_its_config():
    """What a host traces with is a callback in the runnable config, so the
    yielded value has to reach the graph run rather than be dropped."""
    started: list[str] = []

    class Recorder(BaseCallbackHandler):
        def on_chain_start(
            self, serialized: dict[str, Any], inputs: dict[str, Any], **kwargs: Any
        ) -> None:
            started.append("chain")

    @contextmanager
    def around(
        request: Request, thread_id: str, run_id: str
    ) -> Iterator[dict[str, Any]]:
        yield {"callbacks": [Recorder()]}

    async with _client(turn_context=around) as client:
        await _run(client)

    assert started


async def test_the_turn_context_is_in_force_while_the_tool_runs():
    """The reason this is a context manager and not a config factory: a
    correlation id read by an httpx hook at request time is a context
    variable, and the tool call that reads it happens long after the handler
    has returned. Entered where ``user_credentials`` is, for that reason."""
    correlation: ContextVar[str | None] = ContextVar("correlation", default=None)
    seen: list[str | None] = []

    async def call() -> str:
        seen.append(correlation.get())
        return "ok"

    peek = StructuredTool(
        name="peek",
        description="peek",
        args_schema={"type": "object", "properties": {}},
        coroutine=call,
    )
    agent, _ = with_session_state(
        StreamingScriptedModel(
            script=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "peek", "args": {}, "id": "c1", "type": "tool_call"}
                    ],
                ),
                AIMessage(content="done"),
            ]
        ),
        [peek],
        InMemorySaver(),
    )

    @contextmanager
    def around(request: Request, thread_id: str, run_id: str) -> Iterator[None]:
        token = correlation.set(f"trace-{run_id}")
        try:
            yield None
        finally:
            correlation.reset(token)

    async with _client(
        _built(agent=agent, tools=[peek]), turn_context=around
    ) as client:
        events = await _run(client)

    assert seen == [f"trace-{events[0]['runId']}"]


async def test_the_client_does_not_get_to_write_history():
    """Only the trailing user message is read. A client posting a transcript of
    its own would otherwise be able to put words in the thread."""
    body = _ask(threadId="t1")
    body["messages"] = [
        {"id": "f1", "role": "user", "content": "an earlier question"},
        {"id": "f2", "role": "assistant", "content": "an answer nobody gave"},
        *body["messages"],
    ]

    async with _client() as client:
        assert (await client.post("/runs", json=body)).status_code == 200
        thread = (await client.get("/threads/t1")).json()

    said = [str(message.get("content") or "") for message in thread["messages"]]
    assert "clip chirps" in said
    assert "an earlier question" not in said
    assert "an answer nobody gave" not in said


# --- GET /threads/{id} -----------------------------------------------------


async def test_a_thread_reads_back_as_messages():
    async with _client() as client:
        await _run(client, threadId="t1")
        thread = (await client.get("/threads/t1")).json()

    roles = [message["role"] for message in thread["messages"]]
    assert roles[0] == "user"
    assert roles.count("tool") == 2
    # Past turns' activities are not rebuilt; the state and view routes are how
    # a reloaded page recovers what they carried.
    assert "activity" not in roles
    assert thread["state"][STATE_KEY]["kind"] == "geojson.AreaOfInterest"


async def test_a_tool_call_survives_the_round_trip():
    """AG-UI carries a call's arguments as a JSON string, not as an object."""
    async with _client() as client:
        await _run(client, threadId="t1")
        thread = (await client.get("/threads/t1")).json()

    calls = [
        call
        for message in thread["messages"]
        for call in message.get("toolCalls") or []
    ]
    assert [call["function"]["name"] for call in calls] == ["search", "clip"]
    assert json.loads(calls[0]["function"]["arguments"]) == {}


async def test_an_unknown_thread_is_not_an_empty_conversation():
    async with _client() as client:
        response = await client.get("/threads/never-ran")

    assert response.status_code == 404


# --- GET /threads/{id}/state/{key} -----------------------------------------


async def test_a_state_value_is_served_in_full():
    """The wire carries ``{kind, tool, bytes}`` and no payload; this is the
    route a client follows when it decides it wants the geometry."""
    async with _client() as client:
        await _run(client, threadId="t1")
        response = await client.get(f"/threads/t1/state/{STATE_KEY}")

    body = response.json()
    assert body["value"] == AOI
    assert body["kind"] == "geojson.AreaOfInterest"
    assert body["seq"] == 1


async def test_a_qualified_key_is_one_key_and_not_two_path_segments():
    """State keys are namespaced with the publishing toolset, so every real key
    contains the separator the router would otherwise split on."""
    assert "/" in STATE_KEY

    async with _client() as client:
        await _run(client, threadId="t1")
        response = await client.get(f"/threads/t1/state/{STATE_KEY}")

    assert response.json()["key"] == STATE_KEY


async def test_an_unknown_state_key_is_a_404():
    async with _client() as client:
        await _run(client, threadId="t1")
        response = await client.get("/threads/t1/state/nothing/here")

    assert response.status_code == 404


# --- GET /views/{toolset}/{view} -------------------------------------------


async def test_a_view_bundle_is_served_as_html(monkeypatch: pytest.MonkeyPatch):
    reads = 0

    async def bundles(connections: Any, required: Any) -> dict[str, str]:
        nonlocal reads
        reads += 1
        return {VIEW_URI: VIEW_HTML}

    monkeypatch.setattr("mcp_agent_api.routes.view_bundles", bundles)

    async with _client() as client:
        first = await client.get("/views/raster-ops/map")
        second = await client.get("/views/raster-ops/map")

    assert first.text == VIEW_HTML
    assert first.headers["content-type"].startswith("text/html")
    assert second.text == VIEW_HTML
    # A bundle can be hundreds of kilobytes and does not change under a running
    # deployment, so the toolsets are read once.
    assert reads == 1


async def test_an_undeclared_view_is_a_404(monkeypatch: pytest.MonkeyPatch):
    async def bundles(connections: Any, required: Any) -> dict[str, str]:
        return {VIEW_URI: VIEW_HTML}

    monkeypatch.setattr("mcp_agent_api.routes.view_bundles", bundles)

    async with _client() as client:
        response = await client.get("/views/raster-ops/missing")

    assert response.status_code == 404


async def test_nothing_is_read_from_the_toolsets_at_mount_time(
    monkeypatch: pytest.MonkeyPatch,
):
    """The connections are not known until the lifespan has built the agent, so
    a cache that filled itself on construction would be built from nothing."""
    resolved = 0

    def provider() -> Built:
        nonlocal resolved
        resolved += 1
        return _built()

    async def bundles(connections: Any, required: Any) -> dict[str, str]:
        return {VIEW_URI: VIEW_HTML}

    monkeypatch.setattr("mcp_agent_api.routes.view_bundles", bundles)
    cache = ViewCache(provider)
    assert resolved == 0

    assert await cache.html(VIEW_URI) == VIEW_HTML
    assert resolved == 1


# --- credentials -----------------------------------------------------------


def test_only_declared_credential_headers_are_forwarded():
    """``Authorization`` on the way in belongs to this API. Putting it on an
    outbound MCP call would transit a token addressed to somebody else, which
    the MCP authorization spec forbids outright."""
    headers = {
        "authorization": "Bearer this-api",
        "x-cds-token": "the user's",
        "x-unrelated": "no",
    }

    resolved = credentials_for(headers, {"cds": ["x-cds-token"]})

    assert resolved == {"x-cds-token": "the user's"}


def test_a_deployment_with_no_declaration_forwards_nothing():
    """``required`` is ``None`` for a server pointed at directly, so there is no
    declaration saying which header is a credential — and guessing would mean
    forwarding this API's own."""
    assert credentials_for({"authorization": "Bearer this-api"}, None) == {}


def test_an_undeclared_header_falls_back_to_the_environment(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("X_CDS_TOKEN", "the deployment's")

    assert credentials_for({}, {"cds": ["x-cds-token"]}) == {
        "x-cds-token": "the deployment's"
    }


def test_a_request_header_beats_the_environment(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("X_CDS_TOKEN", "the deployment's")

    resolved = credentials_for({"x-cds-token": "the user's"}, {"cds": ["x-cds-token"]})

    assert resolved == {"x-cds-token": "the user's"}


# --- request and transcript shapes -----------------------------------------


def test_the_question_is_read_out_of_content_parts():
    """AG-UI content is a string or a list of typed parts; a client sending an
    image alongside its question still asked a question."""
    messages = [
        {"role": "user", "content": "stale"},
        {"role": "assistant", "content": "…"},
        {
            "role": "user",
            "content": [
                {"type": "image", "url": "…"},
                {"type": "text", "text": "what is this?"},
            ],
        },
    ]

    assert latest_user_text(messages) == "what is this?"


def test_a_request_with_nothing_to_say_is_empty_not_an_error():
    assert latest_user_text([{"role": "assistant", "content": "hi"}]) == ""


def test_structured_answer_content_is_flattened_not_repr_d():
    """A provider's answer arrives as a list of blocks. ``str(.content)`` on it
    renders a Python repr, brackets and quotes and all."""
    message = AIMessage(content=[{"type": "text", "text": "Three datasets."}])

    (converted,) = thread_messages([message])

    assert converted.content == "Three datasets."


# --- what the OpenAPI document says ----------------------------------------


def _openapi() -> dict[str, Any]:
    app = FastAPI()
    app.include_router(create_router(lambda: _built()))
    return app.openapi()


def test_the_read_routes_document_their_shapes():
    """They returned a bare `object, additionalProperties: true` before, which
    told a client writing against them nothing at all."""
    schemas = _openapi()["components"]["schemas"]

    assert set(schemas) >= {
        "ThreadResponse",
        "TurnsResponse",
        "StateValueResponse",
        "StateEntryInfo",
    }
    assert set(schemas["StateEntryInfo"]["properties"]) == {
        "kind",
        "tool",
        "bytes",
        "seq",
    }
    # seq is the one that may legitimately be absent; the rest always travel.
    assert set(schemas["StateEntryInfo"]["required"]) == {"kind", "tool", "bytes"}


def test_the_run_route_is_documented_as_an_event_stream():
    """The default would say application/json, which is the one thing a turn
    never is — and it is the endpoint everyone reads first."""
    content = _openapi()["paths"]["/runs"]["post"]["responses"]["200"]["content"]

    assert list(content) == ["text/event-stream"]


async def test_documenting_the_shapes_left_the_wire_alone():
    """The models are attached through `responses`, not `response_model`, so
    nothing is filtered or re-serialised on the way out.

    `state_metadata` omits `seq` until the write is merged — a client sorting
    by it would otherwise be sorting nulls — while `kind: null` is meaningful
    and must stay. A response model would have turned the first into
    `"seq": null` and `response_model_exclude_none` would have dropped the
    second, so this asserts both.
    """
    async with _client() as client:
        await _run(client, threadId="t1")
        body = (await client.get("/threads/t1")).json()

    entries = list(body["state"].values())
    assert entries, "the turn published nothing, so this proves nothing"
    for entry in entries:
        assert "kind" in entry
        if entry.get("seq") is None:
            assert "seq" not in entry


def test_a_posted_message_is_an_ag_ui_message():
    """Typed off the protocol's own union rather than described by hand, so
    every role it defines — including the `activity` messages this server
    emits, which a client echoing its history sends back — is accepted."""
    schemas = _openapi()["components"]["schemas"]
    posted = _openapi()["paths"]["/runs"]["post"]["requestBody"]["content"][
        "application/json"
    ]["schema"]

    assert posted["$ref"].endswith("RunRequest")
    assert {"UserMessage", "AssistantMessage", "ToolMessage", "ActivityMessage"} <= set(
        schemas
    )


async def test_a_message_without_an_id_is_refused():
    """The protocol requires one on every message. Nothing here reads it — the
    id a client posts is discarded — but claiming AG-UI conformance and then
    accepting a message the protocol forbids helps nobody."""
    async with _client() as client:
        response = await client.post(
            "/runs", json={"messages": [{"role": "user", "content": "hi"}]}
        )

    assert response.status_code == 422


def test_the_run_route_documents_every_event_it_can_emit():
    """A client cannot write a reader against `{}`. The union is AG-UI's own,
    so the document names all 33 event types and discriminates on `type`."""
    document = _openapi()
    schema = document["paths"]["/runs"]["post"]["responses"]["200"]["content"][
        "text/event-stream"
    ]["schema"]

    assert len(schema["oneOf"]) == 33
    assert schema["discriminator"]["propertyName"] == "type"
    # Every branch resolves: passing the model is what registers the event
    # schemas into components, and a dangling $ref renders as nothing at all.
    defined = set(document["components"]["schemas"])
    referenced = {
        ref["$ref"].rsplit("/", 1)[-1] for ref in schema["oneOf"] if "$ref" in ref
    }
    assert referenced <= defined
    assert {
        "RunStartedEvent",
        "TextMessageContentEvent",
        "ActivitySnapshotEvent",
    } <= referenced
