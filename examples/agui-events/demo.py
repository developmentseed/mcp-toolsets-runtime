"""One turn on the wire, as AG-UI — `mcp_agent_api` driven live.

Four real MCP servers on ephemeral ports, a real agent with session state, and
`stream_turn` -> `agui_events` -> `EventEncoder`.

**`--api` is the one to run.** It serves `mcp_agent_api.app` and stays up, for
the React chat client in `web/`:

    PROVIDER_MODEL=mistral-small-latest PROVIDER_API_KEY=… \
        uv run python examples/agui-events/demo.py --api

With those two set — from the environment or a `.env` — a real model drives
the toolsets and decides for itself which tool to call. Without them a scripted
stub answers by keyword instead, so the example still runs with no key and no
network; `--delay` paces its tokens, since a stub emits a sentence faster than
a screen can draw it.

The rest is the same machinery in a terminal, printed frame by frame:

    uv run python examples/agui-events/demo.py            # step, one keypress per event
    uv run python examples/agui-events/demo.py --run      # no stepping
    uv run python examples/agui-events/demo.py --raw      # the encoded SSE frames too
    uv run python examples/agui-events/demo.py --serve    # over HTTP, through the routes
    uv run python examples/agui-events/demo.py --scenario 2

Pick a scenario from the menu; each one exercises a different branch of
`events.py`. The turn's events are shown as they go out, then the transcript a
client assembles from them, in the order it would render.

`--serve` puts the HTTP layer in the middle: the same turn goes through
`create_app`, is POSTed to over a socket, and is read back off the SSE body —
so what is printed is what a browser would parse, not what the generator
yielded.
The turn is followed by the three routes a client needs afterwards: the thread,
the geometry that never entered the transcript, and the view bundle.
"""

import argparse
import asyncio
import importlib.util
import itertools
import json
import logging
import socket
import sys
import threading
from pathlib import Path
from collections.abc import AsyncIterator
from typing import Any

import httpx
import uvicorn
from ag_ui.core import (
    ActivitySnapshotEvent,
    BaseEvent,
    Event,
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
from ag_ui.encoder import EventEncoder
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import TypeAdapter, ValidationError

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SESSION_STATE = REPO / "examples" / "session-state"

# Two of the four servers are examples/session-state's, which is a directory
# rather than a package — neither it nor the toolsets under it are importable
# until they are on the path. The two servers this example adds sit beside it.
sys.path[:0] = [
    str(SESSION_STATE),
    str(SESSION_STATE / "toolsets"),
    str(HERE / "toolsets"),
]


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


from foreign_server import mcp as foreign_mcp  # noqa: E402
from mcp_agent.main import (  # noqa: E402
    AgentSettings,
    BuiltAgent,
    with_session_state,
)
from mcp_agent.streaming import stream_turn  # noqa: E402
from mcp_agent_api.events import agui_events  # noqa: E402
from mcp_agent_api.app import create_app  # noqa: E402
from mcp_runtime.server import build_server  # noqa: E402
from mcp_state import handle_for  # noqa: E402

# The suite's streaming stub, so this demo and the tests drive the agent with
# one model rather than two that can drift. GenericFakeChatModel drops
# tool_calls on its streaming path, so a demo built on that would show an answer
# and nothing else. Loaded by path: the suite has no __init__.py (pytest imports
# it in importlib mode) and an installed distribution shadows a bare `tests`.
StreamingScriptedModel = _load(
    "streaming_stub", REPO / "tests" / "mcp_agent" / "test_streaming.py"
).StreamingScriptedModel

for noisy in ("mcp", "httpx", "uvicorn", "sse_starlette", "langchain"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logging.basicConfig(level=logging.WARNING)

AOI_KEY = "dataset-search/geometry"
VIEW = ("raster-ops", "clip")
THREAD, RUN = "t1", "run-1"
QUESTION = "find rainfall data and clip chirps to my area"

#: Decodes an SSE frame back into the event class it was encoded from, so the
#: served path and the in-process one are printed by the same code.
EVENT = TypeAdapter(Event)

# toolset name -> module, for the servers built from this runtime.
SERVERS = {
    "dataset-search": "dataset_search.tools",
    "raster-ops": "clip_view.tools",  # the example's clip_raster, plus a view
    "contour-ops": "contour_ops.tools",  # withheld: nothing publishes its kind
}
FOREIGN = "terrain"

BOLD, DIM, OFF = "\033[1m", "\033[2m", "\033[0m"
CYAN, GREEN, YELLOW, RED = "\033[36m", "\033[32m", "\033[33m", "\033[31m"


def call(call_id: str, name: str, args: dict[str, Any]) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def cited(text: str, *ids: str) -> AIMessage:
    """A final answer carrying citations, the way a provider standardises them."""
    return AIMessage(
        content=[
            {
                "type": "text",
                "text": text,
                "annotations": [{"type": "citation", "id": ref} for ref in ids],
            }
        ]
    )


class Scenario:
    def __init__(
        self, title: str, blurb: str, script: list[AIMessage], toolsets: list[str]
    ) -> None:
        self.title, self.blurb, self.script, self.toolsets = (
            title,
            blurb,
            script,
            toolsets,
        )


SCENARIOS = [
    Scenario(
        "The full turn",
        "Both receipt paths, a publish, a ui:// view, then the answer — note "
        "that every activity is emitted before the answer's text message opens.",
        [
            call("1", "search_datasets", {"query": "rainfall"}),
            # No `aoi`: the declaration removed it from this tool's schema.
            call("2", "clip_raster", {"dataset_id": "chirps-daily"}),
            # The foreign tool's parameter is in the schema, so the model fills
            # it with a handle read off the [state updated: …] breadcrumb.
            call("3", "describe_geometry", {"geometry": handle_for(AOI_KEY)}),
            AIMessage(content="Done — clipped chirps-daily to your catchment."),
        ],
        ["dataset-search", "raster-ops", FOREIGN],
    ),
    Scenario(
        "A tool the deployment cannot offer",
        "contour-ops declares a kind nothing publishes and forbids the model "
        "inventing one, so it is withheld — announced once, before anything else.",
        [
            call("1", "search_datasets", {"query": "rainfall"}),
            AIMessage(content="I can find datasets, but I cannot smooth contours."),
        ],
        ["dataset-search", "raster-ops", "contour-ops", FOREIGN],
    ),
    Scenario(
        "An answer with citations",
        "Citations are the deliberate exception to the ordering rule: they "
        "belong after the answer, and are only known then.",
        [
            call("1", "search_datasets", {"query": "rainfall"}),
            cited("Three datasets cover it.", "era5-land", "chirps-daily"),
        ],
        ["dataset-search", "raster-ops", FOREIGN],
    ),
    Scenario(
        "The turn fails mid-answer",
        "The script runs out while the answer is open. The open text message "
        "is closed first, then RUN_ERROR — a client is told, not just dropped.",
        [
            call("1", "search_datasets", {"query": "rainfall"}),
        ],
        ["dataset-search", "raster-ops", FOREIGN],
    ),
]


# --- the chat model the web client talks to --------------------------------


def chat_script(turn: int, question: str) -> list[AIMessage]:
    """The scripted reply to one question, keyed off a word in it.

    The model is a stub, so it answers by keyword rather than by understanding.
    Call ids carry the turn number because a thread's ids have to stay unique:
    a client keys its rendered tool calls on them, and a second turn reusing
    ``c1`` would land in the first turn's message.
    """
    asked = question.lower()
    if any(word in asked for word in ("clip", "rainfall", "chirps", "area")):
        return [
            call(f"t{turn}-1", "search_datasets", {"query": "rainfall"}),
            # No `aoi`: the declaration removed it from this tool's schema.
            call(f"t{turn}-2", "clip_raster", {"dataset_id": "chirps-daily"}),
            # The foreign tool's parameter is in the schema, so the model fills
            # it with a handle read off the [state updated: …] breadcrumb.
            call(f"t{turn}-3", "describe_geometry", {"geometry": handle_for(AOI_KEY)}),
            AIMessage(content="Done — clipped chirps-daily to your catchment."),
        ]
    if any(word in asked for word in ("contour", "smooth")):
        return [
            AIMessage(
                content="I can't smooth contours here. smooth_contours wants a "
                "geojson.ContourSet and nothing connected publishes one, so it "
                "was withheld before the conversation started."
            )
        ]
    if any(word in asked for word in ("source", "cite", "which")):
        return [
            call(f"t{turn}-1", "search_datasets", {"query": "rainfall"}),
            cited("Three datasets cover the catchment.", "era5-land", "chirps-daily"),
        ]
    if any(word in asked for word in ("explain", "why", "tell me", "long")):
        return [AIMessage(content=ESSAY)]
    return [
        AIMessage(
            content="Try 'clip chirps to my area' for the full turn, 'what "
            "about contours?' for a withheld tool, 'which datasets, with "
            "sources?' for citations, or 'explain session state' for enough "
            "text to watch it stream."
        )
    ]


#: A long answer, because the other four are a sentence each and a sentence
#: streams past faster than a screen can show. Text only: no tool call, so
#: what arrives is tokens and nothing else.
ESSAY = """Session state exists because the useful values in a geospatial \
conversation are too big to say out loud.

An area of interest is a few thousand vertices. Put it in the transcript and \
you pay for it on every subsequent turn, the model can transcribe it wrongly, \
and a long conversation eventually stops fitting. So a tool's large return is \
captured into a namespace beside the messages, and what the model is told is \
that a value now exists under a key.

Getting it back out again happens two ways. When a server tags a parameter \
with a kind, the client matches the kind, fills the value, and removes the \
parameter from the schema the model is offered — the model cannot get the \
geometry wrong because it never learns there was one to get. When nothing is \
tagged, which is every third-party server, the parameter stays in the schema \
with a second accepted form, and the model passes an @state: handle of about \
ten tokens. The client swaps in the payload before the call, and the server \
receives ordinary GeoJSON with no idea any of this happened.

Both paths leave a receipt, which is what you can see in this transcript: the \
parameter, the key it came from, the kind, and which tool published it. That \
matters because a value the model never saw is otherwise unaccountable — \
something decided what this tool ran on, and a user is entitled to know what \
and why.

None of that is visible on the wire as bytes. The stream carries the kind, the \
publishing tool and a size for each key, and the payload stays on the server \
until a client asks for it by name. Which is the whole trick: the geometry is \
in the answer without ever being in the conversation."""


class ScriptedChat(StreamingScriptedModel):
    """The same stub, driven by the conversation instead of a fixed list.

    The agent calls the model once per tool round, so how far into a reply we
    are is how many AI messages have arrived since the last human one. Deriving
    that from the messages rather than holding a counter is what lets one
    instance serve a whole thread, and every thread the process is asked for.

    ``delay`` is the pause between tokens. A stub has none, and a real provider
    has tens of milliseconds — so without it an answer arrives faster than a
    screen can draw it and streaming that works looks exactly like streaming
    that does not. It slows the demo down and nothing else.
    """

    delay: float = 0.0

    async def _astream(self, messages: list[Any], *args: Any, **kwargs: Any) -> Any:
        turns = [message for message in messages if message.type == "human"]
        since = list(
            itertools.takewhile(
                lambda message: message.type != "human", reversed(messages)
            )
        )
        self.script = chat_script(len(turns), turns[-1].text if turns else "")
        self.index = sum(1 for message in since if message.type == "ai")
        async for chunk in super()._astream(messages, *args, **kwargs):
            if self.delay:
                await asyncio.sleep(self.delay)
            yield chunk


def chat_model(delay: float) -> tuple[Any, str]:
    """A real provider when one is configured, the stub when not.

    ``PROVIDER_MODEL`` and ``PROVIDER_API_KEY`` — from the environment or a
    ``.env``, the same two settings the CLI and the Chainlit host read — put a
    real model behind the routes, which is the only way to see a tool called
    because a model decided to call it. Without them the scripted stub answers
    by keyword, so the example still runs with no key and no network.
    """
    try:
        settings = AgentSettings()
    except ValidationError:
        return ScriptedChat(script=[], delay=delay), "scripted stub, no key set"
    return (
        init_chat_model(
            settings.provider_model,
            api_key=settings.provider_api_key.get_secret_value(),
        ),
        settings.provider_model,
    )


async def api(port: int, delay: float) -> None:
    """Serve the agent on ``port`` and stay up, for the web client to talk to.

    ``create_app`` rather than a `FastAPI()` of our own, so this is the whole
    stack a deployment gets: the lifespan below connects to the four MCP
    servers, and until it returns the routes answer 503 — which is what
    :func:`wait_for` is actually waiting for.

    Every toolset is connected, contour-ops included, so `tools.withheld` is
    announced on each run — a client can say what the deployment cannot do
    rather than appear to ignore the request.
    """
    model, named = chat_model(delay)
    connected: dict[str, Any] = {}

    async def build() -> BuiltAgent:
        """What a deployment's lifespan does, with this demo's own model."""
        tools = await MultiServerMCPClient(CONNECTIONS).get_tools()
        agent, withheld = with_session_state(model, tools, InMemorySaver())
        connected.update(tools=tools, withheld=withheld)
        return BuiltAgent(agent, CONNECTIONS, tools, withheld, None)

    # The browser talks to Vite on 5173, which proxies /api here, so this is
    # one origin as far as it is concerned and no CORS is needed. A client
    # served from anywhere else would pass `origins=[...]`.
    serve(create_app(build), port)
    await wait_for(port)

    print(f"\n{BOLD}mcp_agent_api.app on http://127.0.0.1:{port}{OFF}")
    print(f"{DIM}model: {named}{OFF}")
    print(
        f"{DIM}{len(connected['tools'])} tool(s) connected, "
        f"{len(connected['withheld'])} withheld{OFF}"
    )
    print(f"\n{DIM}now, in another terminal:{OFF}")
    print(f"  cd {HERE / 'web'} && npm install && npm run dev\n")
    await asyncio.Event().wait()


# --- the terminal demo -----------------------------------------------------


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def serve(app: Any, port: int) -> None:
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    threading.Thread(target=uvicorn.Server(config).run, daemon=True).start()


async def wait_for(port: int, attempts: int = 80) -> None:
    """Block until the app is *serving the agent*, not merely listening.

    A bound socket is the wrong signal here: `create_app` connects to the MCP
    servers in its lifespan, and a request arriving before that finishes gets
    a 503. So this asks for a thread that cannot exist and waits for the 404
    that means the agent is up — the same distinction any client of this API
    has to make on a cold start.
    """
    async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}") as client:
        for _ in range(attempts):
            try:
                if (await client.get("/threads/_probe")).status_code != 503:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.1)
    raise RuntimeError(f"127.0.0.1:{port} never started serving")


def summarise(event: Any) -> str:
    """One line per event: the type, then what a client reads off it."""
    match event:
        case RunStartedEvent() | RunFinishedEvent():
            return f"run={event.run_id}"
        case RunErrorEvent():
            return f"{RED}{event.message}{OFF}"
        case ToolCallStartEvent():
            return f"{event.tool_call_name}  (id={event.tool_call_id})"
        case ToolCallArgsEvent():
            return event.delta
        case ToolCallEndEvent():
            return f"id={event.tool_call_id}"
        case ToolCallResultEvent():
            return str(event.content).splitlines()[0][:88]
        case ActivitySnapshotEvent():
            content = dict(event.content)
            display = content.get("display")
            if display is None and "received" in content:
                display = " · ".join(
                    receipt["display"] for receipt in content["received"].values()
                )
            return f"{YELLOW}{event.activity_type}{OFF}  {display}"
        case StateSnapshotEvent():
            return json.dumps(event.snapshot, separators=(",", ":"))
        case TextMessageStartEvent() | TextMessageEndEvent():
            return f"id={event.message_id}"
        case TextMessageContentEvent():
            return repr(event.delta)
    return ""


def transcript(events: list[Any]) -> list[str]:
    """The messages a client holds, in creation order — what it would render."""
    lines: list[str] = []
    index: dict[str, int] = {}
    for event in events:
        match event:
            case ToolCallStartEvent():
                index[event.tool_call_id] = len(lines)
                lines.append(f"  {CYAN}tool{OFF}      {event.tool_call_name}(…)")
            case ToolCallResultEvent():
                lines.append(
                    f"  {DIM}result{OFF}    {str(event.content).splitlines()[0][:76]}"
                )
            case ActivitySnapshotEvent():
                content = dict(event.content)
                if "received" in content:
                    for parameter, receipt in content["received"].items():
                        lines.append(
                            f"  {YELLOW}receipt{OFF}   {parameter}: {receipt['display']}"
                            f"   {DIM}via={receipt['via']}{OFF}"
                        )
                else:
                    lines.append(f"  {YELLOW}activity{OFF}  {content.get('display')}")
            case TextMessageStartEvent():
                index[event.message_id] = len(lines)
                lines.append(f"  {GREEN}answer{OFF}    ")
            case TextMessageContentEvent():
                lines[index[event.message_id]] += event.delta
            case RunErrorEvent():
                lines.append(f"  {RED}error{OFF}     {event.message}")
    return lines


def generated(built: BuiltAgent) -> AsyncIterator[BaseEvent]:
    """The events as `agui_events` yields them, with nothing in between."""
    turn = stream_turn(built.agent, QUESTION, THREAD)
    return agui_events(
        turn,
        thread_id=THREAD,
        run_id=RUN,
        tools={tool.name: tool for tool in built.tools},
        withheld=built.withheld,
    )


async def over_http(base: str) -> AsyncIterator[BaseEvent]:
    """The same events, POSTed to `mcp_agent_api.routes` and read back off SSE.

    Decoded from the frames rather than reused, so anything the wire cannot
    carry — a field the encoder drops, a shape the union will not validate —
    shows up here instead of passing unnoticed.
    """
    body = {
        "threadId": THREAD,
        "runId": RUN,
        "messages": [{"id": "u1", "role": "user", "content": QUESTION}],
    }
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", f"{base}/runs", json=body) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    yield EVENT.validate_json(line.removeprefix("data: "))


async def show_routes(base: str) -> None:
    """What a client fetches once the turn is over.

    The three read routes exist because the stream deliberately does not carry
    this: the transcript so a reload can restore it, the payload behind a state
    key, and the view bundle behind a `ui://` URI.
    """
    async with httpx.AsyncClient(timeout=None) as client:
        thread = (await client.get(f"{base}/threads/{THREAD}")).json()
        print(f"\n{BOLD}GET /threads/{THREAD}{OFF}")
        for message in thread["messages"]:
            summary = message.get("content") or message.get("toolCalls") or ""
            print(f"  {DIM}{message['role']:10}{OFF}{str(summary)[:70]}")

        for key, entry in thread["state"].items():
            state = (await client.get(f"{base}/threads/{THREAD}/state/{key}")).json()
            print(f"\n{BOLD}GET /threads/{THREAD}/state/{key}{OFF}")
            print(
                f"  {DIM}the stream carried{OFF} "
                f"{json.dumps(entry, separators=(',', ':'))}"
            )
            print(f"  {DIM}this carries{OFF}       {json.dumps(state['value'])[:70]}…")

        toolset, view = VIEW
        bundle = await client.get(f"{base}/views/{toolset}/{view}")
        print(f"\n{BOLD}GET /views/{toolset}/{view}{OFF}")
        print(f"  {DIM}{len(bundle.text)} bytes of HTML{OFF}  {bundle.text[:60]}…")


async def play(scenario: Scenario, *, step: bool, raw: bool, serve_api: bool) -> None:
    tools: list[BaseTool] = [
        tool
        for tool in await MultiServerMCPClient(CONNECTIONS).get_tools()
        if TOOLSET_OF[tool.name] in scenario.toolsets
    ]
    agent, withheld = with_session_state(
        StreamingScriptedModel(script=scenario.script), tools, InMemorySaver()
    )
    built = BuiltAgent(agent, CONNECTIONS, tools, withheld, None)

    print(f"\n{BOLD}{scenario.title}{OFF}\n{scenario.blurb}\n")
    print(
        f"{DIM}{len(tools)} tool(s) connected, {len(withheld)} withheld · "
        f"[Enter] next event · [c] run on · [q] back{OFF}\n"
    )

    base = ""
    if serve_api:
        # A fresh app per scenario: each one connects a different set of tools,
        # and an app holds the agent its factory returned. The other half of
        # `create_app`'s contract — the agent exists already, so the factory
        # hands it straight over rather than connecting anything.
        async def ready() -> BuiltAgent:
            return built

        port = free_port()
        serve(create_app(ready), port)
        await wait_for(port)
        base = f"http://127.0.0.1:{port}"
        print(f"{DIM}POST {base}/runs{OFF}\n")

    encoder = EventEncoder()
    collected: list[Any] = []

    async for event in over_http(base) if serve_api else generated(built):
        collected.append(event)
        print(f"  {BOLD}{event.type.value:24}{OFF}{summarise(event)}")
        if raw:
            for line in encoder.encode(event).strip().splitlines():
                print(f"    {DIM}{line[:150]}{OFF}")
        if step:
            answer = await asyncio.to_thread(input, "")
            if answer.strip().lower() == "q":
                return
            step = answer.strip().lower() != "c"

    print(f"\n{BOLD}What a client renders, in order{OFF}")
    print("\n".join(transcript(collected)))
    print(f"\n{DIM}{len(collected)} events{OFF}")

    if serve_api and collected and collected[-1].type.value == "RUN_FINISHED":
        await show_routes(base)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="do not step")
    parser.add_argument("--raw", action="store_true", help="show encoded SSE frames")
    parser.add_argument(
        "--serve", action="store_true", help="go through mcp_agent_api.routes over HTTP"
    )
    parser.add_argument("--scenario", type=int, help="run one and exit")
    parser.add_argument(
        "--api",
        nargs="?",
        type=int,
        const=8765,
        help="serve the API for web/ and stay up (default port 8765)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.04,
        help="pause between the stub's tokens; ignored when a real model runs",
    )
    args = parser.parse_args()

    global CONNECTIONS, TOOLSET_OF
    ports = {name: free_port() for name in [*SERVERS, FOREIGN]}
    for toolset, module in SERVERS.items():
        server = build_server(
            toolset=toolset,
            module_name=module,
            host="127.0.0.1",
            port=ports[toolset],
        )
        serve(server.streamable_http_app(), ports[toolset])
    serve(foreign_mcp.streamable_http_app(), ports[FOREIGN])
    for port in ports.values():
        await wait_for(port)

    CONNECTIONS = {
        name: {"transport": "streamable_http", "url": f"http://127.0.0.1:{port}/mcp"}
        for name, port in ports.items()
    }
    client = MultiServerMCPClient(CONNECTIONS)
    TOOLSET_OF = {}
    for name in CONNECTIONS:
        for tool in await client.get_tools(server_name=name):
            TOOLSET_OF[tool.name] = name

    if args.api:
        await api(args.api, args.delay)
        return

    surface = "app, over HTTP" if args.serve else "events, in process"
    print(f"\n{BOLD}mcp_agent_api — one turn as AG-UI ({surface}){OFF}")
    print(f"{DIM}4 MCP servers on 127.0.0.1:{min(ports.values())}…{OFF}")

    options = {"step": not args.run, "raw": args.raw, "serve_api": args.serve}
    if args.scenario:
        await play(SCENARIOS[args.scenario - 1], **options)
        return

    while True:
        print()
        for number, scenario in enumerate(SCENARIOS, 1):
            print(f"  {number}. {scenario.title}")
        choice = (await asyncio.to_thread(input, "\nscenario (q to quit): ")).strip()
        if choice.lower() in {"q", ""}:
            return
        if choice.isdigit() and 1 <= int(choice) <= len(SCENARIOS):
            await play(SCENARIOS[int(choice) - 1], **options)


if __name__ == "__main__":
    asyncio.run(main())
