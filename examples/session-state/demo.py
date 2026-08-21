"""Run the whole session-state contract against three real MCP servers.

Two are built with this runtime and declare what they publish; the third
(``foreign_server.py``) is raw FastMCP and declares nothing at all. The demo
connects one agent to all three and drives a conversation in which a
2000-vertex geometry moves between two servers that share no code, no imports
and no vocabulary — only a **name**.

- ``search_datasets`` returns an ``area_of_interest`` data key, captured to
  ``dataset-search/search_datasets/area_of_interest``.
- The model reads that key off a breadcrumb and hands it to ``clip_raster`` as
  ``@state:<key>``. That parameter is ``NotAuthored``, so its schema accepts a
  handle and nothing else — the model could not have written a geometry into
  it even if it tried, which the last section demonstrates by trying.
- The foreign server's ``describe_geometry`` takes a structured parameter
  nobody declared. It gains the handle branch as well, and the model points it
  at the same value. Its ``elevation_profile`` returns a large array nobody
  declared, captured on size alone.

Nothing is stubbed: the declarations travel as MCP ``_meta`` over the wire, and
the foreign server genuinely carries none. The chat model *is* stubbed — it
replays a fixed script — so the demo needs no API key and no network.

    uv run python examples/session-state/demo.py
"""

import asyncio
import json
import logging
import socket
import sys
import textwrap
import threading
from pathlib import Path
from typing import Any

import uvicorn
from langchain.agents import create_agent
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_mcp_adapters.client import MultiServerMCPClient

from mcp_runtime.declarations import NOT_AUTHORED_META_KEY, PRODUCES_META_KEY
from mcp_runtime.server import build_server
from mcp_state import (
    StateCaptureMiddleware,
    StateRefusal,
    bind_all_injected,
    handle_for,
    make_inspect_state,
    owners,
    publications,
    state_keys,
    with_server_name,
)

# `build_server` imports a toolset by name at call time. These live beside this
# file rather than being installed, so put them on the path.
sys.path.insert(0, str(Path(__file__).parent / "toolsets"))
sys.path.insert(0, str(Path(__file__).parent))

from foreign_server import mcp as foreign_mcp  # noqa: E402  (needs the path above)

# The MCP SDK and httpx log every request at INFO, which buries the report.
for noisy in ("mcp", "httpx", "uvicorn", "sse_starlette"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logging.basicConfig(level=logging.WARNING)

TOOLSETS = ["dataset-search", "raster-ops"]
FOREIGN = "terrain"
AOI_KEY = "dataset-search/search_datasets/area_of_interest"

#: Takes a bounding box a model is welcome to sketch.
GENERATABLE = "preview_extent"
#: Takes the same JSON, and says a model may not write it.
NARROWED = "clip_to_bbox"


class ScriptedModel(GenericFakeChatModel):
    """Replays a fixed run of tool calls, so the demo needs no provider key."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def run_app(app: Any, port: int) -> None:
    """Serve an ASGI app on a daemon thread."""
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    threading.Thread(target=uvicorn.Server(config).run, daemon=True).start()


async def wait_for(port: int, attempts: int = 60) -> None:
    for _ in range(attempts):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                return
        except OSError:
            await asyncio.sleep(0.1)
    raise RuntimeError(f"127.0.0.1:{port} never came up")


def size_of(value: Any) -> str:
    chars = len(json.dumps(value))
    return f"{chars / 1024:.1f} kB" if chars >= 1024 else f"{chars} bytes"


def rule(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m\n{'─' * len(title)}")


def script() -> list[AIMessage]:
    """The run the stub model replays, one AIMessage per turn."""

    def call(index: int, name: str, args: dict[str, Any]) -> AIMessage:
        return AIMessage(
            content="",
            tool_calls=[
                {"name": name, "args": args, "id": str(index), "type": "tool_call"}
            ],
        )

    return [
        call(1, "search_datasets", {"query": "rainfall"}),
        # `aoi` is NotAuthored, so a handle is the only thing its schema
        # accepts — read off the [state updated: …] breadcrumb above.
        call(
            2, "clip_raster", {"dataset_id": "chirps-daily", "aoi": handle_for(AOI_KEY)}
        ),
        # The foreign tool's parameter is untouched, so a literal would have
        # been accepted. The model points it at the same value anyway.
        call(3, "describe_geometry", {"geometry": handle_for(AOI_KEY)}),
        call(4, "elevation_profile", {"region": "Severn catchment"}),
        AIMessage(content="Done — clipped and described your area of interest."),
    ]


async def report_refusals(bound: list[Any], state: dict[str, Any]) -> None:
    """Two calls the binding will not let through, run for real.

    Both are what a model would have to do to get round ``NotAuthored``, and
    both come back as a message addressed to the model rather than an
    exception, so a run recovers instead of ending.
    """
    clip = next(tool for tool in bound if tool.name == "clip_raster")

    print("  A. The model writes a geometry of its own")
    try:
        await clip.coroutine(
            injected_state=state,
            dataset_id="chirps-daily",
            aoi={"type": "FeatureCollection", "features": []},
        )
    except StateRefusal as refusal:
        for line in str(refusal).splitlines():
            print(f"     {line}")

    print(f"\n  B. {NARROWED} called with nothing in state at all")
    narrowed = next(tool for tool in bound if tool.name == NARROWED)
    try:
        await narrowed.coroutine(injected_state={}, dataset_id="chirps-daily")
    except StateRefusal as refusal:
        for line in str(refusal).splitlines():
            print(f"     {line}")

    print(
        "\n  Neither is raised at the model: both arrive as an error result, so\n"
        "  the model reads them, runs a producer, and retries."
    )


async def main() -> None:
    ports = {name: free_port() for name in [*TOOLSETS, FOREIGN]}
    for toolset in TOOLSETS:
        server = build_server(toolset=toolset, host="127.0.0.1", port=ports[toolset])
        run_app(server.streamable_http_app(), ports[toolset])
    run_app(foreign_mcp.streamable_http_app(), ports[FOREIGN])

    connections = {
        name: {"transport": "streamable_http", "url": f"http://127.0.0.1:{port}/mcp"}
        for name, port in ports.items()
    }
    for port in ports.values():
        await wait_for(port)

    client = MultiServerMCPClient(connections)
    # Stamped with where each came from, so an undeclared capture is keyed the
    # same three-part way a declared one is. The adapter takes a `server_name`
    # and records it nowhere, so a host that wants it does this itself.
    tools = [
        with_server_name(tool, server)
        for server in connections
        for tool in await client.get_tools(server_name=server)
    ]
    published = publications(tools)

    rule("1. What each server declared, over the wire")
    for tool in sorted(tools, key=lambda t: t.name):
        meta = (tool.metadata or {}).get("_meta") or {}
        label = tool.name
        for declaration in meta.get(PRODUCES_META_KEY, []):
            print(f"  {label:19} publishes  {declaration['stateKey']}")
            label = ""
        for parameter in meta.get(NOT_AUTHORED_META_KEY, []):
            print(f"  {label:19} will not let a model write  {parameter!r}")
            label = ""
        if not meta.get(PRODUCES_META_KEY) and not meta.get(NOT_AUTHORED_META_KEY):
            print(f"  {label:19} declares nothing")
    print(
        "\n  No kinds, no shared vocabulary: the only thing crossing between\n"
        "  toolsets is the state key, which is a name a model reads."
    )

    bound = bind_all_injected(tools)

    rule("2. What the model is offered")
    for name in ("clip_raster", GENERATABLE, NARROWED, "describe_geometry"):
        bound_tool = next(tool for tool in bound if tool.name == name)
        served = next(tool for tool in tools if tool.name == name)
        offered = convert_to_openai_tool(bound_tool)["function"]["parameters"]
        print(f"  {name}")
        print(f"    server advertises: {sorted(served.args_schema['properties'])}")
        print(f"    model is offered:  {sorted(offered['properties'])}")
        for parameter, schema in sorted(offered["properties"].items()):
            if "anyOf" in schema:
                print(f"      {parameter}: a value, or an @state:<key> handle")
            elif schema.get("pattern") == "^@state:":
                print(f"      {parameter}: an @state:<key> handle and nothing else")

    agent = create_agent(
        ScriptedModel(messages=iter(script())),
        [*bound, make_inspect_state(state_keys(published))],
        middleware=[StateCaptureMiddleware(published, owners=owners(tools))],
    )
    result = await agent.ainvoke(
        {"messages": [HumanMessage("find rainfall data and clip chirps to my area")]}
    )

    rule("3. The conversation the model actually saw")
    for message in result["messages"]:
        # `.text` rather than `str(.content)`: a server answering in content
        # blocks makes that a list, and the model is shown the text, not a
        # Python repr of the blocks carrying it.
        if not (text := message.text.strip()):
            continue
        label = message.type
        for line in text.splitlines():
            for wrapped in textwrap.wrap(line, width=86):
                print(f"  {label:9} {wrapped}")
                label = ""

    rule("4. Session state")
    declared_keys = state_keys(published)
    for key, entry in sorted(
        result["tool_state"].items(), key=lambda item: item[1].get("seq", 0)
    ):
        how = "declared" if key in declared_keys else "captured on size"
        print(f"  {key}")
        print(f"    {size_of(entry['value'])}, from {entry['tool']}  ({how})")

    rule("5. Did the payload ever enter the transcript?")
    # The whole serialised content, content blocks included: this asks whether a
    # vertex leaked anywhere into the transcript, so it searches everything.
    transcript = " ".join(str(message.content) for message in result["messages"])
    aoi = result["tool_state"][AOI_KEY]["value"]
    ring = aoi["features"][0]["geometry"]["coordinates"][0]
    vertices = sum(
        len(ring) for f in aoi["features"] for ring in f["geometry"]["coordinates"]
    )
    # An interior vertex: the bounds describe_geometry computed legitimately
    # include the corners, so testing one of those would be a false positive.
    interior = repr(ring[len(ring) // 2][0])
    print(f"  area of interest:      {size_of(aoi)}")
    print(f"  whole transcript:      {size_of(transcript)}")
    print(f"  a vertex ({interior}) in it: {'yes' if interior in transcript else 'no'}")
    print(
        f"\n  Both clip_raster and describe_geometry ran against the same\n"
        f"  {vertices}-vertex geometry, each pointed at it by a key costing about\n"
        "  ten tokens. Neither server received it from the model."
    )

    rule("6. What the binding refuses")
    await report_refusals(bound, result["tool_state"])


if __name__ == "__main__":
    asyncio.run(main())
