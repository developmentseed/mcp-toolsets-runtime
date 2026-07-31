"""Run the whole session-state contract against three real MCP servers.

Two are built with this runtime and declare what they exchange; the third
(``foreign_server.py``) is raw FastMCP and declares nothing at all. The demo
connects one agent to all three and drives a conversation that exercises both
paths:

- **Declared.** ``search_datasets`` publishes an area of interest, tagged with
  a ``Kind``. ``clip_raster`` takes one, tagged with the same ``Kind``. The
  client matches them, and ``aoi`` never appears in the model's schema.
- **Undeclared.** The foreign server's ``describe_geometry`` takes a structured
  parameter nobody declared. The client offers it as an ``@state:<key>``
  handle, and the model points it at the same geometry by name. Its
  ``elevation_profile`` returns a large array nobody declared, captured on
  size alone.

Nothing is stubbed: the declarations travel as MCP ``_meta`` over the wire,
and the foreign server genuinely carries none. The chat model *is* stubbed —
it replays a fixed script — so the demo needs no API key and no network.

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

from mcp_runtime.declarations import CONSUMES_META_KEY, PRODUCES_META_KEY
from mcp_runtime.server import build_server
from mcp_state import (
    AgentState,
    StateCaptureMiddleware,
    bind_all_injected,
    handle_for,
    make_inspect_state,
    partition_usable,
    publications,
    state_keys,
    unsatisfiable,
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
AOI_KEY = "dataset-search/geometry"


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
        # No `aoi`: it is not in this tool's schema at all.
        call(2, "clip_raster", {"dataset_id": "chirps-daily"}),
        # The foreign tool's parameter *is* in the schema, so the model fills
        # it — with a handle it read off the [state updated: …] breadcrumb.
        call(3, "describe_geometry", {"geometry": handle_for(AOI_KEY)}),
        call(4, "elevation_profile", {"region": "Severn catchment"}),
        AIMessage(content="Done — clipped and described your area of interest."),
    ]


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

    tools = await MultiServerMCPClient(connections).get_tools()
    published = publications(tools)

    rule("1. What each server declared, over the wire")
    for tool in sorted(tools, key=lambda t: t.name):
        meta = (tool.metadata or {}).get("_meta") or {}
        label = tool.name
        for declaration in meta.get(PRODUCES_META_KEY, []):
            kind = declaration["kind"] or "(untagged)"
            print(f"  {label:19} publishes  {declaration['stateKey']:25} {kind}")
            label = ""
        for declaration in meta.get(CONSUMES_META_KEY, []):
            policy = (
                "model may generate"
                if declaration.get("modelGeneratable", True)
                else "model must not generate"
            )
            print(
                f"  {label:19} takes      {declaration['parameter']:25} "
                f"{declaration['kind']} ({policy})"
            )
            label = ""
        if not meta.get(PRODUCES_META_KEY) and not meta.get(CONSUMES_META_KEY):
            print(f"  {label:19} declares nothing")

    rule("2. Wiring check, before the agent is built")
    problems = unsatisfiable(tools)
    bound = bind_all_injected(tools)
    agent_tools, withheld = partition_usable(bound)
    print(f"  declared parameters nothing publishes: {len(problems)}")
    print(f"  tools withheld from the agent:         {len(withheld)}")

    if withheld:
        for item in withheld:
            print(f"\n  {item}")
        print(
            "\n  Nothing connected publishes what that parameter needs, and the\n"
            "  tool said a model must not invent it — so every call would raise\n"
            "  and it is not offered. The rest of this demo needs the wire\n"
            "  intact: restore the kind, or drop model_generatable=False to hand\n"
            "  the parameter back to the model."
        )
        return

    rule("3. What the model is offered")
    for name in ("clip_raster", "describe_geometry"):
        bound_tool = next(tool for tool in agent_tools if tool.name == name)
        served = next(tool for tool in tools if tool.name == name)
        offered = convert_to_openai_tool(bound_tool)["function"]["parameters"]
        print(f"  {name}")
        print(f"    server advertises: {sorted(served.args_schema['properties'])}")
        print(f"    model is offered:  {sorted(offered['properties'])}")
        for parameter, schema in sorted(offered["properties"].items()):
            if "anyOf" in schema:
                print(f"      {parameter}: object, or an @state:<key> handle")

    agent = create_agent(
        ScriptedModel(messages=iter(script())),
        [*agent_tools, make_inspect_state(state_keys(published))],
        state_schema=AgentState,
        middleware=[StateCaptureMiddleware(published)],
    )
    result = await agent.ainvoke(
        {"messages": [HumanMessage("find rainfall data and clip chirps to my area")]}
    )

    rule("4. The conversation the model actually saw")
    for message in result["messages"]:
        if not (text := str(message.content).strip()):
            continue
        label = message.type
        for line in text.splitlines():
            for wrapped in textwrap.wrap(line, width=86):
                print(f"  {label:9} {wrapped}")
                label = ""

    rule("5. Session state")
    declared_keys = state_keys(published)
    for key, entry in sorted(
        result["tool_state"].items(), key=lambda item: item[1].get("seq", 0)
    ):
        how = "declared" if key in declared_keys else "captured on size"
        print(f"  {key}  —  {size_of(entry['value'])}, from {entry['tool']}")
        print(f"    kind={entry['kind'] or 'unrecognised'}  ({how})")

    rule("6. Did the payload ever enter the transcript?")
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
        f"  {vertices}-vertex geometry. One found it by kind and never showed the\n"
        "  model the parameter; the other was pointed at it by name, in about ten\n"
        "  tokens. Neither server received it from the model."
    )


if __name__ == "__main__":
    asyncio.run(main())
