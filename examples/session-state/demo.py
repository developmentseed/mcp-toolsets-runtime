"""Run the whole session-state contract against two real MCP servers.

Serves ``dataset_search`` and ``raster_ops`` over HTTP on separate ports,
connects an agent to both, and drives one conversation that makes the
producing tool publish an area of interest and the consuming tool receive it.

Nothing is stubbed: the declarations travel as MCP ``_meta`` over the wire,
which is the part worth seeing work. The chat model *is* stubbed — it replays
a fixed script — so the demo needs no API key and no network.

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

from mcp_runtime.injected import INJECTED_META_KEY, PRODUCES_META_KEY
from mcp_runtime.server import build_server
from mcp_state import (
    AgentState,
    StateCaptureMiddleware,
    bind_all_injected,
    make_inspect_state,
    partition_usable,
    publications,
    state_keys,
    unsatisfiable,
)

# `build_server` imports a toolset by name at call time. These two live beside
# this file rather than being installed, so put them on the path.
sys.path.insert(0, str(Path(__file__).parent / "toolsets"))

# The MCP SDK and httpx log every request at INFO, which buries the report.
for noisy in ("mcp", "httpx", "uvicorn", "sse_starlette"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logging.basicConfig(level=logging.WARNING)

TOOLSETS = ["dataset-search", "raster-ops"]


class ScriptedModel(GenericFakeChatModel):
    """Replays a fixed run of tool calls, so the demo needs no provider key."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def serve(toolset: str, port: int) -> None:
    """Run one toolset's MCP server in a daemon thread."""
    server = build_server(toolset=toolset, host="127.0.0.1", port=port)
    config = uvicorn.Config(
        server.streamable_http_app(), host="127.0.0.1", port=port, log_level="error"
    )
    threading.Thread(target=uvicorn.Server(config).run, daemon=True).start()


async def wait_for(url: str, attempts: int = 50) -> None:
    for _ in range(attempts):
        try:
            with socket.create_connection(
                (url.split("/")[2].split(":")[0], int(url.split(":")[2].split("/")[0])),
                timeout=0.1,
            ):
                return
        except OSError:
            await asyncio.sleep(0.1)
    raise RuntimeError(f"{url} never came up")


def size_of(value: Any) -> str:
    chars = len(json.dumps(value))
    return f"{chars / 1024:.1f} kB" if chars >= 1024 else f"{chars} bytes"


def rule(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m\n{'─' * len(title)}")


async def main() -> None:
    ports = {toolset: free_port() for toolset in TOOLSETS}
    for toolset, port in ports.items():
        serve(toolset, port)
    connections = {
        toolset: {
            "transport": "streamable_http",
            "url": f"http://127.0.0.1:{port}/mcp",
        }
        for toolset, port in ports.items()
    }
    for connection in connections.values():
        await wait_for(connection["url"])

    tools = await MultiServerMCPClient(connections).get_tools()
    published = publications(tools)

    rule("1. What each server declared, over the wire")
    for tool in sorted(tools, key=lambda t: t.name):
        meta = (tool.metadata or {}).get("_meta") or {}
        label = tool.name
        for declaration in meta.get(PRODUCES_META_KEY, []):
            kind = declaration["kind"] or "(no kind — captured, but not injectable)"
            print(f"  {label:16} publishes  {declaration['stateKey']:25} {kind}")
            label = ""
        for declaration in meta.get(INJECTED_META_KEY, []):
            wanted = declaration.get("kind") or f"key:{declaration.get('stateKey')}"
            needed = "required" if declaration.get("required", True) else "optional"
            print(
                f"  {label:16} consumes   {declaration['parameter']:25} "
                f"{wanted} ({needed})"
            )
            label = ""

    rule("2. Wiring check, before the agent is built")
    problems = unsatisfiable(tools)
    bound = bind_all_injected(tools)
    agent_tools, withheld = partition_usable(bound)
    print(f"  injected parameters nothing publishes: {len(problems)}")
    print(f"  tools withheld from the agent:         {len(withheld)}")

    if withheld:
        for item in withheld:
            print(f"\n  {item}")
        print(
            "\n  Nothing connected publishes what that parameter needs, so every\n"
            "  call would raise and the tool is not offered to the model. The rest\n"
            "  of this demo needs the wire intact — restore the kind, or declare\n"
            "  model_fallback=True to hand the parameter back to the model."
        )
        return

    rule("3. What the model is offered for clip_raster")
    clip = next(tool for tool in agent_tools if tool.name == "clip_raster")
    served = next(tool for tool in tools if tool.name == "clip_raster")
    print(f"  server advertises: {sorted(served.args_schema['properties'])}")
    print(
        "  model is offered:  "
        f"{sorted(convert_to_openai_tool(clip)['function']['parameters']['properties'])}"
        "   <- aoi is gone"
    )

    agent = create_agent(
        ScriptedModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "search_datasets",
                                "args": {"query": "rainfall"},
                                "id": "1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "clip_raster",
                                "args": {"dataset_id": "chirps-daily"},
                                "id": "2",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="Done — clipped to your area of interest."),
                ]
            )
        ),
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
    for key, entry in result["tool_state"].items():
        note = (
            "" if entry["kind"] else "   (untagged: captured, but cannot be injected)"
        )
        print(f"  {key}  —  {size_of(entry['value'])}, from {entry['tool']}")
        print(f"    kind={entry['kind']}{note}")

    rule("6. Did the payload ever enter the transcript?")
    transcript = " ".join(str(message.content) for message in result["messages"])
    aoi = result["tool_state"]["dataset-search/geometry"]["value"]
    print(f"  area of interest:      {size_of(aoi)}")
    print(f"  whole transcript:      {size_of(transcript)}")
    print(f"  coordinates in it:     {'yes' if '-3.0' in transcript else 'no'}")
    print(
        "\n  clip_raster ran against a "
        f"{sum(len(r) for f in aoi['features'] for r in f['geometry']['coordinates'])}"
        "-vertex geometry the model never generated, read, or paid for."
    )


if __name__ == "__main__":
    asyncio.run(main())
