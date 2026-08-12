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

Then the two degradation cases, from the same connected servers. ``raster-ops``
carries two tools taking a ``geo.BoundingBox``, which nothing here declares it
publishes, and they differ only in whether a model may invent one:
``preview_extent`` keeps its parameter and gains a handle branch, while
``clip_to_bbox`` is withheld from the agent before a model ever sees it — even
though a *detected* bounding box in state would have satisfied it at call time.

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
    StateCaptureMiddleware,
    StateEntry,
    bind_all_injected,
    detect_kind,
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

#: The tools `script()` calls. Anything the wiring check withholds is fine
#: unless it is one of these — scenario F withholds a tool on purpose.
SCRIPTED = frozenset(
    {"search_datasets", "clip_raster", "describe_geometry", "elevation_profile"}
)

#: The two tools that exist to be unsatisfiable. Both take a
#: `geo.BoundingBox`, which nothing here declares it publishes; they differ
#: only in whether a model may invent one.
GENERATABLE = "preview_extent"
WITHHELD = "clip_to_bbox"


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


async def report_degradation(
    tools: list[Any],
    bound: list[Any],
    agent_tools: list[Any],
    withheld: list[Any],
    result: dict[str, Any],
) -> None:
    """Print scenarios E and F: a tagged parameter nothing can satisfy.

    Both tools take a ``geo.BoundingBox`` and differ only in whether a model
    may invent one. Each half says so itself rather than being asserted, so
    publishing that kind — see the README's "things to try" — makes the
    degradations report that they no longer apply instead of misreporting.
    """
    served = {tool.name: tool for tool in tools}
    offered = {tool.name: tool for tool in agent_tools}

    if GENERATABLE not in offered:
        print("  E. Tagged, model may generate")
        print(f"     {GENERATABLE} is withheld, so this is scenario F, not E.")
    else:
        schema = convert_to_openai_tool(offered[GENERATABLE])["function"]["parameters"]
        advertised = sorted(served[GENERATABLE].args_schema["properties"])
        bbox = schema["properties"].get("bbox")
        outcome = (
            "the tag is dropped, nothing else is"
            if bbox is not None
            else "the tag is honoured, so this is scenario A"
        )
        print(f"  E. Tagged, model may generate — {outcome}")
        print(f"     server advertises: {advertised}")
        print(f"     offered to the model: {sorted(schema['properties'])}")
        if bbox is None:
            print("     bbox was filled from state instead — something publishes it.")
        else:
            print(f"     bbox also accepts a handle: {'anyOf' in bbox}")

    taken_away = WITHHELD not in offered
    print(
        f"\n  F. Tagged model_generatable=False — {WITHHELD} "
        f"{'is taken away' if taken_away else 'is offered after all'}"
    )
    declares = "nothing DECLARES" if taken_away else "something DECLARES"
    print(f"     Connect time ({declares} it publishes geo.BoundingBox):")
    print(f"       withheld: {[str(item) for item in withheld]}")
    print(f"       would the host offer it? {'no' if taken_away else 'yes'}")
    if not taken_away:
        print("\n     Something publishes geo.BoundingBox now, so nothing degrades.")
        return

    # The bounds the foreign server really returned in section 4. Capture left
    # them in the transcript — four floats are far below the size gate — so
    # this is the entry `StateCaptureMiddleware(capture_undeclared=…)` would
    # have written had the value been large, labelled by the same detector.
    described = next(
        message
        for message in result["messages"]
        if getattr(message, "name", None) == "describe_geometry"
    )
    bounds = described.artifact["structured_content"]["bounds"]
    detected = StateEntry(
        value=bounds, kind=detect_kind(bounds), tool="describe_geometry"
    )
    _, artifact = await next(tool for tool in bound if tool.name == WITHHELD).coroutine(
        injected_state={"terrain/bounds": detected}, dataset_id="chirps-daily"
    )

    print(f"\n     Call time (a value DETECTED as {detected['kind']} is in state):")
    print(f"       terrain/bounds = {bounds}")
    answer = artifact["structured_content"]["message"]
    print(f"       the withheld tool, run anyway: {answer!r}")
    print(
        "\n  The wiring check reads declarations only, because at connect nothing\n"
        "  has run and a detected kind is a value that may never appear. So it\n"
        "  withholds a tool that would in fact have worked — fail-safe, and the\n"
        "  reason not to put model_generatable=False on a consumer whose producer\n"
        "  is a third-party server."
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
    for item in problems:
        print(f"    {item}")
    print(f"  tools withheld from the agent:         {len(withheld)}")
    print(
        "\n  unsatisfiable() lists every one of them; partition_usable() acts on\n"
        "  the fatal ones alone, so a parameter that merely degrades to the\n"
        "  model leaves its tool callable. Section 7 is those two lines in full."
    )

    if blocked := SCRIPTED.intersection(item.tool for item in withheld):
        print(
            f"\n  {', '.join(sorted(blocked))} withheld, and the scripted run needs\n"
            "  it. Restore the kind its parameter asks for, or drop\n"
            "  model_generatable=False to hand the parameter back to the model."
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
        middleware=[StateCaptureMiddleware(published)],
    )
    result = await agent.ainvoke(
        {"messages": [HumanMessage("find rainfall data and clip chirps to my area")]}
    )

    rule("4. The conversation the model actually saw")
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

    rule("5. Session state")
    declared_keys = state_keys(published)
    for key, entry in sorted(
        result["tool_state"].items(), key=lambda item: item[1].get("seq", 0)
    ):
        how = "declared" if key in declared_keys else "captured on size"
        print(f"  {key}  —  {size_of(entry['value'])}, from {entry['tool']}")
        print(f"    kind={entry['kind'] or 'unrecognised'}  ({how})")

    rule("6. Did the payload ever enter the transcript?")
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
        f"  {vertices}-vertex geometry. One found it by kind and never showed the\n"
        "  model the parameter; the other was pointed at it by name, in about ten\n"
        "  tokens. Neither server received it from the model."
    )

    rule("7. Degradation: a kind nothing publishes")
    await report_degradation(tools, bound, agent_tools, withheld, result)


if __name__ == "__main__":
    asyncio.run(main())
