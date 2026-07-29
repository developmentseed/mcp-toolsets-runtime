"""The whole contract, wired the way a consuming agent wires it.

Everything else tests a part; this drives ``create_agent`` end to end over a
mix of toolsets that participate and a third-party server that does not, and
asserts the property the design exists for: a large value moves from the tool
that produced it to the tool that needs it *without ever being in the
transcript*.
"""

from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool

from mcp_runtime.injected import INJECTED_META_KEY, PRODUCES_META_KEY
from mcp_runtime.kinds import GEOJSON_AREA_OF_INTEREST
from mcp_state import (
    AgentState,
    StateCaptureMiddleware,
    bind_all_injected,
    make_inspect_state,
    publications,
    state_keys,
)

AOI = {"type": "FeatureCollection", "features": ["<100kb of coordinates>"]}


class Scripted(GenericFakeChatModel):
    """A fake model that accepts ``bind_tools`` so ``create_agent`` drives it."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self


def mcp_tool(
    name: str,
    properties: dict[str, Any],
    required: list[str],
    *,
    meta: dict[str, Any] | None = None,
    returns: dict[str, Any] | None = None,
    seen: dict[str, Any],
) -> StructuredTool:
    """A stand-in for a tool the adapter loaded from an MCP server."""

    async def call(runtime: Any = None, **arguments: Any) -> Any:
        seen[name] = arguments
        return "ok", ({"structured_content": returns} if returns else None)

    return StructuredTool(
        name=name,
        description=name,
        args_schema={
            "type": "object",
            "properties": properties,
            "required": required,
        },
        coroutine=call,
        response_format="content_and_artifact",
        metadata={"_meta": meta} if meta else None,
    )


def call_of(name: str, args: dict[str, Any], id_: str) -> dict[str, Any]:
    return {"name": name, "args": args, "id": id_, "type": "tool_call"}


async def test_a_payload_crosses_servers_without_entering_the_transcript() -> None:
    seen: dict[str, Any] = {}
    # "dataset-search": publishes an area of interest.
    search = mcp_tool(
        "search",
        {"q": {"type": "string"}},
        ["q"],
        meta={
            PRODUCES_META_KEY: [
                {
                    "stateKey": "dataset-search/geometry",
                    "field": "geometry",
                    "kind": GEOJSON_AREA_OF_INTEREST,
                }
            ]
        },
        returns={"message": "found 3 datasets", "geometry": AOI},
        seen=seen,
    )
    # "raster-ops", a different server: consumes one, naming only its kind.
    clip = mcp_tool(
        "clip",
        {"id": {"type": "string"}, "aoi": {"type": "object"}},
        ["id", "aoi"],
        meta={
            INJECTED_META_KEY: [
                {
                    "parameter": "aoi",
                    "kind": GEOJSON_AREA_OF_INTEREST,
                    "required": True,
                }
            ]
        },
        seen=seen,
    )
    # A third-party MCP server, declaring nothing.
    weather = mcp_tool("weather", {"city": {"type": "string"}}, ["city"], seen=seen)

    tools = [search, clip, weather]
    published = publications(tools)
    agent = create_agent(
        Scripted(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[call_of("search", {"q": "era5"}, "1")],
                    ),
                    AIMessage(
                        content="",
                        tool_calls=[
                            call_of("clip", {"id": "era5"}, "2"),
                            call_of("weather", {"city": "Reading"}, "3"),
                        ],
                    ),
                    AIMessage(content="done"),
                ]
            )
        ),
        [*bind_all_injected(tools), make_inspect_state(state_keys(published))],
        state_schema=AgentState,
        middleware=[StateCaptureMiddleware(published)],
    )

    result = await agent.ainvoke({"messages": [HumanMessage("clip era5 to my aoi")]})

    # Published under its qualified key, and only there.
    assert list(result["tool_state"]) == ["dataset-search/geometry"]
    # The model got the message and a breadcrumb, never the payload.
    assert not any("coordinates" in str(m.content) for m in result["messages"])
    assert "[state updated: dataset-search/geometry" in result["messages"][2].content
    # The consuming tool on the other server got it anyway.
    assert seen["clip"]["aoi"] == AOI
    # The third-party tool is entirely unaffected.
    assert seen["weather"] == {"city": "Reading"}


async def test_injection_works_without_the_middleware_but_capture_does_not() -> None:
    """The two halves have different requirements, which is easy to trip over.

    Injection is an ``InjectedState`` mechanism, so it runs wherever LangGraph
    executes a tool. Capture is *agent middleware* — a consumer assembling a
    bare ``StateGraph``/``ToolNode`` instead of using ``create_agent`` gets
    injection and silently gets no capture, leaving nothing to inject.
    """
    seen: dict[str, Any] = {}
    search = mcp_tool(
        "search",
        {"q": {"type": "string"}},
        ["q"],
        meta={
            PRODUCES_META_KEY: [
                {
                    "stateKey": "dataset-search/geometry",
                    "field": "geometry",
                    "kind": GEOJSON_AREA_OF_INTEREST,
                }
            ]
        },
        returns={"message": "found", "geometry": AOI},
        seen=seen,
    )
    agent = create_agent(
        Scripted(
            messages=iter(
                [
                    AIMessage(
                        content="", tool_calls=[call_of("search", {"q": "x"}, "1")]
                    ),
                    AIMessage(content="done"),
                ]
            )
        ),
        bind_all_injected([search]),
        state_schema=AgentState,
        # no middleware
    )
    result = await agent.ainvoke({"messages": [HumanMessage("search")]})
    assert "tool_state" not in result or not result["tool_state"]
