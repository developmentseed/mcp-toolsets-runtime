"""The whole contract, wired the way a consuming agent wires it.

Everything else tests a part; this drives ``create_agent`` end to end over a
mix of toolsets that participate and third-party servers that do not, and
asserts the property the design exists for: a large value moves from the tool
that produced it to the tools that need it *without ever being in the
transcript* — by declaration on one, and by handle on the other.
"""

from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool

from mcp_runtime.declarations import CONSUMES_META_KEY, PRODUCES_META_KEY
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
            CONSUMES_META_KEY: [
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
    # Also third-party, but with a structured parameter — so the client offers
    # it as a handle and the model can point it at the same geometry by name.
    describe = mcp_tool(
        "describe", {"geometry": {"type": "object"}}, ["geometry"], seen=seen
    )

    tools = [search, clip, weather, describe]
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
                            call_of(
                                "describe",
                                {"geometry": "@state:dataset-search/geometry"},
                                "4",
                            ),
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
    # And so did the third-party one, which declared nothing: the model named
    # the key, the client swapped in the payload before the call.
    assert seen["describe"]["geometry"] == AOI
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


async def test_capture_needs_no_state_schema_from_the_host() -> None:
    """The middleware carries the channel it writes to.

    A host that adds the middleware but forgets ``state_schema=AgentState``
    would otherwise lose every captured payload in silence: the value is
    already stripped off the tool message by the time the write is dropped,
    and the model is told to go and read a key that was never written.
    """
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
        seen={},
    )
    published = publications([search])
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
        [search],
        middleware=[StateCaptureMiddleware(published)],
        # deliberately no state_schema=AgentState
    )
    result = await agent.ainvoke({"messages": [HumanMessage("search")]})

    assert result["tool_state"]["dataset-search/geometry"]["value"] == AOI
    # The key the breadcrumb advertises is one `inspect_state` can now read.
    assert "[state updated: dataset-search/geometry" in result["messages"][2].content


async def test_a_host_may_still_bring_its_own_state_schema() -> None:
    """Middleware schemas are merged, so a host subclass keeps its own channels."""

    class HostState(AgentState):
        run_id: str

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
        seen={},
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
        [search],
        state_schema=HostState,
        middleware=[StateCaptureMiddleware(publications([search]))],
    )
    result = await agent.ainvoke({"messages": [HumanMessage("search")], "run_id": "r1"})

    assert result["run_id"] == "r1"
    assert result["tool_state"]["dataset-search/geometry"]["value"] == AOI
