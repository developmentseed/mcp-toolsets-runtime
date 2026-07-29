"""Whether an injected parameter has anything that could ever fill it."""

from typing import Any

import pytest
from langchain_core.tools import StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_tool

from mcp_runtime.injected import INJECTED_META_KEY, PRODUCES_META_KEY
from mcp_runtime.kinds import GEOJSON_AREA_OF_INTEREST, GEOJSON_FOOTPRINT
from mcp_state.injection import bind_all_injected
from mcp_state.wiring import raise_unsatisfiable, unsatisfiable, usable

PUBLISHES_AOI = {
    PRODUCES_META_KEY: [
        {
            "stateKey": "dataset-search/geometry",
            "field": "geometry",
            "kind": GEOJSON_AREA_OF_INTEREST,
        }
    ]
}


def injects(kind: str, *, required: bool = True) -> dict[str, Any]:
    return {
        INJECTED_META_KEY: [{"parameter": "aoi", "kind": kind, "required": required}]
    }


def mcp_tool(name: str, meta: dict[str, Any] | None = None) -> StructuredTool:
    async def call(**arguments: Any) -> Any:
        return "ok", None

    return StructuredTool(
        name=name,
        description=name,
        args_schema={"type": "object", "properties": {}},
        coroutine=call,
        metadata={"_meta": meta} if meta else None,
    )


def test_a_connected_publisher_satisfies_a_consumer() -> None:
    tools = [
        mcp_tool("search", PUBLISHES_AOI),
        mcp_tool("clip", injects(GEOJSON_AREA_OF_INTEREST)),
    ]
    assert unsatisfiable(tools) == []


def test_a_consumer_with_no_publisher_is_reported() -> None:
    (found,) = unsatisfiable([mcp_tool("clip", injects(GEOJSON_AREA_OF_INTEREST))])
    assert (found.tool, found.parameter, found.wants) == (
        "clip",
        "aoi",
        GEOJSON_AREA_OF_INTEREST,
    )
    assert found.required


def test_a_mistyped_kind_shows_up_as_unsatisfiable() -> None:
    """What lets the vocabulary stay open: the wiring is checked, not the name."""
    tools = [
        mcp_tool("search", PUBLISHES_AOI),
        mcp_tool("clip", injects("geojson.AreaofInterest")),
    ]
    (found,) = unsatisfiable(tools)
    assert found.wants == "geojson.AreaofInterest"


def test_a_publisher_of_the_wrong_kind_does_not_count() -> None:
    publishes_footprint = {
        PRODUCES_META_KEY: [
            {"stateKey": "x/f", "field": "f", "kind": GEOJSON_FOOTPRINT}
        ]
    }
    tools = [
        mcp_tool("coverage", publishes_footprint),
        mcp_tool("clip", injects(GEOJSON_AREA_OF_INTEREST)),
    ]
    assert len(unsatisfiable(tools)) == 1


def test_an_explicit_state_key_is_checked_against_published_keys() -> None:
    wants_key = {
        INJECTED_META_KEY: [
            {"parameter": "aoi", "stateKey": "other/geometry", "required": True}
        ]
    }
    tools = [mcp_tool("search", PUBLISHES_AOI), mcp_tool("clip", wants_key)]
    (found,) = unsatisfiable(tools)
    assert found.wants == "key:other/geometry"


def test_an_unfillable_tool_is_withheld_from_the_agent() -> None:
    """The model is never offered a tool whose every call would raise.

    No publisher of the kind `clip` wants is connected, so it is withheld —
    and only it: `weather` is untouched.
    """
    tools = [mcp_tool("weather"), mcp_tool("clip", injects(GEOJSON_AREA_OF_INTEREST))]
    agent_tools, withheld = usable(tools)
    assert [tool.name for tool in agent_tools] == ["weather"]
    assert [item.tool for item in withheld] == ["clip"]


def test_a_satisfiable_tool_stays_available_before_anything_is_published() -> None:
    """Satisfiability is about a connected publisher, not a published value.

    `clip` must stay callable so the model can be told to run `search` first —
    the error path in scenario 4 of docs/SESSION-STATE.md. `usable` is handed
    no state at all, which is what guarantees it.
    """
    tools = [
        mcp_tool("search", PUBLISHES_AOI),
        mcp_tool("clip", injects(GEOJSON_AREA_OF_INTEREST)),
    ]
    agent_tools, withheld = usable(tools)
    assert {tool.name for tool in agent_tools} == {"search", "clip"}
    assert withheld == []


def test_an_optional_parameter_never_withholds_its_tool() -> None:
    tools = [mcp_tool("clip", injects(GEOJSON_AREA_OF_INTEREST, required=False))]
    agent_tools, withheld = usable(tools)
    assert [tool.name for tool in agent_tools] == ["clip"]
    assert withheld == []
    assert len(unsatisfiable(tools)) == 1  # still reported


def test_third_party_tools_are_never_implicated() -> None:
    tools = [mcp_tool("weather"), mcp_tool("search", PUBLISHES_AOI)]
    assert unsatisfiable(tools) == []
    assert len(usable(tools)[0]) == 2


def test_a_fallback_parameter_is_handed_back_to_the_model() -> None:
    """With no publisher connected, the tool degrades to plain MCP.

    The parameter is in the server's advertised schema either way, so leaving
    it there is exactly what a client implementing none of this would do — and
    strictly better than deleting a usable tool.
    """
    falls_back = {
        INJECTED_META_KEY: [
            {
                "parameter": "bbox",
                "kind": "geo.BoundingBox",
                "required": True,
                "modelFallback": True,
            }
        ]
    }
    tool = StructuredTool(
        name="clip",
        description="clip",
        args_schema={
            "type": "object",
            "properties": {"id": {"type": "string"}, "bbox": {"type": "array"}},
            "required": ["id", "bbox"],
        },
        coroutine=mcp_tool("x").coroutine,
        metadata={"_meta": falls_back},
    )
    agent_tools, withheld = usable([tool])
    assert agent_tools == [tool]
    assert withheld == []

    (reported,) = unsatisfiable([tool])
    assert reported.model_fallback and not reported.fatal

    bound = bind_all_injected([tool])[0]
    assert (
        "bbox" in convert_to_openai_tool(bound)["function"]["parameters"]["properties"]
    )


def test_a_fallback_parameter_is_still_injected_when_it_can_be() -> None:
    """Fallback is the degraded path, not the normal one."""
    falls_back = {
        INJECTED_META_KEY: [
            {
                "parameter": "aoi",
                "kind": GEOJSON_AREA_OF_INTEREST,
                "required": True,
                "modelFallback": True,
            }
        ]
    }
    tool = StructuredTool(
        name="clip",
        description="clip",
        args_schema={
            "type": "object",
            "properties": {"aoi": {"type": "object"}},
            "required": ["aoi"],
        },
        coroutine=mcp_tool("x").coroutine,
        metadata={"_meta": falls_back},
    )
    bound = bind_all_injected([mcp_tool("search", PUBLISHES_AOI), tool])[1]
    parameters = convert_to_openai_tool(bound)["function"]["parameters"]
    assert "aoi" not in parameters["properties"]


def test_raise_unsatisfiable_names_the_broken_wire() -> None:
    tools = [mcp_tool("clip", injects(GEOJSON_AREA_OF_INTEREST))]
    with pytest.raises(RuntimeError, match="clip.aoi"):
        raise_unsatisfiable(tools)


def test_raise_unsatisfiable_ignores_non_fatal_by_default() -> None:
    tools = [mcp_tool("clip", injects(GEOJSON_AREA_OF_INTEREST, required=False))]
    raise_unsatisfiable(tools)
    with pytest.raises(RuntimeError):
        raise_unsatisfiable(tools, fatal_only=False)
