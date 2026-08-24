"""The path that needs nothing from the server.

A tool nobody annotated, on a server that has never heard of this project,
still gets session state — the model points a structured parameter at a value
by name, and the client swaps in the payload before the call.
"""

from typing import Any

from langchain_core.tools import StructuredTool, ToolException
from langchain_core.utils.function_calling import convert_to_openai_tool

from mcp_state.detect import describe
from mcp_state.handles import (
    HANDLE_PREFIX,
    available,
    dereference,
    handle_for,
    is_handle,
    offer_handles,
    unresolved,
    unresolved_message,
)
from mcp_state.injection import bind_injected
from mcp_state.state import StateEntry
from tests.mcp_state.test_injection import run_tools, tool_call

AOI = {"type": "FeatureCollection", "features": [{"id": "big-polygon"}]}


def foreign_tool(
    properties: dict[str, Any], seen: dict[str, Any] | None = None
) -> StructuredTool:
    """A tool from a server carrying no ``_meta`` whatsoever."""

    async def call(runtime: Any = None, **arguments: Any) -> Any:
        if seen is not None:
            seen.update(arguments)
        return "called", None

    return StructuredTool(
        name="describe_geometry",
        description="Describe a geometry.",
        args_schema={
            "type": "object",
            "properties": properties,
            "required": sorted(properties),
        },
        coroutine=call,
        response_format="content_and_artifact",
    )


async def test_an_undeclared_tool_receives_state_the_model_named() -> None:
    """The whole claim: no annotation anywhere, and the payload still arrives."""
    seen: dict[str, Any] = {}
    bound = bind_injected(foreign_tool({"geometry": {"type": "object"}}, seen))

    await run_tools(
        [bound],
        {
            "messages": [
                tool_call(
                    "describe_geometry",
                    {"geometry": handle_for("dataset-search/geometry")},
                )
            ],
            "tool_state": {
                "dataset-search/geometry": StateEntry(value=AOI, tool="search", seq=1)
            },
        },
    )
    assert seen == {"geometry": AOI}


def test_a_structured_parameter_gains_a_handle_branch() -> None:
    """The model has to be told the affordance exists, in the schema itself."""
    bound = bind_injected(foreign_tool({"geometry": {"type": "object"}}))
    schema = convert_to_openai_tool(bound)["function"]["parameters"]
    branches = schema["properties"]["geometry"]["anyOf"]
    assert {"type": "object"} in branches
    assert any(branch.get("pattern") == f"^{HANDLE_PREFIX}" for branch in branches)


def test_a_cheap_parameter_is_left_alone() -> None:
    """A string costs less to generate than to name, so it gains nothing."""
    schema = offer_handles(
        {"type": "object", "properties": {"region": {"type": "string"}}}
    )
    assert schema["properties"]["region"] == {"type": "string"}


def test_a_declared_parameter_is_not_also_offered_as_a_handle() -> None:
    """It is about to be removed from the schema; offering it would confuse."""
    schema = offer_handles(
        {"type": "object", "properties": {"aoi": {"type": "object"}}},
        skip=frozenset({"aoi"}),
    )
    assert schema["properties"]["aoi"] == {"type": "object"}


def test_an_unknown_handle_is_left_for_the_caller_to_catch() -> None:
    """Substitution is pure and reports nothing; ``unresolved`` is the check."""
    assert dereference({"geometry": handle_for("nope")}, {}) == {
        "geometry": handle_for("nope")
    }


# --- handles that substitution cannot reach --------------------------------

STORED = {"gazet/get_aoi/aoi": StateEntry(value=AOI, tool="get_aoi")}


async def submit(args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """One call through a real graph, answering with what the model is told.

    Taken from the raised ``ToolException`` or from the ``ToolMessage``,
    whichever the graph produces: what matters here is the text and that the
    tool did not run, not which of the two carried it.
    """
    seen: dict[str, Any] = {}
    bound = bind_injected(foreign_tool({"request": {"type": "object"}}, seen))
    state = {
        "messages": [tool_call("describe_geometry", args)],
        "tool_state": dict(STORED),
    }
    try:
        result = await run_tools([bound], state)
    except ToolException as error:
        return str(error), seen
    return str(result["messages"][-1].content), seen


async def test_a_nested_handle_stops_the_call_instead_of_reaching_the_server() -> None:
    """ecmwf/dss-agentic-ai-services#111, in miniature.

    A tool taking an opaque ``dict`` gets a handle branch on the whole
    parameter, so a model with a value to place puts the handle in the field
    that wants it — one level down. Nothing substituted it and the tool's own
    schema accepts anything, so the literal string reached the CDS API and came
    back as ``undefined value : "@state:gazet" for parameter AREA`` minutes
    later.
    """
    answer, seen = await submit(
        {"request": {"area": handle_for("gazet/get_aoi/aoi"), "n": 1}}
    )

    assert "request.area" in answer
    assert seen == {}, "the call must not go out"


async def test_the_message_says_which_of_the_two_failures_it_is() -> None:
    """A key nobody published is the model's to fix by running another tool; a
    nested handle is one the mechanism cannot serve, so it is pointed at
    ``inspect_state`` to read the value and write the field itself."""
    nested, _ = await submit({"request": {"area": handle_for("gazet/get_aoi/aoi")}})
    unknown, _ = await submit({"request": {"area": handle_for("gazet/nope")}})

    assert "never inside one" in nested
    assert "inspect_state" in nested
    assert "no such key" in unknown
    # What is in state either way, so the model can point at something real.
    assert "gazet/get_aoi/aoi" in nested and "gazet/get_aoi/aoi" in unknown


async def test_a_whole_argument_handle_still_resolves() -> None:
    """The guard sits after substitution, so the path that works is untouched."""
    seen: dict[str, Any] = {}
    bound = bind_injected(foreign_tool({"geometry": {"type": "object"}}, seen))

    await run_tools(
        [bound],
        {
            "messages": [
                tool_call(
                    "describe_geometry", {"geometry": handle_for("gazet/get_aoi/aoi")}
                )
            ],
            "tool_state": dict(STORED),
        },
    )

    assert seen == {"geometry": AOI}


def test_unresolved_names_the_path_not_just_the_parameter() -> None:
    """On an opaque dict the parameter is the whole request; only the path
    says which field is wrong."""
    assert unresolved({"request": {"area": handle_for("a/b")}}) == [
        ("request.area", "a/b")
    ]
    assert unresolved({"items": [1, {"geometry": handle_for("a/b")}]}) == [
        ("items[1].geometry", "a/b")
    ]
    assert unresolved({"geometry": handle_for("a/b")}) == [("geometry", "a/b")]
    assert unresolved({"request": {"area": [-3.0, 51.0, -2.0, 52.0]}}) == []


def test_a_nested_handle_is_not_answered_with_write_it_yourself() -> None:
    """The advice for an unresolved handle depends on whether the tool holds a
    parameter a model must not write.

    Observed live: a model put a handle inside an opaque ``request`` dict, was
    refused, read the closing line, fetched the value with ``inspect_state``
    and wrote it in — carrying it around the very constraint the tool declared.
    """
    state: dict[str, StateEntry] = {
        "gazet/get_aoi/bbox": StateEntry(
            value=[-3.0, 51.0, -2.0, 52.0], tool="get_aoi", seq=1
        )
    }
    found = [("request.area", "gazet/get_aoi/bbox")]

    open_tool = unresolved_message("submit_request", found, state)
    assert "write the field yourself" in open_tool

    narrowed = unresolved_message("submit_request", found, state, frozenset({"area"}))
    assert "write the field yourself" not in narrowed
    assert "'area'" in narrowed
    assert "Do not read the value and write it in" in narrowed

    # Both still list what is actually stored, which is the part the model
    # needs either way.
    assert "@state:gazet/get_aoi/bbox" in narrowed


def test_an_empty_state_says_so_however_the_tool_is_declared() -> None:
    """Nothing to point at, so neither closing applies."""
    found = [("request.area", "gazet/get_aoi/bbox")]
    for declared in (frozenset(), frozenset({"area"})):
        message = unresolved_message("submit_request", found, {}, declared)
        assert "Nothing has been published to session state yet." in message


def test_a_literal_value_passes_through_untouched() -> None:
    assert dereference({"geometry": AOI}, {}) == {"geometry": AOI}
    assert is_handle(AOI) is False


def test_available_lists_what_a_model_could_point_at() -> None:
    key = "dataset-search/search/area_of_interest"
    listed = available({key: StateEntry(value=AOI, tool="search", seq=1)})
    assert listed == [f"{handle_for(key)} — 1 feature(s), 0 vertices, from search"]


# --- recognising a value by its own shape ---------------------------------


def test_describe_summarises_without_revealing() -> None:
    assert describe(AOI) == "1 feature(s), 0 vertices"
    assert describe(["a", "b"]) == "2 item(s)"
