"""The path that needs nothing from the server.

A tool nobody annotated, on a server that has never heard of this project,
still gets session state — the model points a structured parameter at a value
by name, and the client swaps in the payload before the call.
"""

from typing import Any

from langchain_core.tools import StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_tool

from mcp_runtime.kinds import BBOX, GEOJSON_FOOTPRINT, STAC_ITEM_COLLECTION
from mcp_state.detect import describe, detect_kind
from mcp_state.handles import (
    HANDLE_PREFIX,
    available,
    dereference,
    handle_for,
    is_handle,
    offer_handles,
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


def test_an_unknown_handle_is_left_for_the_tool_to_reject() -> None:
    """Better a legible schema error than a silent None the tool trips over."""
    assert dereference({"geometry": handle_for("nope")}, {}) == {
        "geometry": handle_for("nope")
    }


def test_a_literal_value_passes_through_untouched() -> None:
    assert dereference({"geometry": AOI}, {}) == {"geometry": AOI}
    assert is_handle(AOI) is False


def test_available_lists_what_a_model_could_point_at() -> None:
    listed = available(
        {
            "a/geometry": StateEntry(
                value=AOI, kind=GEOJSON_FOOTPRINT, tool="search", seq=1
            )
        }
    )
    assert listed == [
        f"{handle_for('a/geometry')} — {GEOJSON_FOOTPRINT}, "
        "1 feature(s), 0 vertices, from search"
    ]


# --- recognising a value by its own shape ---------------------------------


def test_geojson_announces_itself() -> None:
    assert detect_kind({"type": "FeatureCollection", "features": []}) == (
        GEOJSON_FOOTPRINT
    )


def test_stac_is_distinguished_from_plain_geojson() -> None:
    assert (
        detect_kind(
            {"type": "FeatureCollection", "stac_version": "1.0.0", "features": []}
        )
        == STAC_ITEM_COLLECTION
    )


def test_a_bounding_box_is_four_or_six_numbers() -> None:
    assert detect_kind([-3.0, 51.0, -2.0, 52.0]) == BBOX
    assert detect_kind([-3.0, 51.0, -2.0]) is None
    # `bool` is an `int` in Python; a list of flags is not a bounding box.
    assert detect_kind([True, False, True, False]) is None


def test_an_unrecognised_shape_stays_unlabelled() -> None:
    """No label is safe; a wrong one gets injected somewhere it does not belong."""
    assert detect_kind([{"distance_m": 0, "elevation_m": 40}]) is None
    assert detect_kind({"datasets": ["a", "b"]}) is None


def test_describe_summarises_without_revealing() -> None:
    assert describe(AOI) == "1 feature(s), 0 vertices"
    assert describe(["a", "b"]) == "2 item(s)"
