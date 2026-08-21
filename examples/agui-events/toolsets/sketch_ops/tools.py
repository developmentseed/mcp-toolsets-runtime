"""The other side of `NotAuthored`: a structured parameter left open.

`clip_view.clip_raster` tags its `aoi`, so its schema accepts a handle and
nothing else and a model cannot write a polygon into it. `sketch_area` does
not tag `boundary`, which is the ordinary case and the right one here — the
whole point of a sketch is that the model draws it.

What that costs is visible rather than prevented. The boundary the model
writes is captured into session state like any other data key, and the entry
records that the call which produced it was given a `boundary` the model
wrote. So the panel shows a value resting on a few hundred characters of
invented geometry, and shows the geometry.

That is the case worth seeing. A model inlining a large literal into an
untagged parameter is what session state exists to avoid, and it is not an
error — it is a judgement the tool's author made, and this is what it looks
like when it goes the other way.
"""

from typing import Any, NotRequired

from langchain_core.tools import tool

from mcp_runtime.tool_result import ToolError, ToolResult

#: Rough metres per degree at mid-latitudes. Good enough for a sketch, which
#: is the only thing this claims to be.
_METRES_PER_DEGREE = 111_320


class SketchResult(ToolResult):
    """The boundary the model drew, and how big it turned out to be."""

    boundary: NotRequired[dict[str, Any]]
    area_km2: NotRequired[float]


def _ring_area_km2(ring: list[list[float]]) -> float:
    """The shoelace area of one ring, in square kilometres."""
    total = 0.0
    for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1], strict=False):
        total += x1 * y2 - x2 * y1
    degrees = abs(total) / 2
    return degrees * (_METRES_PER_DEGREE**2) / 1_000_000


@tool
async def sketch_area(name: str, boundary: dict[str, Any]) -> SketchResult | ToolError:
    """Record a rough area you have drawn yourself, as a GeoJSON polygon.

    Write the polygon out in full: this takes a sketch, not a reference to
    something another tool produced.
    """
    features = boundary.get("features") or []
    rings = [
        ring
        for feature in features
        if isinstance(feature, dict)
        for ring in (feature.get("geometry") or {}).get("coordinates") or []
        if isinstance(ring, list) and len(ring) >= 3
    ]
    if not rings:
        return ToolError(
            error="empty_boundary",
            detail="Expected a FeatureCollection with at least one polygon ring.",
        )
    area = sum(_ring_area_km2(ring) for ring in rings)
    return SketchResult(
        message=f"Recorded {name!r} — {len(rings)} ring(s), about {area:,.0f} km².",
        boundary=boundary,
        area_km2=round(area, 1),
    )


TOOLS = [sketch_area]
