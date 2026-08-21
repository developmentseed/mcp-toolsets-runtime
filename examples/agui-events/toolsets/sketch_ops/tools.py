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

**The published boundary is normalised, not echoed.** Any real geometry tool
closes the ring and rounds the coordinates, and this one does too — which
matters here beyond realism. A tool that returned its argument unchanged would
publish a value byte-identical to the one the model wrote, and a reader would
see the same polygon twice in one card with nothing to say why. It is also the
case that defeats inferring provenance from a *return*: compare the output to
the arguments and a normalised boundary looks derived, while the same tool
without the rounding looks like a passthrough. The record is of what the call
was given, precisely so nothing has to make that call.
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


def _normalised(ring: list[list[float]]) -> list[list[float]]:
    """One ring, closed and rounded to a sketch's worth of precision.

    Four decimal places is about 11 m, which is finer than anything drawn by
    hand deserves. Closing the ring is not cosmetic: a polygon whose first and
    last positions differ is invalid GeoJSON, and a model writing one out by
    hand forgets regularly.
    """
    rounded = [[round(x, 4), round(y, 4)] for x, y in ring]
    if rounded[0] != rounded[-1]:
        rounded.append(list(rounded[0]))
    return rounded


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
    tidied = [_normalised(ring) for ring in rings]
    area = sum(_ring_area_km2(ring) for ring in tidied)
    return SketchResult(
        message=(
            f"Recorded {name!r} — {len(tidied)} ring(s), "
            f"{sum(len(ring) for ring in tidied)} vertices, about {area:,.0f} km²."
        ),
        boundary={
            "type": "FeatureCollection",
            "name": name,
            "features": [
                {
                    "type": "Feature",
                    "properties": {"name": name},
                    "geometry": {"type": "Polygon", "coordinates": [ring]},
                }
                for ring in tidied
            ],
        },
        area_km2=round(area, 1),
    )


TOOLS = [sketch_area]
