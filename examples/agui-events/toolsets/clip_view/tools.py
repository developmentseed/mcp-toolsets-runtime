"""`clip_raster` with a real view over it.

The session-state example's version returns a one-line message, which is all a
transcript needs. A view needs the shape it clipped to, so this one returns the
geometry as well — and that is the whole demonstration: the polygon reaches the
map without the model ever seeing it. It arrives as a declared `aoi` the model
was never offered, it is captured back out of the reply before the transcript
is written, and `restore_structured` puts it back for the view alone.

`bounds` and `vertices` ride along because the view would otherwise compute
them from the geometry every render, and a tool that knows its own answer
should say it.
"""

from typing import Annotated, Any, NotRequired

from langchain_core.tools import tool
from mcp_runtime.declarations import Kind
from mcp_runtime.kinds import GEOJSON_AREA_OF_INTEREST
from mcp_runtime.tool_result import ToolError, ToolResult


class ClipResult(ToolResult):
    """What `clip_raster` returns, and what its view is written against."""

    dataset: NotRequired[str]
    vertices: NotRequired[int]
    bounds: NotRequired[list[float]]
    #: The clipped area itself. Large enough to be captured into session state
    #: rather than left in the transcript, which is the point.
    geometry: NotRequired[dict[str, Any]]


def _rings(geometry: dict[str, Any]) -> list[list[list[float]]]:
    """Every ring in a Polygon or MultiPolygon, as flat coordinate lists."""
    coordinates = geometry.get("coordinates") or []
    if geometry.get("type") == "MultiPolygon":
        return [ring for polygon in coordinates for ring in polygon]
    return list(coordinates)


@tool
async def clip_raster(
    dataset_id: str,
    aoi: Annotated[dict, Kind(GEOJSON_AREA_OF_INTEREST, model_generatable=False)],
) -> ClipResult | ToolError:
    """Clip a dataset to the area of interest currently in play."""
    features = aoi.get("features", [])
    if not features:
        return ToolError(
            error="empty_aoi", detail="The area of interest has no features."
        )
    points = [
        point
        for feature in features
        for ring in _rings(feature.get("geometry") or {})
        for point in ring
    ]
    if not points:
        return ToolError(error="empty_aoi", detail="No coordinates to clip to.")
    longitudes = [point[0] for point in points]
    latitudes = [point[1] for point in points]
    return ClipResult(
        message=f"Clipped {dataset_id} to a {len(points)}-vertex area of interest.",
        dataset=dataset_id,
        vertices=len(points),
        bounds=[min(longitudes), min(latitudes), max(longitudes), max(latitudes)],
        geometry=aoi,
    )


TOOLS = [clip_raster]
VIEWS = {"clip_raster": "clip"}
