"""A toolset that consumes values from session state.

The consuming half. `clip_raster` needs a geometry it has no way to ask a
model for, so it tags the parameter with the `Kind` it takes. It names a
*kind*, never the toolset that publishes one — `dataset_search` is a separate
package served by a separate MCP server, and neither imports the other.

`model_generatable` is the tool author saying whether a model may invent the
value. It is the one policy the client cannot work out for itself: a
2000-vertex catchment boundary and a four-number bounding box are both
"geometry", and only the tool knows which of them a model could plausibly
produce.

The other two tools are here to be *unsatisfiable*, and they differ only in
that flag. Both take a `geo.BoundingBox`, which nothing in this example
declares it publishes, so the client has to degrade — and what it degrades to
is the whole difference:

- `preview_extent` lets a model sketch a box, so its parameter stays in the
  schema and additionally accepts an `@state:<key>` handle. The tool keeps
  working exactly as it would under a client that implements none of this.
- `clip_to_bbox` clips to an exact extent, so a guessed box is worse than no
  answer. Its parameter is hidden and nothing can fill it, which makes the
  tool uncallable — and the wiring check withholds it before a model is
  offered it at all.
"""

from typing import Annotated

from langchain_core.tools import tool

from mcp_runtime.declarations import Kind
from mcp_runtime.kinds import BBOX, GEOJSON_AREA_OF_INTEREST
from mcp_runtime.tool_result import ToolError, ToolResult


@tool
async def clip_raster(
    dataset_id: str,
    aoi: Annotated[dict, Kind(GEOJSON_AREA_OF_INTEREST, model_generatable=False)],
) -> ToolResult | ToolError:
    """Clip a dataset to the area of interest currently in play."""
    features = aoi.get("features", [])
    if not features:
        return ToolError(
            error="empty_aoi", detail="The area of interest has no features."
        )
    vertices = sum(
        len(ring)
        for feature in features
        for ring in feature.get("geometry", {}).get("coordinates", [])
    )
    return ToolResult(
        message=f"Clipped {dataset_id} to a {vertices}-vertex area of interest."
    )


def _extent(bbox: list[float]) -> str:
    return ", ".join(f"{edge:g}" for edge in bbox)


@tool
async def preview_extent(
    dataset_id: str,
    bbox: Annotated[list[float], Kind(BBOX)],
) -> ToolResult | ToolError:
    """Render a low-resolution preview of a dataset over a bounding box."""
    if len(bbox) not in (4, 6):
        return ToolError(
            error="bad_bbox", detail="Expected [west, south, east, north]."
        )
    return ToolResult(message=f"Previewed {dataset_id} over [{_extent(bbox)}].")


@tool
async def clip_to_bbox(
    dataset_id: str,
    bbox: Annotated[list[float], Kind(BBOX, model_generatable=False)],
) -> ToolResult | ToolError:
    """Clip a dataset to an exact bounding box."""
    if len(bbox) not in (4, 6):
        return ToolError(
            error="bad_bbox", detail="Expected [west, south, east, north]."
        )
    return ToolResult(message=f"Clipped {dataset_id} to exactly [{_extent(bbox)}].")


TOOLS = [clip_raster, preview_extent, clip_to_bbox]
