"""A toolset that consumes an area of interest from session state.

The consuming half. `clip_raster` needs a geometry it has no way to ask a
model for, so it tags the parameter with the `Kind` it takes. It names a
*kind*, never the toolset that publishes one — `dataset_search` is a separate
package served by a separate MCP server, and neither imports the other.

`model_generatable=False` is the tool author saying a model should never
invent this value. It is the one policy the client cannot work out for itself:
a 2000-vertex catchment boundary and a four-number bounding box are both
"geometry", and only the tool knows which of them a model could plausibly
produce. Where nothing publishes the kind, the client withholds the tool
rather than letting the model guess.
"""

from typing import Annotated

from langchain_core.tools import tool

from mcp_runtime.declarations import Kind
from mcp_runtime.kinds import GEOJSON_AREA_OF_INTEREST
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


TOOLS = [clip_raster]
