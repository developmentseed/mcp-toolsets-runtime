"""A toolset that consumes an area of interest from session state.

The consuming half. `clip_raster` needs a geometry it has no way to ask a
model for, so it marks the parameter `Injected`. It names a *kind*, never the
toolset that publishes one — `dataset_search` is a separate package served by
a separate MCP server, and neither imports the other.
"""

from typing import Annotated

from langchain_core.tools import tool

from mcp_runtime.injected import Injected
from mcp_runtime.kinds import GEOJSON_AREA_OF_INTEREST
from mcp_runtime.tool_result import ToolError, ToolResult


@tool
async def clip_raster(
    dataset_id: str,
    aoi: Annotated[dict, Injected(kind=GEOJSON_AREA_OF_INTEREST)],
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
