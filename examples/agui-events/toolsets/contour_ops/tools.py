"""A tool whose declared parameter nothing connected publishes.

`geojson.ContourSet` is a kind no server here produces, and the tool says a
model must not invent one — so `partition_usable` withholds it, which is what
`tools.withheld` announces to a client.
"""

from typing import Annotated

from langchain_core.tools import tool

from mcp_runtime.declarations import Kind
from mcp_runtime.tool_result import ToolResult


@tool
async def smooth_contours(
    contours: Annotated[dict, Kind("geojson.ContourSet", model_generatable=False)],
) -> ToolResult:
    """Smooth a contour set nobody in this deployment can produce."""
    return ToolResult(message=f"Smoothed {len(contours.get('features', []))} contours.")


TOOLS = [smooth_contours]
