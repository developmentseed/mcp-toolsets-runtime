"""A toolset that publishes an area of interest into session state.

The producing half of the example. ``search_datasets`` returns a `message`
for the model and a `geometry` the model never sees — tagged with a `Kind`,
which is the only thing the consuming toolset in `raster_ops` matches on.
"""

from typing import Annotated, NotRequired

from langchain_core.tools import tool

from mcp_runtime.injected import Kind
from mcp_runtime.kinds import GEOJSON_AREA_OF_INTEREST
from mcp_runtime.tool_result import ToolError, ToolResult


def _ring(points: int = 2000) -> list[list[float]]:
    """A polygon ring big enough for the token cost to be the point."""
    return [[-3.0 + i / points, 51.0 + (i % 7) / points] for i in range(points)]


AREA_OF_INTEREST = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {"name": "Severn catchment"},
            "geometry": {"type": "Polygon", "coordinates": [_ring()]},
        }
    ],
}


class SearchDatasetsResult(ToolResult):
    """A summary for the model, plus the area those datasets cover."""

    datasets: NotRequired[list[str]]
    geometry: NotRequired[Annotated[dict, Kind(GEOJSON_AREA_OF_INTEREST)]]


@tool
async def search_datasets(query: str) -> SearchDatasetsResult | ToolError:
    """Find datasets matching a query, and the area of interest they cover."""
    if not query.strip():
        return ToolError(error="bad_query", detail="Give me something to search for.")
    return SearchDatasetsResult(
        message=f"Found 3 datasets for {query!r}, covering the Severn catchment.",
        datasets=["era5-land", "chirps-daily", "modis-lst"],
        geometry=AREA_OF_INTEREST,
    )


TOOLS = [search_datasets]
