"""A toolset that publishes an area of interest into session state.

The producing half of the example. ``search_datasets`` returns a `message` for
the model and an `area_of_interest` the model never sees: it is a data key of
the `ToolResult`, so it is captured into session state under
`dataset-search/search_datasets/area_of_interest`.

**The field name is the interface.** It is what the model reads when it
decides which stored value to hand to `raster_ops.clip_raster`, and the two
toolsets share nothing else — separate packages, separate servers, neither
importing the other. `area_of_interest` rather than `geometry` for exactly
that reason: a footprint is also a geometry, and the model would have no way
to tell them apart.
"""

from typing import NotRequired

from langchain_core.tools import tool

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
    area_of_interest: NotRequired[dict]


@tool
async def search_datasets(query: str) -> SearchDatasetsResult | ToolError:
    """Find datasets matching a query, and the area of interest they cover."""
    if not query.strip():
        return ToolError(error="bad_query", detail="Give me something to search for.")
    return SearchDatasetsResult(
        message=f"Found 3 datasets for {query!r}, covering the Severn catchment.",
        datasets=["era5-land", "chirps-daily", "modis-lst"],
        area_of_interest=AREA_OF_INTEREST,
    )


TOOLS = [search_datasets]
