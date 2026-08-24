"""A toolset that consumes values from session state.

The consuming half. It names no other toolset — `dataset_search` is a separate
package served by a separate MCP server, and neither imports the other. What
the two share is a *name*: the model reads
`dataset-search/search_datasets/area_of_interest` off a breadcrumb and passes
it as `@state:<that key>` to `clip_raster`'s `aoi`.

The three tools differ only in what they say about who may write the value,
and that difference is the whole example:

- `clip_raster` clips to a 2000-vertex catchment boundary. A model cannot
  produce one, and a model that tried would produce something plausible and
  wrong, so the parameter is `NotAuthored`: its schema accepts a handle and
  nothing else.
- `preview_extent` renders a rough preview, where a model sketching a box is a
  perfectly good answer. Its parameter is left alone — it also accepts a
  handle, but a literal is fine.
- `clip_to_bbox` clips to an *exact* extent, where a guessed box is worse than
  no answer, so it is `NotAuthored` too. Same JSON as `preview_extent`'s
  parameter; different tool, different call about who may write it.
"""

from typing import Annotated, NotRequired

from langchain_core.tools import tool

from mcp_runtime.declarations import NotAuthored
from mcp_runtime.tool_result import ToolError, ToolResult


class ClipResult(ToolResult):
    """What the clip produced, beside the sentence the model is shown.

    Both are data keys, so both are captured — and both record that the call
    which produced them was given a `dataset_id` the model wrote and an `aoi`
    it named from state. That is what makes the provenance of `bounds`
    readable three turns later, when this call has scrolled out of view.
    """

    dataset: NotRequired[str]
    bounds: NotRequired[list[float]]


@tool
async def clip_raster(
    dataset_id: str,
    aoi: Annotated[dict, NotAuthored()],
) -> ClipResult | ToolError:
    """Clip a dataset to the area of interest currently in play."""
    features = aoi.get("features", [])
    if not features:
        return ToolError(
            error="empty_aoi", detail="The area of interest has no features."
        )
    rings = [
        ring
        for feature in features
        for ring in feature.get("geometry", {}).get("coordinates", [])
    ]
    points = [point for ring in rings for point in ring]
    vertices = sum(len(ring) for ring in rings)
    return ClipResult(
        message=f"Clipped {dataset_id} to a {vertices}-vertex area of interest.",
        dataset=dataset_id,
        bounds=[
            min(x for x, _ in points),
            min(y for _, y in points),
            max(x for x, _ in points),
            max(y for _, y in points),
        ],
    )


def _extent(bbox: list[float]) -> str:
    return ", ".join(f"{edge:g}" for edge in bbox)


@tool
async def preview_extent(
    dataset_id: str,
    bbox: list[float],
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
    bbox: Annotated[list[float], NotAuthored()],
) -> ToolResult | ToolError:
    """Clip a dataset to an exact bounding box."""
    if len(bbox) not in (4, 6):
        return ToolError(
            error="bad_bbox", detail="Expected [west, south, east, north]."
        )
    return ToolResult(message=f"Clipped {dataset_id} to exactly [{_extent(bbox)}].")


TOOLS = [clip_raster, preview_extent, clip_to_bbox]
