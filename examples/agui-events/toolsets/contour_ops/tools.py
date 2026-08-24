"""A tool whose value nothing in this deployment has produced yet.

`smooth_contours` says a model may not write its `contours`, and no server
here publishes a contour set — so the tool is offered, the model calls it, and
the binding refuses with a message naming what session state actually holds.

That refusal is the whole demonstration. Nothing is hidden from the model and
no tool is taken away at connect: the deployment cannot know in advance what
will have run by the time a call is made, so it lets the call happen and
answers it with something the model can act on.
"""

from typing import Annotated

from langchain_core.tools import tool

from mcp_runtime.declarations import NotAuthored
from mcp_runtime.tool_result import ToolResult


@tool
async def smooth_contours(
    contours: Annotated[dict, NotAuthored()],
) -> ToolResult:
    """Smooth a contour set nobody in this deployment can produce."""
    return ToolResult(message=f"Smoothed {len(contours.get('features', []))} contours.")


TOOLS = [smooth_contours]
