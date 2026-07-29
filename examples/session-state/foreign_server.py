"""A third-party MCP server that has never heard of this project.

Deliberately built on raw ``FastMCP`` rather than ``build_server``: no
``ToolResult``, no ``Kind``, no ``_meta``, no import from ``mcp_runtime`` at
all. This is what somebody else's MCP server looks like, and the point of the
example is that session state still works across it.

``elevation_profile`` returns a large ``samples`` array. Nothing declares it,
so it is captured on size alone, and its shape is not one the detectors
recognise — it lands in state untyped, which is enough to hand to a tool by
name but not enough to match to a parameter automatically.

``describe_geometry`` takes a bulk parameter. Nothing declares that either, so
the client offers it as an ``@state:<key>`` handle and the model passes the
geometry ``dataset-search`` published by naming it — without the payload
passing through the transcript in either direction.
"""

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

mcp = FastMCP("terrain")


class ElevationProfile(BaseModel):
    """Elevation sampled along a transect."""

    region: str
    samples: list[dict]


class GeometryDescription(BaseModel):
    """What a geometry turned out to contain."""

    features: int
    vertices: int
    bounds: list[float]


@mcp.tool()
async def elevation_profile(region: str) -> ElevationProfile:
    """Sample ground elevation across a named region."""
    return ElevationProfile(
        region=region,
        samples=[
            {"distance_m": index * 25, "elevation_m": 40 + (index % 130) * 0.7}
            for index in range(1200)
        ],
    )


@mcp.tool()
async def describe_geometry(geometry: dict) -> GeometryDescription:
    """Report the feature count, vertex count and bounds of a GeoJSON object."""
    features = geometry.get("features", [])
    points = [
        point
        for feature in features
        for ring in feature.get("geometry", {}).get("coordinates", [])
        for point in ring
    ]
    longitudes = [point[0] for point in points] or [0.0]
    latitudes = [point[1] for point in points] or [0.0]
    return GeometryDescription(
        features=len(features),
        vertices=len(points),
        bounds=[
            min(longitudes),
            min(latitudes),
            max(longitudes),
            max(latitudes),
        ],
    )
