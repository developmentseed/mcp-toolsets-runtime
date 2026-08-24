"""Describe a stored value's shape, without claiming to know what it means.

A model choosing which stored value to pass to a tool never sees the value
itself — it sees one line per key, and this writes the part of that line that
comes from the value.

Shape only, deliberately. Recognising a *format* is easy and tempting: GeoJSON
carries ``"type": "FeatureCollection"``, a bounding box is four or six numbers.
Recognising what a value is *for* is not possible from the bytes at all. An
area of interest and a coverage footprint are the same JSON, and a detector
that labelled one would be confidently wrong about the other roughly half the
time — on the very listing a model reads to choose between them. What a value
means is carried by the name it is stored under, which its tool chose.
"""

from typing import Any


def _is_feature_collection(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and value.get("type") == "FeatureCollection"
        and isinstance(value.get("features"), list)
    )


def _positions(coordinates: Any) -> int:
    """How many ``[lon, lat]`` positions are under a GeoJSON coordinate array.

    One walk covers Point through MultiPolygon, because the thing that
    distinguishes them is nesting depth and the leaves are the same pair either
    way. Counting at a fixed depth gets Polygon right and everything else
    wrong: the items one level into a MultiPolygon are its polygons, and one
    level into a LineString they are its positions, so the first is counted as
    a handful and the second as double.
    """
    if not isinstance(coordinates, list) or not coordinates:
        return 0
    first = coordinates[0]
    if isinstance(first, (int, float)) and not isinstance(first, bool):
        return 1
    return sum(_positions(item) for item in coordinates)


def describe(value: Any) -> str:
    """A short human- and model-readable summary of a value's shape.

    Used in the handle listing a model chooses from, where the whole point is
    that it never sees the value itself.
    """
    if _is_feature_collection(value):
        features = value.get("features") or []
        vertices = sum(
            _positions((feature.get("geometry") or {}).get("coordinates"))
            for feature in features
            if isinstance(feature, dict)
        )
        return f"{len(features)} feature(s), {vertices} vertices"
    if isinstance(value, list):
        return f"{len(value)} item(s)"
    if isinstance(value, dict):
        return f"object with {len(value)} key(s)"
    return type(value).__name__
