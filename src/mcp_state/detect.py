"""Recognise a value's kind from its own shape.

A server that declares nothing still returns values that say what they are:
GeoJSON carries ``"type": "FeatureCollection"``, a STAC ItemCollection adds
``stac_version``, a bounding box is four or six numbers. :func:`detect_kind`
reads that, so a value captured from an unmodified MCP server lands in
``tool_state`` labelled well enough for a declared parameter to resolve
against it.

This is the *value* side of matching, and it is the easy side — a value is a
concrete thing that can be inspected. The parameter side has no equivalent: a
parameter is a hole, and the JSON Schema for a large object is almost always
``{"type": "object"}``, which matches everything. That asymmetry is why the
general path (:mod:`mcp_state.handles`) asks the model to name the value it
wants rather than trying to infer the match.

Detection is deliberately conservative. A wrong label is worse than no label:
an unlabelled value is still captured and still readable with ``inspect_state``,
whereas a mislabelled one can be injected somewhere it does not belong. Every
detector here keys on a discriminator the format itself defines, never on a
field name or a guess.
"""

from typing import Any

from mcp_runtime.kinds import BBOX, GEOJSON_FOOTPRINT, STAC_ITEM_COLLECTION

# A bounding box is [west, south, east, north], or the 6-element form with
# elevation. Anything else of numbers is some other array.
_BBOX_LENGTHS = (4, 6)


def _is_number(value: Any) -> bool:
    """Whether ``value`` is a JSON number (``bool`` is not, despite ``int``)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_bbox(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) in _BBOX_LENGTHS
        and all(_is_number(item) for item in value)
    )


def _is_feature_collection(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and value.get("type") == "FeatureCollection"
        and isinstance(value.get("features"), list)
    )


def _is_item_collection(value: Any) -> bool:
    """A STAC ItemCollection is a FeatureCollection that admits to a version."""
    if not _is_feature_collection(value):
        return False
    if "stac_version" in value:
        return True
    features = value.get("features") or []
    return bool(features) and all(
        isinstance(feature, dict) and "stac_version" in feature for feature in features
    )


def detect_kind(value: Any) -> str | None:
    """The kind ``value`` announces itself as, or ``None``.

    GeoJSON is reported as :data:`~mcp_runtime.kinds.GEOJSON_FOOTPRINT` rather
    than an area of interest: both are FeatureCollections and the bytes cannot
    tell them apart, so the detector names the one whose meaning is "here is
    where some data is", which is what a tool return almost always is. A tool
    that means an area of interest says so with a ``Kind`` tag, and a declared
    tag always wins over detection.
    """
    if _is_item_collection(value):
        return STAC_ITEM_COLLECTION
    if _is_feature_collection(value):
        return GEOJSON_FOOTPRINT
    if _is_bbox(value):
        return BBOX
    return None


def describe(value: Any) -> str:
    """A short human- and model-readable summary of a value's shape.

    Used in the handle listing a model chooses from, where the whole point is
    that it never sees the value itself.
    """
    if _is_feature_collection(value):
        features = value.get("features") or []
        vertices = sum(
            len(ring)
            for feature in features
            if isinstance(feature, dict)
            for ring in (feature.get("geometry") or {}).get("coordinates") or []
            if isinstance(ring, list)
        )
        return f"{len(features)} feature(s), {vertices} vertices"
    if isinstance(value, list):
        return f"{len(value)} item(s)"
    if isinstance(value, dict):
        return f"object with {len(value)} key(s)"
    return type(value).__name__
