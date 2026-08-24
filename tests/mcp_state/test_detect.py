"""What :func:`mcp_state.describe` says about a value's shape.

The line these produce is what a model reads to judge how big a stored value
is before pointing a tool at it, so a wrong number here is wrong on the one
surface built to inform that choice.
"""

from typing import Any

from mcp_state.detect import describe


def _collection(*geometries: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "geometry": geometry} for geometry in geometries
        ],
    }


def _vertices(value: Any) -> int:
    """The vertex count out of a description, so a test asserts the number."""
    return int(describe(value).split(",")[1].split()[0])


def test_a_polygon_counts_its_positions() -> None:
    """A closed square is five positions: the first is repeated to close it."""
    square = {
        "type": "Polygon",
        "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
    }

    assert describe(_collection(square)) == "1 feature(s), 5 vertices"


def test_a_polygon_with_a_hole_counts_both_rings() -> None:
    with_hole = {
        "type": "Polygon",
        "coordinates": [
            [[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]],
            [[1, 1], [2, 1], [2, 2], [1, 2], [1, 1]],
        ],
    }

    assert _vertices(_collection(with_hole)) == 10


def test_a_line_string_is_not_counted_twice() -> None:
    """Its coordinates are positions, not rings. Counting one level in reads
    each ``[lon, lat]`` as a ring of length two and doubles the answer."""
    line = {"type": "LineString", "coordinates": [[0, 0], [1, 1], [2, 2], [3, 3]]}

    assert _vertices(_collection(line)) == 4


def test_a_multi_polygon_counts_vertices_and_not_polygons() -> None:
    """The items one level in are whole polygons, so a fixed-depth count
    reports three for what is in fact fifteen positions — the error that
    matters most, because it makes a large value look small."""
    squares = {
        "type": "MultiPolygon",
        "coordinates": [
            [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
            [[[2, 0], [3, 0], [3, 1], [2, 1], [2, 0]]],
            [[[4, 0], [5, 0], [5, 1], [4, 1], [4, 0]]],
        ],
    }

    assert _vertices(_collection(squares)) == 15


def test_a_multi_line_string_counts_every_line() -> None:
    border = {
        "type": "MultiLineString",
        "coordinates": [[[0, 0], [1, 1], [2, 2]], [[3, 3], [4, 4], [5, 5]]],
    }

    assert _vertices(_collection(border)) == 6


def test_a_point_is_one_vertex() -> None:
    """Its coordinates are a bare pair, with no list of positions to walk."""
    point = {"type": "Point", "coordinates": [0, 0]}

    assert _vertices(_collection(point)) == 1


def test_features_are_summed_across_the_collection() -> None:
    point = {"type": "Point", "coordinates": [0, 0]}
    line = {"type": "LineString", "coordinates": [[0, 0], [1, 1], [2, 2]]}

    assert describe(_collection(point, line)) == "2 feature(s), 4 vertices"


def test_a_geometry_that_carries_nothing_countable_is_still_described() -> None:
    """A feature with no geometry, and one whose coordinates are missing:
    neither should raise on the way to a listing."""
    assert describe(_collection({"type": "Polygon"})) == "1 feature(s), 0 vertices"
    assert describe({"type": "FeatureCollection", "features": []}) == (
        "0 feature(s), 0 vertices"
    )


def test_other_values_fall_back_to_a_shape() -> None:
    assert describe([1, 2, 3, 4]) == "4 item(s)"
    assert describe({"a": 1, "b": 2}) == "object with 2 key(s)"
    assert describe("plain") == "str"
