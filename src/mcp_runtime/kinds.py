"""The shared vocabulary injected parameters resolve on.

A ``Kind`` string is the only thing a producing toolset and a consuming
toolset agree on. They may live in different repos, be served by different
MCP servers, and never reference each other — the agent matches
``Kind(GEOJSON_AREA_OF_INTEREST)`` on one tool's output against
``Injected(kind=GEOJSON_AREA_OF_INTEREST)`` on another tool's input, and that
string is the entire contract.

So it cannot be ad hoc. Two toolsets writing ``geojson.AreaOfInterest`` and
``geojson-area-of-interest`` do not fail — they silently never interoperate,
which is worse. Kinds live here, in the package both sides already depend on,
and are added by PR like any other public API.

Adding one is safe: the wire carries the string, so a producer and a consumer
interoperate as soon as both use the same text, whatever runtime version they
pin. Redefining one is not — matching is nominal, so narrowing or reusing an
existing kind changes what every current producer and consumer means by it,
silently.

**Naming.** ``<domain>.<Type>``, dotted, domain first. Be specific enough
that two values of the same kind really are interchangeable: an area of
interest and a result footprint are both GeoJSON, but a tool asking for one
must never be handed the other, so they are separate kinds.

**Size.** A kind names what a value *is*, not how big it is. When a payload
is large enough that shipping it to the consuming server is itself the cost,
publish a pre-signed URL and use a ``*.Ref`` kind — the injected value is
then a short string the consumer fetches, and the saving is bandwidth as
well as tokens.
"""

# --- geospatial -----------------------------------------------------------

#: A GeoJSON FeatureCollection delimiting the area a user is working in.
GEOJSON_AREA_OF_INTEREST = "geojson.AreaOfInterest"

#: A GeoJSON FeatureCollection describing where data *is* (coverage,
#: footprints, tiles). Deliberately distinct from an area of interest: a clip
#: tool wants the former and would silently produce nonsense given the latter.
GEOJSON_FOOTPRINT = "geojson.Footprint"

#: A bounding box as ``[west, south, east, north]`` in EPSG:4326.
BBOX = "geo.BoundingBox"

# --- catalogue ------------------------------------------------------------

#: A STAC ItemCollection, inline.
STAC_ITEM_COLLECTION = "stac.ItemCollection"

#: A URL resolving to a STAC ItemCollection, for payloads too large to inline.
STAC_ITEM_COLLECTION_REF = "stac.ItemCollection.Ref"

#: The identifiers a search settled on, in result order.
DATASET_IDS = "catalogue.DatasetIds"


#: Every kind this package defines. A client may use it to reject an
#: ``Injected`` declaration naming a kind nothing in the vocabulary defines —
#: a typo that would otherwise present as "never resolves, no error".
KINDS = frozenset(
    {
        GEOJSON_AREA_OF_INTEREST,
        GEOJSON_FOOTPRINT,
        BBOX,
        STAC_ITEM_COLLECTION,
        STAC_ITEM_COLLECTION_REF,
        DATASET_IDS,
    }
)
