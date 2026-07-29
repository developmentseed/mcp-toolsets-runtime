# Session state, end to end

A runnable demonstration of [`docs/SESSION-STATE.md`](../../docs/SESSION-STATE.md):
a value produced by a tool on one MCP server reaching a tool on another,
without passing through the model — including a server that has never heard of
this project.

```bash
uv run python examples/session-state/demo.py
```

No API key, no network, nothing to start first. It serves three MCP servers on
ephemeral local ports and drives a scripted chat model, so the only thing faked
is the model.

## What's here

| | |
| --- | --- |
| `toolsets/dataset_search/tools.py` | Publishes a 38 kB area of interest, tagged `Kind(GEOJSON_AREA_OF_INTEREST)` |
| `toolsets/raster_ops/tools.py` | Takes one, tagged with the same kind and `model_generatable=False` |
| `foreign_server.py` | **Raw FastMCP. No `ToolResult`, no `Kind`, no import from `mcp_runtime`.** |
| `demo.py` | Serves all three, connects an agent, reports what happened |

The two toolsets follow the plugin contract and neither imports the other; the
only thing they share is a kind string. The foreign server shares nothing at
all — it is there to prove the mechanism does not depend on cooperation.

## The two paths

**Declared, when a server tags a parameter.** `clip_raster` takes an
`aoi: Annotated[dict, Kind(...)]`, so the client matches the kind, fills the
value, and removes the parameter from the model's schema entirely:

```
clip_raster
  server advertises: ['aoi', 'dataset_id']
  model is offered:  ['dataset_id']
```

The model emitted `dataset_id` and nothing else. It could not have got the
geometry wrong, because it never saw that there was one.

**Undeclared, when nothing is tagged.** The foreign server's
`describe_geometry(geometry: dict)` declares nothing, so the client cannot know
that parameter wants an area of interest — a structured parameter's schema is
`{"type": "object"}`, which matches every object ever written. Instead it adds
a second accepted form:

```
describe_geometry
  server advertises: ['geometry']
  model is offered:  ['geometry']
    geometry: object, or an @state:<key> handle
```

The model passes `@state:dataset-search/geometry` — about ten tokens, read off
the `[state updated: …]` breadcrumb — and the client swaps in the payload
before the call. The foreign server receives an ordinary GeoJSON object and
has no idea any of this happened.

That is the trade: the declared path costs zero tokens and the model *cannot*
get it wrong; the undeclared path costs ten tokens, works against anything, and
the model chooses — which is also the only party with the conversation in front
of it when there is more than one candidate.

## What else the run shows

**Capture does not need a declaration either.** `elevation_profile` returns a
55 kB `samples` array that nothing declared. It is captured on size alone:

```
elevation_profile/samples  —  54.7 kB, from elevation_profile
  kind=unrecognised  (captured on size)
```

`unrecognised` is honest — a list of `{distance_m, elevation_m}` is not a shape
the detectors know. It can still be handed to a tool by name; it just cannot be
matched to a parameter automatically. That is exactly what a `Kind` tag buys.

**The payload never entered the transcript.**

```
area of interest:      38.5 kB
whole transcript:      814 bytes
a vertex (-2.5) in it: no
```

Both tools ran against all 2000 vertices.

## Things to try

- **Break the wire.** Change the kind in `raster_ops/tools.py` to something
  nothing publishes. Section 2 reports it and `clip_raster` is withheld from the
  agent, rather than failing when a user finally triggers it.
- **Let the model try instead.** Drop `model_generatable=False` from that same
  broken declaration: the tool comes back, with `aoi` visible to the model
  again — the behaviour of a client implementing none of this.
- **Remove the tag entirely.** Delete the `Kind` from `clip_raster`'s `aoi` and
  it falls back to the general path: the parameter reappears in the schema, now
  with a handle branch, and the model has to point it at the geometry the same
  way `describe_geometry` does.
- **Raise the capture threshold.** `StateCaptureMiddleware(published,
  capture_undeclared=None)` turns undeclared capture off; the foreign server's
  55 kB array then stays in the transcript, which is the cost of not capturing.

## Not installed

This directory is outside `src/`, so it is not in the wheel and reaches nobody
who installs the package. It is here to be read and run from a checkout.
