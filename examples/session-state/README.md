# Session state, end to end

A runnable demonstration of [`docs/SESSION-STATE.md`](../../docs/SESSION-STATE.md):
a value produced by a tool on one MCP server reaching a tool on another,
without passing through the model.

```bash
uv run python examples/session-state/demo.py
```

No API key, no network, nothing to start first. It serves both toolsets on
ephemeral local ports and drives a scripted chat model, so the only thing
faked is the model.

## What's here

| | |
| --- | --- |
| `toolsets/dataset_search/tools.py` | Publishes a 38 kB area of interest, tagged `Kind(GEOJSON_AREA_OF_INTEREST)` |
| `toolsets/raster_ops/tools.py` | Consumes one, via `Injected(kind=GEOJSON_AREA_OF_INTEREST)` |
| `demo.py` | Serves both, connects an agent, reports what happened |

The two toolsets are ordinary packages following the plugin contract — `TOOLS`
plus `ToolResult` returns. Neither imports the other, and neither knows the
other's name; the only thing they share is the kind string.

## What it shows

The run prints six things. The ones worth reading:

**The declarations arrived over the wire.** `search_datasets` advertises what
it publishes and `clip_raster` what it consumes, both as MCP `_meta` from a
real HTTP server — not constructed in-process by the demo.

**The model is offered a smaller tool than the server serves.**

```
server advertises: ['aoi', 'dataset_id']
model is offered:  ['dataset_id']   <- aoi is gone
```

**The payload never entered the transcript.**

```
area of interest:      38.5 kB
whole transcript:      350 bytes
coordinates in it:     no
```

`clip_raster` still ran against all 2000 vertices. The model emitted
`dataset_id` and nothing else.

It also shows the wiring check passing before the agent is built, the
`[state updated: …]` breadcrumb the model gets instead of the payload, and an
*untagged* data key (`datasets`) — captured into state and readable with
`inspect_state`, but unable to satisfy an injected parameter, because only
`Kind` makes a value matchable.

## Things to try

- **Break the wire.** Change the kind in `raster_ops/tools.py` to something
  nothing publishes. Section 2 reports it and `clip_raster` is withheld from
  the agent, rather than failing when a user finally triggers it.
- **Make it recoverable.** Add `model_fallback=True` to that same broken
  declaration: the tool comes back, with `aoi` visible to the model again.
- **Break the contract.** Add `required=False` without giving the parameter a
  Python default — `build_server` refuses to start and says why.

## Not installed

This directory is outside `src/`, so it is not in the wheel and reaches nobody
who installs the package. It is here to be read and run from a checkout.
