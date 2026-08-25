# Session state, end to end

Three real MCP servers, one agent, one scripted conversation — and a 38 kB
geometry that moves from the tool that produced it to the tools that need it
without ever entering the transcript.

```bash
uv run python examples/session-state/demo.py
```

No API key and no network: the chat model is a stub replaying a fixed script.
Everything else is real — three uvicorn servers, MCP over HTTP, declarations
travelling as `_meta`.

## What is here

| | |
| --- | --- |
| `demo.py` | Starts the servers, drives the agent, prints seven sections |
| `toolsets/dataset_search/tools.py` | Publishes a 38 kB `area_of_interest` data key |
| `toolsets/raster_ops/tools.py` | Three tools that differ only in who may write their parameter |
| `foreign_server.py` | **Raw FastMCP. No `ToolResult`, no import from `mcp_runtime`.** |

`dataset_search` and `raster_ops` are separate packages on separate servers.
Neither imports the other, and neither names the other. The only thing they
share is a **state key** — a name a model reads. The foreign server shares
nothing at all and still takes part.

## What it shows

**A name is the whole contract.** `search_datasets` returns an
`area_of_interest` data key, which lands in session state as:

```
dataset-search/search_datasets/area_of_interest
```

The model reads that off a `[state updated: …]` breadcrumb and passes
`@state:<that key>` to `clip_raster` on the other server. About ten tokens, and
the 38 kB payload never enters the conversation.

The field is called `area_of_interest` rather than `geometry` deliberately.
A coverage footprint is also a geometry, and the two are identical JSON — the
name is the only thing that says which is which, and it is what the model reads
when choosing.

**Three parameters, three different answers about who may write them.**

```
clip_raster
    aoi: an @state:<key> handle and nothing else
preview_extent
    bbox: a value, or an @state:<key> handle
clip_to_bbox
    bbox: an @state:<key> handle and nothing else
```

`preview_extent` renders a rough preview, so a model sketching a box is a fine
answer and its parameter is left alone. `clip_to_bbox` clips to an *exact*
extent, where a guessed box is worse than no answer — same JSON, different call,
so it is `NotAuthored`. Only the tool's author can make that distinction.

**Nothing is required of a server.** The foreign server's
`describe_geometry(geometry: dict)` declares nothing at all. Its parameter is
structured, so it gains the handle branch anyway and the model points it at the
same stored value. Its `elevation_profile` returns a 55 kB array nobody
declared, captured on size alone:

```
terrain/elevation_profile/samples
  54.7 kB, from elevation_profile  (captured on size)
```

**A call names its value, and the record says what that was.** The transcript
holds `@state:dataset-search/search_datasets/area_of_interest` — a key, not a
payload. The receipt on the tool message's artifact says which tool published
it and what shape it was, which is what a host renders and what the model is
*not* charged for.

**Every value says what its call was given.** Section 6 walks one:

```
raster-ops/clip_raster/bounds
  from clip_raster, given {'dataset_id': 'model', 'aoi': 'dataset-search/…/area_of_interest'}
  ...of which the model wrote: dataset_id
  dataset-search/search_datasets/area_of_interest
    from search_datasets, given {'query': 'model'}
    ...of which the model wrote: query
```

Each recorded argument names either the model or *another key*, so a value's
history is a walk over facts rather than a flag anything propagated. Nothing
refuses on it — it exists so that a value the model chose is visible where it
is later reused, which is the one place the transcript cannot help, because the
call that produced it has scrolled away.

**The binding refuses what it cannot serve.** Section 7 runs both failures for
real. A model writing its own geometry into `clip_raster`:

```
clip_raster was not called. 'aoi' was given a value you wrote. It takes a
reference to a value some tool already produced. Pass @state:<key> naming one of:
  @state:dataset-search/search_datasets/area_of_interest — 1 feature(s), 2000 vertices, from search_datasets
  …
```

And `clip_to_bbox` called with nothing in state at all:

```
clip_to_bbox was not called. 'bbox' takes a value that already exists in this
session; you cannot write one. Nothing has been published to session state yet,
so run the tool that produces this first.
```

Both arrive as an error *result*, not an exception: the assistant message's
tool calls are all answered, the transcript stays well-formed, and the model
reads the message and retries.

**Nothing is taken away at connect.** `clip_to_bbox` is offered even though
nothing here publishes a bounding box, because a producer might run later in the
same turn. The client cannot know at connect what will have run by the time a
call is made, so it lets the call happen and answers it with something
actionable.

## Things to try

- **Rename a data key.** Change `area_of_interest` to `geometry` in
  `dataset_search/tools.py` and watch the key the model has to recognise get
  less informative. Nothing breaks — that is the point, and the risk.
- **Drop the `NotAuthored()` on `clip_raster`'s `aoi`.** Section 2 will show
  it accepting a literal again. The scripted model still passes a handle, but
  nothing now stops it inlining a geometry.
- **Add `NotAuthored()` to `preview_extent`'s `bbox`.** It joins `clip_to_bbox`
  in section 2, and a call with nothing in state is refused rather than
  previewing a guessed box.
- **Turn undeclared capture off** — `StateCaptureMiddleware(published,
  capture_undeclared=None)` in `demo.py`. The foreign server's 55 kB array goes
  back into the transcript, and section 5 shows the difference.

## See also

- [`docs/SESSION-STATE.md`](../../docs/SESSION-STATE.md) — the contract in full
- [`docs/CONSUMING.md`](../../docs/CONSUMING.md) — wiring this into your own agent
- [`examples/agui-events/`](../agui-events/) — the same machinery over HTTP,
  with a browser client
