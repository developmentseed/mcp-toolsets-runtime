# Remove the Kind marker

**Status:** Phase 1 implemented (`feat/not-authored-parameters`); Phases 2-3
proposed
**Decided:** provenance is recorded, never enforced, and taught to the model
as well as rendered in the UI; one level, not a chain walk; `produces` leaves
`/health` but the `_meta` manifest stays; an unfillable parameter refuses at
call time; `detect_kind` goes too; state keys unify on
`<toolset>/<tool>/<field>`
**Interacts with:** `injection-receipts.md` — its Phases 1-3 have landed
(`mcp_state/receipts.py`); its Phase 4 ledger changes shape under this proposal
and its `via` open question is answered by it.

## The claim

`Kind` is a nominal type shared between a producing toolset and a consuming
one. The string is the entire contract, and the contract is only meaningful
between toolsets whose authors already talk to each other. Outside that, it is
mute — and the way it goes mute is the problem.

A declaration whose kind nothing connected publishes is **dropped**.
`_bindable` (`injection.py:276`) removes it, the parameter stays in the model's
schema, and the model fills it. That is precisely where a client with no
`Kind` support starts. So a mismatched or unrecognised kind does not degrade
to something better than name-matching; it degrades *to* name-matching, having
charged a registry, a wire protocol and a connect-time check on the way.

Meanwhile a model matching the state key `dataset-search/geometry` to a
parameter named `aoi` is doing the same semantic work the kind string encodes,
with no cross-repo agreement required.

### The registry is already not a gate

- `KINDS` (`kinds.py:60`) is exported and imported by nothing. `wiring.py`
  says so outright: "the check is on the wiring, not on membership of
  `mcp_runtime.kinds`."
- `examples/agui-events/toolsets/contour_ops/tools.py:18` already ships
  `Kind("geojson.ContourSet")` — a string not in the registry — and it works.
- Of the six registered kinds, `stac.ItemCollection.Ref` and
  `catalogue.DatasetIds` have zero uses anywhere, including tests.
- Across every real toolset there are **five** `Kind` tags: four in dss
  `gazet`, one in `cds`.

## What is actually being removed

Three separable things travel under the name "kind". Only the second is in
scope.

| | what | in scope |
|---|---|---|
| `mcp_runtime/kinds.py` | six geospatial constants | **deleted** |
| the `Kind` marker + `_meta` produces/consumes | the nominal-type wire protocol | **removed** |
| `mcp_state/detect.py` | `detect_kind` labelling / `describe` shapes | `detect_kind` **removed**, `describe` kept |

`detect_kind` looked like a keeper — it never required a declaration, and it
labels the handle listing a model chooses from. It goes too, for a reason that
only shows up once declarations are gone. See "Why `detect_kind` goes" below.
With it, `kinds.py` deletes entirely: `describe()` is the half of `detect.py`
worth keeping, and it names no kinds.

## What replaces it

### 1. `NotAuthored` — a fact about the parameter, not about the plumbing

`model_generatable=False` was only ever a statement about one parameter in
isolation: *a model cannot author this value*. It needed the kind for
**resolution**, never for the **refusal**.

The first draft of this proposal called the replacement `FromState`, which was
the wrong name for the right idea. A tool author should not have to know that
session state exists in order to write a tool. "This comes from state" is a
claim about a client mechanism they have never heard of; "a model must not
write this" is a claim about their own parameter, true whether or not anything
downstream implements state at all.

```python
@tool
async def clip_raster(
    dataset_id: str,
    aoi: Annotated[FeatureCollection, NotAuthored()],
) -> ClipResult | ToolError: ...
```

Read it without knowing anything about this runtime: *the caller must supply an
existing feature collection; do not invent one*. That is meaningful to a plain
MCP client, and safely ignorable by one — the tool behaves exactly as it does
today. It happens to be the thing our client needs in order to make the
parameter handle-only, but the author is not writing it for us.

This gives three tiers of enforcement for free, by how much the client is
willing to do:

| client | effect |
|---|---|
| ignores it | the parameter behaves normally; the model fills it |
| reads the description | advisory — "must already exist, do not author one" |
| implements this proposal | the schema accepts only `@state:<key>` |

**Nothing is required on the output side.** The produces manifest is derived
from the `ToolResult` subclass an author already writes for other reasons
(section 6). So the entire state-awareness demanded of a toolset is one
optional annotation that is not, itself, about state.

**Why the type alone will not do it.** Rich parameter types make a structural
match possible — `resolve` already validates a candidate against the parameter
schema, so with `kind` gone that validation could stand on its own. It does not
replace the annotation, for two reasons. An area of interest and a footprint
both validate against `FeatureCollection`, so structure yields a candidate set
and never a choice. And binding runs at connect (`bind_all_injected`), when
state is empty, so a schema cannot depend on what happens to be stored. Making
schemas state-dependent per turn is possible through middleware and is a much
larger change than this proposal.

### 2. Handle-only schemas — enforcement moves into the schema

`_with_handle_branch` (`handles.py:84`) currently widens a structured
parameter to `anyOf: [original, handle_branch]` — literal *or* handle.
`NotAuthored` means: emit the handle branch alone.

```json
{"type": "string", "pattern": "^@state:", "description": "..."}
```

This is **stronger** than today's behaviour. `model_generatable=False`
currently hides the parameter and has the client fill it; a model determined
to inline a 2000-vertex polygon is stopped by the parameter being absent. Under
handle-only the schema itself refuses — the only string it accepts starts with
`@state:`. That closes the gap `handles.py` presently concedes as "the price of
requiring nothing of the server".

The miss case is already written. Nothing in state means no valid key, the
model guesses, `unresolved()` catches it before the call leaves, and
`unresolved_message` refuses *with the available keys listed*. Self-correcting,
and it reaches the model as a result rather than an exception.

### 3. Naming becomes the contract — on the producer side

The semantic weighting the kind string carried has to live in the names, and
the side that matters is the one currently named worst.

State keys are `qualified()` as `<toolset>/<ToolResult field>`. The model is
not matching a parameter to a concept; it is matching a parameter to a **state
key**. Today's listing reads:

```
@state:dataset-search/geometry — geojson.Footprint, 12 feature(s), from search_datasets
```

Strip the kind and the model bridges `dataset-search/geometry` to `aoi` on the
word "geometry" alone — the single weakest name in the repo for this, and it
is in our own example toolset (`dataset_search/tools.py:43`). It is also the
exact case `detect.py` says the bytes cannot settle: an area of interest and a
footprint are identical JSON.

The rule to write down is therefore not "name parameters explicitly" but:

> **A `ToolResult` data key is a public name. The next toolset's model reads
> it to decide what the value is.**

Renaming the example toolsets' data keys is part of this change, not a
follow-up.

### 4. An unfillable parameter refuses at call time

Losing `check_wiring` means a required `NotAuthored` parameter with nothing in
state is no longer caught at connect. It is caught on the first call instead,
and the refusal is what does the work `check_wiring` used to.

The sequence: the schema accepts only `^@state:`, so the model emits some key —
a guess, if it has seen no `[state updated: ...]` breadcrumb. `unresolved()`
catches it before the call leaves the client, and `unresolved_message` returns a
refusal as a *tool result*, not an exception. The model reads it and runs
something that publishes.

That message has to push the model toward a producer, and here is the one place
the loss of kinds bites. `_missing` (`injection.py:242`) can currently name the
producing tools, because `published` maps each kind to the tools that publish
it. With nothing declaring what it publishes, the client cannot name them.

Two ways to close that, in order of preference:

**The model works it out.** The refusal lists what *is* in state and says the
parameter needs a published value. The model holds the full tool listing with
every description, which is the same information a human would use. For a
toolset of any reasonable size this is enough.

**The tool author writes a hint.** `NotAuthored` takes optional free text —
`NotAuthored("run get_aoi or generate_aoi first")` — which lands in the
parameter description and in the refusal. This is a soft coupling to another
toolset, and worth being honest that it is one. It is still strictly better
than a kind string: a stale hint is a confusing sentence a reader can see and
correct, where a stale kind is a silent non-match that degrades to the model
filling the parameter with an invention.

Start with the first. Add the second only if real sessions show the model
casting about.

### 5. Why `detect_kind` goes, and what covers the gap

`detect_kind` is only reached when a field has no declaration
(`middleware.py:245`). Today that is the fallback path. After this change it is
the *only* path — so every captured value would be labelled by inference.

Run it against the area of interest in our own example toolset:

```
declared as : geojson.AreaOfInterest   (the Kind tag on SearchDatasetsResult.geometry)
detect_kind : geojson.Footprint
describe    : 1 feature(s), 2000 vertices
```

The detector is not being sloppy; `detect.py` documents this exact choice. An
area of interest and a footprint are identical JSON, so it names the one a tool
return usually is. That was harmless while a declaration could override it. With
declarations gone, nothing overrides it, and the single most important value in
the system gets labelled as its own opposite — on the listing the model reads to
choose values. A confidently wrong label is worse than no label, which is
`detect.py`'s own stated principle applied to itself.

`describe()` is the half worth keeping. It reports the value, not a guess about
its meaning:

```
@state:dataset-search/search_datasets/geometry — 1 feature(s), 2000 vertices
```

**The parameter side is covered by typing the parameter.** `detect.py` argues
the parameter is a hole that cannot be inspected, because "the JSON Schema for
a large object is almost always `{"type": "object"}`". That is true of our own
toolsets, and it is self-inflicted — they annotate `aoi: dict`:

```json
"aoi": {"type": "object", "additionalProperties": true}
```

Annotate it as a real model and the schema describes it fully:

```json
"aoi": {"$ref": "#/$defs/FeatureCollection"}
```

with `FeatureCollection`, `Feature` and `Geometry` in `$defs`. The model
already receives `$defs` in the same document, so **there is nothing to
resolve** — inlining the `$ref` would duplicate what is already there, and
cannot terminate in general anyway, since GeoJSON is recursive through
`GeometryCollection`. The win is entirely in typing the parameter, not in
flattening the schema.

Measured on a minimal GeoJSON model, the whole input schema goes from ~94
tokens to ~465. That cost is real for an ordinary parameter, and **zero for a
`NotAuthored` one**: the handle-only branch drops the structural arm entirely, so
the model never sees the type at all while the server still validates against
it. Type those as strictly as you like.

One implementation note: `_prune` deliberately leaves `$defs` "byte-for-byte"
(`injection.py:181`), so a pruned or handle-only parameter currently leaves its
definitions orphaned in the schema and still sent to the model. That wants
fixing as part of Phase 1, or richly-typed `NotAuthored` parameters are not free
after all.

**What no amount of typing recovers** is the distinction kinds existed for. An
area of interest and a footprint are both `FeatureCollection`; structure cannot
separate things that are structurally identical. That separation now lives
entirely in the names — the state key and the parameter — which is what
section 3 is about.

### 6. What `produces` must keep doing

`Kind` on a `ToolResult` field looks like it only names a kind. It does not:
the `_meta` produces entry is also the **capture manifest**, and the two must
be separated carefully or capture breaks.

`output_kinds` returns *every* `ToolResult` data key, mapping untagged ones to
`None`, and `with_state_meta` stamps them all into `PRODUCES_META_KEY`.
`publications()` reads that back, and `_updates` (`middleware.py:229`) captures
a field **because it appears there** — the `kind` on the entry is only carried
along:

```python
if declaration := declarations.get(field):        # ← the manifest
    updates[declaration["stateKey"]] = StateEntry(value=value, kind=..., tool=...)
elif threshold is not None and _size(value) >= threshold:
    ...
```

Delete the manifest and capture falls through to the size threshold alone,
`DEFAULT_CAPTURE_BYTES = 2048`. A bounding box is about thirty bytes. It would
never be captured, so nothing could ever publish one, so a `NotAuthored` bbox
parameter would be permanently unfillable — the tool refuses on every call
forever. gazet's `bbox` is exactly this shape.

So:

- **`PRODUCES_META_KEY` stays.** Entries become `{"stateKey": ..., "field": ...}`
  with the `kind` removed. `output_kinds` becomes `output_fields`, returning the
  set of data keys.
- **`/health`'s `produces` goes.** That is the *other* one — a flat list of kind
  strings for the index, computed by `state_declarations` from the tags. Nothing
  reads it and there are no tags left to compute it from.

Being a `ToolResult` data key rather than `message` is what marks a field as
data. That was always the real declaration; the kind was decoration on top.

### 7. `[state used: ...]` disappears

`breadcrumb()` (`receipts.py:127`) emits the note only for `BY_DECLARATION`
receipts, on the stated grounds that a handle is already in the tool call
arguments the model wrote. With the FILL path gone every receipt is
`BY_HANDLE`, so the note is never emitted and the function deletes.

That follows the existing logic rather than contradicting it, but three things
have to move with it:

- `Receipt.via` collapses to one value and can go. This answers the "is `via`
  worth carrying?" question left open in `injection-receipts.md`.
- `prompt.py` currently teaches the model about hidden parameters and
  `[state used: ...]` notes in two places, including the provenance paragraph
  that tells it how to describe a result. Both need rewriting around handles
  and `inputs`. The model **is** taught `inputs` — see below.
- `host.py:147` leads its receipt rendering with `receipt.get("kind") or
  "untyped"`. It becomes `describe`-led, matching the listing.

### 8. State keys unify on `<toolset>/<tool>/<field>`

Two schemes currently share one namespace. A declared capture is keyed
`qualified(toolset, field)` — `dataset-search/geometry`. An undeclared one is
keyed `qualified(tool_name, field)` (`middleware.py:244`) —
`search_datasets/geometry`. Harmless while a key was a lookup token that kind
resolution never read; not harmless once the key is the semantic carrier the
model matches on.

Unify on three parts:

```
dataset-search/search_datasets/geometry
```

The tool name is worth its tokens now. It is the difference between "some
geometry this toolset produced" and "the geometry `search_datasets` returned",
and the latter is what a model is actually reasoning about when it decides
whether a stored value is the area the user meant.

**One gap to close first.** `qualified(toolset, ...)` works for declared
captures because `with_state_meta` bakes the toolset name into `stateKey` at
build time. For an undeclared capture from a third-party server the client has
only `request.tool_call["name"]` — `langchain_mcp_adapters` takes a
`server_name` but stamps it nowhere on the resulting tool, putting it only in
the call request and, optionally, in a tool-name prefix. We load the tools
ourselves in `main.py`, so the fix is to stamp the connection name into the
tool's metadata at load and read it back at capture. Without that, undeclared
captures fall back to `<tool>/<field>` and the inconsistency survives in a
quieter form.

After this change undeclared capture only applies to servers that are not ours
— everything we build declares through the produces manifest — so the fallback
is narrow either way.

## Provenance: what a call was given

### The tool owns its outputs

A tool that returns a value is responsible for what that value is. The runtime
does not inspect a return to decide whether it was genuinely derived or merely
echoed back.

An output-side heuristic is available — compare each captured field to the
call's arguments — and it is tempting because it catches the passthrough case
exactly. It is still the wrong seam. It catches passthrough and misses
transformation, so it produces a label that is *sometimes* right with no way
for a reader to know which time this is. Worse, it puts the runtime in the
position of contradicting a tool about its own output. Where the line falls
between "derived from" and "echoed" is the tool author's judgement, on a value
only they understand.

So: no inference on returns.

### The runtime owns the inputs

What the runtime knows for certain, at the moment of the call, is where each
argument came from. A handle is a state reference the model wrote; anything
else the model authored this turn. No comparison, no threshold, no guess:

```python
MODEL_AUTHORED = "model"

inputs = {
    name: handle_key(value) if is_handle(value) else MODEL_AUTHORED
    for name, value in request.tool_call["args"].items()
}
```

A `NotAuthored` parameter can never appear as `MODEL_AUTHORED` — the handle-only
schema forbids it — so the two mechanisms do not overlap. `NotAuthored` is the
strong form (*this parameter may not be model-authored, enforced*); `inputs` is
the record kept for every other parameter (*this one was, and here it is*).

### `inputs` on `StateEntry`

```python
class StateEntry(TypedDict):
    value: Any
    tool: NotRequired[str]
    seq: NotRequired[int]
    #: Parameter name to origin, for the call that produced this entry:
    #: either ``"model"`` or the ``tool_state`` key the argument was
    #: dereferenced from.
    inputs: NotRequired[dict[str, str]]
```

`StateCaptureMiddleware.awrap_tool_call` already holds `request.tool_call`, so
this costs one dict comprehension over the arguments and no extra plumbing.
It stores parameter names and state keys, never values.

### Provenance is a chain, not a taint

This is the part that earns the change. Each recorded input names either the
model or *another state key*, and that key's entry carries the same record. So
the history of a value is a walk over recorded facts:

```
cds/submit_request/request   inputs {area: "gazet/get_aoi/bbox", dataset: "model"}
  gazet/get_aoi/bbox         inputs {place: "model"}
```

Read bottom-up: the bbox came from `get_aoi` called with a model-authored
`place` — a gazetteer lookup, and fine. Contrast:

```
  gazet/get_aoi/bbox         inputs {bbox: "model"}
```

The same key, the same publishing tool, and the value is the model's own.
Nothing had to propagate a flag to make that visible, and nothing had to
compare two values to infer it. The difference is legible because the record
is of what happened, not of what we concluded.

That is also why this needs no false-positive policy. Taint propagation has to
decide how far a mark spreads, and gets it wrong for somebody; a chain decides
nothing and lets each reader stop where it wants to.

The runtime stops at one level — the call that produced the entry being read.
The chain is a property of the *record*, not a traversal anything here
performs: a host that wants the whole graph has it, and nothing is obliged to
look.

### What reads it

**The listing and the receipt.** The model, the host UI and the user all see
how a stored value came to exist:

```
@state:gazet/get_aoi/bbox — 4 item(s) (bbox: model)
```

**The model.** `prompt.py` already ends with a provenance instruction — name
the tool that produced the data behind a result, and say where it was reused.
`inputs` makes that instruction able to say something it currently cannot: that
a value about to be submitted is one the model wrote itself rather than one a
tool derived. A model that can read the record can also volunteer the caveat,
which is the behaviour worth having and the reason not to keep this
UI-only.

The cost is prompt tokens and a concept the model may over-apply — narrating
provenance nobody asked about. Mitigate in the wording: the instruction should
be to mention model-authored inputs when they bear on the answer's
reliability, not to recite the chain.

**Nothing refuses on it.** The record is never enforced against. `NotAuthored` already
stops the model authoring a parameter directly; `inputs` exists so that a
laundered value is *visible* to a host, a user, or the model reading a listing
— not so that the client can refuse on it.

Enforcement was considered and rejected. It converts visibility into a
guarantee at the price of a tool becoming uncallable whenever its only producer
was itself called with a model-authored argument, and that failure lands on a
user with no way to clear it. A wrong value the user can see beats a right one
they cannot obtain.

**One level, not a walk.** The entry records the call that produced it, and
that is where reading stops. Deeper is available — each named key has its own
entry — but no mechanism here follows it, because at depth "the model authored
something upstream" is true of every value in the session: the model wrote the
search query that found the dataset. A host that wants the full graph can walk
it; the runtime states one fact per value.

## What is lost, with no replacement

**`check_wiring` dies.** All 132 lines. Without kinds there is no way at
connect to know that the consumer was deployed and the producer was not — you
cannot ask "does anything publish this" when nothing declares what it
publishes. The failure moves from a startup report to a call-time refusal, which
is covered below.

`check_wiring` is the only genuinely one-way part of this change. Everything
else degrades to a refusal the model can act on; this degrades to a class of
misconfiguration nobody notices until a user triggers it. It is also a check
that only means anything between toolsets we wrote ourselves, which is the
complaint that opened this proposal.

**`produces` leaves `/health`.** `state_declarations` currently reports both
`produces` and `consumes` so a plain HTTP client can see how a deployment wires
together. Nothing declares what it publishes any more, so `produces` goes.
`consumes` survives in reduced form — parameter, required, handle-only — which
is what the index needs to show that a toolset expects a value it cannot
author.

## Scope

| file | fate |
|---|---|
| `mcp_state/wiring.py` (132) | deleted |
| `mcp_state/injection.py` (383) | resolution path goes — `resolve`, `entries_of_kind`, `satisfiable`, `wants`, `_missing`; `bind_injected` keeps dereference, the unresolved check and the refusal |
| `mcp_runtime/declarations.py` (237) | `Kind` becomes `NotAuthored`; `output_kinds` becomes `output_fields`; `PRODUCES_META_KEY` stays, `kind` leaves its entries; `/health` loses `produces` |
| `mcp_state/middleware.py` (377) | `publishers`, `published_kinds` go; capture records `inputs` |
| `mcp_state/receipts.py` (135) | `via`, `kind` and `breadcrumb()` go |
| `mcp_state/prompt.py` | hidden-parameter and `[state used: ...]` guidance rewritten |
| `mcp_agent/host.py` | receipt rendering stops leading with `kind` |
| `mcp_state/handles.py` (268) | grows — the handle-only branch; `available()` gains provenance |
| `mcp_state/detect.py` (105) | `detect_kind` and its helpers go; `describe` stays |
| `mcp_runtime/kinds.py` (69) | deleted |
| `mcp_state/state.py` | `StateEntry.kind` and `entries_of_kind` go; `inputs` arrives |
| `mcp_runtime/declarations.py` | `qualified()` takes toolset, tool and field |
| `mcp_agent/main.py` | stamps the connection name onto loaded tools |

Roughly 650-750 lines out, one public marker in, one path instead of two,
and no domain vocabulary anywhere in the runtime.

## Work

### Phase 1 — `NotAuthored` and handle-only schemas — **done**

The marker, its `_meta` key and the description note; `handle_only()` beside
`_with_handle_branch`; `offer_handles(..., only=)`; the call-time check for a
literal or a missing required one; `_prune_defs` for the orphaned definitions;
`not_authored` on the `/health` payload and on `StateDeclarations`. Both paths
still exist and `Kind` is untouched — 375 tests pass unchanged, plus 19 new.

### Phase 2 — record what each call was given

`StateEntry.inputs`, written by capture from `request.tool_call["args"]`.
Surface it in the handle listing and on the receipt. Recording only: nothing
refuses on it yet, so the chains can be read against real sessions before
anything acts on them.

Independently useful even if the rest of this proposal is never built — it is
the value-provenance half of the ledger `injection-receipts.md` Phase 4 wants,
and it answers "what produced the data this analysis rests on" without
Langfuse.

### Phase 3 — delete

Remove `Kind`, `wiring.py`, the resolution path, `detect_kind`, `kinds.py`,
and `kind` from `_meta` produces and from `/health`. Rewrite `prompt.py`. Port
gazet (4 tags) and cds (1 tag). Rename the example toolsets' data keys.
Breaking: bump accordingly.

Phases 1 and 2 are additive and shippable independently. Phase 3 is the only
one that breaks anyone.

## Open questions

**Does the `[state updated: ...]` breadcrumb change?** It currently names keys
and explains the two ways to use one:

```
[state updated: dataset-search/geometry — pass the bare key to inspect_state to
read one; pass @state:<key> only to a tool parameter whose schema accepts it]
```

Once the key is both the model's only route to a value *and* the main carrier
of what that value is, this is doing more work than it was designed for. It
also says nothing about shape or provenance, both of which are available:

```
@state:dataset-search/search_datasets/geometry — 1 feature(s), 2000 vertices (query: model)
@state:gazet/get_aoi/bbox — 4 item(s) (bbox: model)
@state:raster-ops/clip_raster/clipped — object with 2 key(s) (aoi: dataset-search/search_datasets/geometry, dataset_id: model)
```

The trade is tokens on every capturing tool call against the model choosing
better and the provenance being visible where it is acted on. `available()`
already renders the second form for a host that asks; the question is whether
the breadcrumb becomes it. Undecided — read a real session first.

## Out of scope

- The Phase 4 ledger from `injection-receipts.md` as a separate channel.
  `StateEntry.inputs` is the same information held per value instead of per
  call, and a walk over the entries reconstructs the graph. If the ledger is
  still wanted afterwards it is a rendering, not a second store.
- Changing which stored value a handle resolves to. The model chooses, and
  under this proposal it is the only thing that chooses.
- Persisting provenance outside the session.
