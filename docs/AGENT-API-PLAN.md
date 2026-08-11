# `mcp_agent_api` — plan

**Status: designed, no code written.** Supersedes the draft at `da327f8` on the
unmerged `feat/agent-api` branch, which predated receipts, `mcp_agent.host` and
the 0.3.0 extension points.

A FastAPI package exposing the agent `mcp_agent` already builds — the one that
discovers toolsets behind an index URL and drives their tools — over HTTP, so a
chat widget on a website or a bespoke frontend can consume it.

## What it must do

A user sends a query and gets back:

1. the model's answer, with **tokens streamed** as they arrive;
2. **which tools were called**, and with what;
3. **where state came from** — what each tool was handed out of session state;
4. **where state went** — what each tool published into it;
5. **MCP Apps views**, rendered from the `ui://` bundles the toolsets serve.

Points 3 and 5 are the ones no off-the-shelf agent protocol gives you, and they
are the reason this is worth designing rather than adopting.

## Who it is for

Two audiences, and the design is pulled between them deliberately.

**A custom frontend**, which is the immediate driver: integrating the DSS agent
into the CDS website, where the chat may live as a popup, as a search-bar
integration, or as something else again. Such a client wants *everything* —
structured receipts it can style, view bundles it can mount where it likes, tool
calls it can lay out on its own terms.

**Something simple**, on the order of a Chainlit host: a client that renders
messages in order and wants the interesting parts to arrive already legible,
without correlation logic or a rendering pipeline.

Where the two conflict, the rule is: **send everything structured, and include a
rendered form beside it.** A simple client reads the rendered form; a custom one
ignores it. Neither is asked to negotiate.

## What the runtime already provides

| | |
| --- | --- |
| `BuiltAgent` (0.3.0) | `agent`, `connections`, `tools`, `withheld`, `required` from one call |
| `build_agent`/`with_session_state` (0.3.0) | take `system_prompt`, `extra_tools`, `middleware`, so this package extends rather than forks |
| `mcp_agent.host` (0.4.0) | `view_bundles`, `view_uri_for`, `view_props`, `tool_name`, `step_input` — **imports no UI framework** |
| Receipts (0.2.1, 0.4.2) | `receipts_of(artifact)` gives `{parameter: {key, via, kind, tool}}` for both paths |
| Checkpointing (0.1.7) | `MCP_AGENT_CHECKPOINT` selects `memory` or a PostgreSQL URL; `build_agent` accepts any LangGraph saver; `[checkpointing-postgres]` carries the driver |

`mcp_agent.host` is what lets an API process resolve a turn's views server-side
without dragging Chainlit in.

**`TurnResult` is not among these.** `run_turn` is built on `agent.ainvoke`
(`main.py:750`), so everything it derives — `answer` flattened through `.text`,
`citations` from `answer_citations`, `sidecar`, and the `seen`-index separating
this turn's replies from history — is computed only *after* a whole turn
completes. A streaming surface never reaches that point. `TurnResult` serves the
deferred one-shot endpoint and nothing else.

## Prerequisite: a streaming sibling to `run_turn`

**Do this first, in `mcp_agent`, before the API package exists.** A streaming
host needs `run_turn`'s outputs as they arrive, so the loop that reads
`agent.astream` and pulls out tokens, tool calls, receipts, captured state and
citations belongs beside `run_turn` in the runtime — not inside a route handler.
`answer_citations` already handles three provider block shapes and was got wrong
once against a real Mistral answer; a second copy in a route handler is a second
thing to get wrong, and the CLI and Chainlit hosts would not benefit from either.

**One call, two stream modes, everything arrives.** Receipts ride
`ToolMessage.artifact` rather than content, and they survive streaming. With
`stream_mode=["updates", "messages"]`:

```
tokens streamed   : Clipped chirps to your area.      (messages)
tool actually ran : True
receipts recovered: {'aoi': {'key': 'dataset_search/geometry', 'via': 'declaration',
                             'kind': 'geojson.AreaOfInterest', 'tool': 'search_datasets'}}
tool_state keys   : ['clip_raster/geometry']          (updates)
```

Tokens come over `messages`; receipts and captured state over `updates`, read off
the artifact of each `ToolMessage`. Nothing has to be persisted to recover them.

**Testing trap.** `GenericFakeChatModel` — the stub the session-state example
uses — **drops `tool_calls` on its streaming path.** As soon as `messages` is
among the stream modes LangGraph streams the model, so a test built on that stub
emits no tool call, runs no tool, produces no artifact, and passes while
exercising nothing. Tests for this primitive need a `BaseChatModel` whose
`_astream` yields real `tool_call_chunks`, and should assert a tool actually ran
rather than only that the assembled output looks right.

## Shape of the package

Three layers, each importable without the one above it — the same split as
`mcp_agent.host` (helpers, no framework) and `mcp_agent.web` (a host built on
them). The requirement is that a consumer can take the minimum needed to drive
the agent from a UI, *or* a complete service.

```
mcp_agent_api.events    — a pure async generator: agent astream -> AG-UI events.
                          Imports no FastAPI. For a consumer using its own
                          transport (websocket, queue) or its own framework.

mcp_agent_api.routes    — an APIRouter: POST /runs (SSE),
                          GET /threads/{id}, GET /threads/{id}/state/{key},
                          GET /views/{toolset}/{view}.
                          Mountable into an existing FastAPI app, so a
                          deployment brings its own auth, CORS and middleware.

mcp_agent_api.app       — create_app(build=None): lifespan builds the agent
                          via `build` (default: the runtime's build_agent),
                          reads settings, sets CORS. A module-level `app` is
                          built from it for uvicorn.
```

Everything a deployment might need to replace is a parameter of `create_app`,
not a module-level global — the checkpointer above all, so swapping
`InMemorySaver` for Postgres is an argument rather than a fork.

## The agent seam

**Both `routes` and `app` take an agent from the caller.** `routes` takes an
already-built agent with its connections and tools. `create_app` takes an
optional `build` factory and defaults to one calling the runtime's `build_agent`
against an index URL, passing `system_prompt`, `extra_tools` and `middleware`
through — so a standalone deployment is one function call, and a consumer
wanting everything else `app` provides is not forced down to `routes` merely to
supply its own agent.

**A factory, not a built agent.** `build_agent` is async and connects to MCP
servers, so it must run inside the lifespan; `create_app()` is called at import
time, synchronously. A caller handing over a finished agent would have had to run
async code before there is a loop. So `build` is an async callable invoked during
startup, and a test that already has an agent passes `lambda: already_built`.

**Read the result by attribute, never by position.** The factory may return
anything exposing `.agent`, `.connections`, `.tools`, `.withheld` and `.required`
— a Protocol, not a concrete type. The runtime's `BuiltAgent` is
`(agent, connections, tools, withheld, required)` and dss's is
`(agent, connections, tools, required, checkpointer, skills, withheld)`.
Positions 4 and 5 are swapped, so unpacking one as the other yields `required`
where `withheld` belongs — a dict where a list is expected, failing at use rather
than at the unpack.

**The seam is required by dss, not tidiness.** `dss_agent.main.build_agent` never
calls the runtime's. It imports `fetch_connections` and `with_session_state` and
reimplements the sequence, for three reasons it cannot give up:

1. it wraps the MCP client in `with_tracing(...)` for Langfuse;
2. `check_required_tools(skills, (t.name for t in tools))` validates skills
   against the *connected* tool names, so it needs the tools mid-sequence;
3. it returns a richer `BuiltAgent` carrying `checkpointer` and `skills`.

The skills themselves fit the extension points exactly as intended — dss already
composes `system_prompt` from `skills_manifest` and passes `make_load_skill` as
an `extra_tool`. It is the surrounding sequence, not the skills, that cannot be
expressed through hooks. So dss supplies its own `build` factory, whether it
mounts `routes` in an app of its own or takes `create_app` wholesale.

## Wire protocol

**AG-UI, without the stock-widget goal.** AG-UI natively covers the answer, token
streaming, tool calls and state deltas. Receipts and views have no place in its
vocabulary and ride the activity channel below, which a stock widget ignores —
so "a CopilotKit widget works out of the box" is not a benefit to design around,
and event design is not constrained by what such a widget can consume. A stock
widget degrades to text and tool calls.

What justifies it is the four native rows: token chunking, the tool-call
lifecycle, and `STATE_DELTA` carrying RFC 6902 JSON Patch, which fits
`tool_state` mutations directly. A bespoke vocabulary would re-derive them almost
identically. `ag-ui-protocol` is maintained (0.1.19 at time of writing).

Namespaced storage keys need JSON Pointer escaping — the `/` in
`dataset_search/geometry` becomes `~1` in a patch path. Verified both ways: the
escaped path applies, and the unescaped one raises `JsonPointerException`
("member 'dataset_search' not found") rather than writing to the wrong place, so
getting this wrong fails loudly server-side.

**`ag_ui.core` types, our own generator.** Depend on `ag_ui.core` and
`ag_ui.encoder`; write the `astream` loop rather than adopting
`ag-ui-langgraph`'s `LangGraphAgent`, since shaping what `tool_state` puts on the
wire means working against its state middleware.

**Receipts and views ride `ACTIVITY_*`, not `CUSTOM`.** `ActivityMessage` is a
first-class role in AG-UI's `Message` union — "an activity progress message
emitted between chat messages" — carrying `activityType` and a structured
`content`, with `ActivitySnapshotEvent` (plus `messageId` and a `replace` flag)
and `ActivityDeltaEvent` (RFC 6902) to deliver it. `CustomEvent` offers only
`name` and `value`, with no tie to a message: a client would have to catch it,
dig a `toolCallId` out of `value`, and work out where it belongs. Because an
activity *is* a message, a client rendering messages in order shows a receipt in
the right place with no correlation code.

Activities are wire types generated at stream time. They are never LangChain
messages, so they never reach the model or the checkpointer, and
server-authoritative history means a client echoing them back is ignored.

### Exercised against a real client

A representative turn was encoded with `ag_ui.encoder.EventEncoder` (Python) and
replayed through `@ag-ui/client`'s own pipeline —
`parseSSEStream` → `verifyEvents` → `AbstractAgent.runAgent` — then
`agent.messages` was read as a client rendering messages in order would. The
result:

```
activity:tools.withheld   {"tools":["submit_request"],"display":"1 tool unavailable"}
assistant                 toolCalls=[{"id":"call_7","function":{"name":"clip_raster",…}}]
tool                      Clipped 1 raster.
activity:state.consumed   {"toolCallId":"call_7","received":{"aoi":{…}}}
activity:mcp.view         {"uri":"ui://raster-ops/map","data":{"count":1}}
assistant                 Clipped chirps to your area.
activity:answer.citations {"ids":["chirps"],"display":"Sources: chirps"}
```

The receipt lands beside the tool call it belongs to, with no correlation code
on the client — which is the whole argument for `ACTIVITY_*` over `CUSTOM`. An
`ACTIVITY_DELTA` applied onto its snapshot (`bytes: 48213` arrived separately),
and a `STATE_SNAPSHOT` keyed `dataset_search/geometry` came through intact.

Six constraints came out of it, each of which the generator has to respect:

- **The wire is camelCase.** `messageId`, `activityType`, `toolCallId` — Pydantic
  aliases them on encode. Fields inside `content` are ours to name; the examples
  below use camelCase to match.
- **A message's position is fixed when it is first created, not when it is
  completed.** An activity emitted between `TEXT_MESSAGE_START` and
  `TEXT_MESSAGE_END` is legal — the verifier accepts it — but the assistant
  message already exists, so the activity renders *after* the answer. **Emit a
  tool call's activities before opening the answer's text message.** Citations
  are the deliberate exception: they belong after the answer and are only known
  then.
- **`replace: false` is create-if-absent, not merge.** On an existing activity
  message it is a no-op — the second snapshot's content is discarded, not merged
  in. Restate with `replace: true` (the default) or amend with `ACTIVITY_DELTA`.
- **Deltas fail silently.** An `ACTIVITY_DELTA` for a `messageId` never
  snapshotted is dropped with no error, and a patch that fails to apply is a
  `console.warn`. Neither ends the run. So the snapshot must carry the truth and
  deltas are an optimisation — never the only carrier of a fact.
- **Activity content must be a JSON object.** The TS client accepts an array, but
  `ActivityMessage.content` is `Dict[str, Any]` in `ag_ui.core`, so a bare list
  would not survive being read back server-side. `answer.citations` therefore
  sends `{"ids": [...], "display": "…"}`, never a bare list.
- **Nothing may precede `RUN_STARTED`** — the verifier rejects the stream. The
  withheld-tools activity goes immediately after it, not before.

`@ag-ui/client` also exposes `onActivitySnapshotEvent` and
`onActivityDeltaEvent` subscriber hooks, so a custom frontend intercepts
receipts and views without reimplementing the message pipeline.

## What the client is sent

**Receipts — structured fields plus a rendered line.** One activity per tool
call, carrying the receipt's own fields *and* a `display` string:

```json
{"type": "ACTIVITY_SNAPSHOT", "messageId": "act_1", "activityType": "state.consumed",
 "content": {"toolCallId": "call_7",
             "received": {"aoi": {"key": "dataset_search/geometry",
                                  "via": "declaration",
                                  "kind": "geojson.AreaOfInterest",
                                  "tool": "search_datasets",
                                  "display": "← dataset_search/geometry · geojson.AreaOfInterest · 1 feature(s), 2000 vertices · from search_datasets"}}}}
```

A minimal client prints `display` and is done, and it matches the bundled
Chainlit host because it is the same `mcp_agent.host.step_input` output,
generated server-side. A bespoke client uses the fields — grouping by publishing
tool, styling the two paths differently, or linking `key` to the state route.

**A client must branch on `via`, never on the rendered string.** `declaration`
means the model never saw the parameter and the client chose the value; `handle`
means the model named it. The `←` prefix encodes that for display only.

This is the host-side channel, distinct from and additional to the model-facing
`[state used: …]` note riding in the tool message content — that one costs
context and is what lets the model describe its own provenance.

**`tool_state` on the wire — metadata only.** Values are replaced by
`{kind, tool, seq, bytes}` in `STATE_SNAPSHOT`/`STATE_DELTA`; a frontend that
wants the geometry fetches it from `GET /threads/{thread_id}/state/{key}`. Keeps
the delta small and stays consistent with why `tool_state` exists at all.

**Views — an activity carrying the URI, plus a fetch route.** An `activityType`
of `mcp.view` carries the `ui://` URI and the tool's structured data; the client
fetches HTML from `GET /views/{toolset}/{view}` and caches it. The route takes the
URI's two components as path segments rather than the whole `ui://toolset/view`
string, which would need its scheme and slashes escaped into one segment. A
bundle can be ~300 kB and would otherwise repeat every turn.
`view_bundles`/`view_uri_for`/`view_props` do the server side, and
`restore_structured` reassembles the tool's whole structured content so a view
cannot tell whether a value took the long way round.

**Citations reach the client as an activity too.** `answer_citations` yields the
ids the model cited on `reference` blocks, and the Chainlit host renders them as
a Sources footnote. For a scientific data service that is not decoration, so an
`activityType` of `answer.citations` carries the ids with a `display` string,
under the same rule as receipts. Emitted once the answer completes, since the ids
are known only then.

**Failures are told, not swallowed.** Three distinct cases:

- **Withheld tools** — `BuiltAgent.withheld` lists tools dropped as uncallable.
  The Chainlit host says "Some tools are unavailable" at connect; the API sends
  the same as an activity when a run opens, so a client can explain a capability
  it does not have rather than appearing to ignore the request.
- **A tool erroring** — surfaces on its `TOOL_CALL_RESULT` and in the message, as
  it already does. No special handling.
- **The run failing** — `RUN_ERROR`, with the message the model or transport
  produced. A toolset going down mid-run lands here.

**No client-to-server capability negotiation.** Every client is sent every
activity; one that cannot render views drops them. A client declaring what it
supports would add a contract to keep in step for a saving of a few hundred bytes
of URI and metadata, and would mean a client gaining a capability needs a server
that already knows about it. The heavy part of a view is the bundle, and that is
already behind a fetch the client simply never makes.

Note this rejects one direction only. AG-UI has `AgentCapabilities`
(`ag_ui.core.capabilities`, with `identity`, `transport`, `tools`, `output`,
`state`, `execution` groups and a `custom` escape hatch) and `@ag-ui/client`'s
`AbstractAgent` has an optional `getCapabilities()` — but that is the *agent
describing itself*, which changes nothing about what a run emits. Whether to
advertise the activity types and the view route through it was weighed and declined for v1 — see "Open".

**Streaming only, for now.** SSE is the single surface; a client that wants one
answer opens the stream and waits for `RUN_FINISHED`. A search-bar integration
may want a plain `POST /chat` returning JSON, but the seam for it is
`TurnResult`, which already holds everything such a response would serialise — so
adding it later is a serialisation change rather than a redesign.

## Threads and state

**Server-authoritative; `InMemorySaver` as the default, injected.** The
checkpointer owns history and `tool_state` keyed on `thread_id`. Of a client's
`RunAgentInput.messages`, only the trailing user message is taken; the rest is
ignored — a deliberate divergence from AG-UI's client-is-authoritative
convention, to keep large injected values out of the browser.

Persistence is a deployment setting, not work: `MCP_AGENT_CHECKPOINT` and the
`[checkpointing-postgres]` extra already exist. What remains true is that the
**default** is prototype-grade — in-memory means history is lost on restart, one
replica only, and threads accumulate with no eviction. Fine for the runtime's own
default; not something a CDS deployment should inherit by saying nothing.

**A thread can be read back — messages only.** `GET /threads/{id}` returns the
thread's messages, so a page reload restores the conversation, which is the point
of the server owning history. Past turns' activities are **not** rebuilt: doing so
would mean re-deriving every historical turn's receipts from stored artifacts and
re-resolving its view props, at the moment a page is trying to load. A client
wanting a past turn's geometry or view has the state and view routes for exactly
that.

**A thread belongs to one surface.** A search bar and a popup on the same page
are separate conversations with separate `thread_id`s. Sharing one would mean a
popup opening onto history the user did not know they had, and a client-side
story for carrying the id between surfaces. If continuity is wanted later, it is
a client-side decision about which id to send, not a server change.

## Auth and credentials

**Anonymous or authenticated, no step-up.** Verify a bearer when present,
otherwise run anonymously. A public popup that demands OAuth before "hello" does
not get used. AG-UI's `resume` + `Interrupt` would give a real step-up channel if
that changes.

**Credential env fallback stays on.** `resolve_credentials` uses, for each header
a toolset declares and no caller supplied, the matching environment variable and
then `.env`. Keeping that makes the API behave like the CLI and Chainlit hosts,
and local development against one account just works.

The consequence must not be glossed: **a deployment that sets `X_CDS_TOKEN` gives
every anonymous user that account.** Requests are attributed to one shared
identity and abuse cannot be traced to a user. Such tools do not fail with a
message the model can relay — the fallback is silent — so a deployment wanting
per-user credentials must leave the variable unset.

**State reads — the thread id is the capability, for now.** An unguessable
`thread_id` is the only credential on `GET /threads/{id}/state/{key}`. This is
what lets an anonymous thread draw its own map, which the anonymous-first
decision requires. The id leaks in logs, referrers and browser history, so this
tightens to a gateway session handle when that lands — the same treatment
credentials get.

**Abuse and cost control are out of scope, and that is an assumption, not an
oversight.** This API runs LLM calls against a server-configured key, so an
unprotected public deployment is an open budget. Rate limiting, quotas and bot
protection belong to whatever sits in front of it. Stated here so it is
explicitly somebody's job.

**Model is server-configured** — `PROVIDER_MODEL` / `PROVIDER_API_KEY` via
`init_chat_model`, as `mcp_agent.main.AgentSettings` does. Bring-your-own-key
makes no sense for a public popup.

## Three token planes

The MCP authorization spec makes these non-negotiable: an MCP server is an OAuth
2.1 resource server that **MUST** validate the token's audience and **MUST NOT**
accept or transit any other token, and a client **MUST** send RFC 8707 `resource`
naming the specific server.

| | Who holds it | What it is |
| --- | --- | --- |
| A | browser → `mcp_agent_api` | this API's own OAuth2 bearer |
| B | `mcp_agent_api` → each toolset | audience-bound per MCP server, RFC 8707 |
| C | toolset tool → third-party API | the user's own credential for that API |

Plane C cannot ride `Authorization` — that slot belongs to plane B. The mechanism
this repo already has *is* the sanctioned channel: `CREDENTIAL_HEADERS` declared
per toolset, injected per request by `user_credentials()` (`main.py:389`),
reaching only the toolset that declared the header. The gateway resolves plane A
identity to plane C secrets; nothing new is needed on the wire. Design the seam,
defer the wiring — the vault is a separate deployable in its own repo.

## Packaging

**`src/mcp_agent_api/`, an `[api]` extra**, and **`[agent]` splits first**.
`[api]` cannot depend on `[agent]`: `[agent]` bundles `chainlit`, so every API
deployment would install a UI framework it never imports. `mcp_agent.main` and
`mcp_agent.host` need `[state]` and `httpx`; the Chainlit dependency belongs only
to `mcp_agent.web`.

So `[agent]` becomes the chainlit-free agent dependencies and a new `[web]` adds
Chainlit on top, leaving `[api]` and `[agent]` both lean. The module layering
from #44 already draws this line — this makes the packaging agree with it.

**That is a breaking change**: anyone installing `[agent]` for the Chainlit host
must move to `[web]`, so it lands as `feat!` and takes the version to 0.5.0.
Since dss pins `<0.5.0`, its four package declarations move in the same step.

**Do it first, and do not soften it.** Every consumer is in the same hands, and
dss has no API of its own to hold still — so there is no case for a deprecation
window, a transitional extra that installs both, or `[agent]` keeping Chainlit
for a release. Landing it before any API code means the 0.5.0 bump happens once,
against a small diff, rather than alongside a new package.

**Reference clients — examples only, one per audience.** An `examples/` directory
outside `src/`, exactly as `examples/session-state/` is. Two clients, because the
two audiences are the whole design premise and a single example would silently
favour one: a minimal one that renders messages and `display` strings in order,
and a fuller one that mounts a view and styles receipts from their fields. No
component is published.

## Deferred: state authored by the client

**Not in v1.** Threads are server-authoritative and views are read-only surfaces
that can nudge the conversation with text. Written down because the obvious next
feature — an MCP App that draws an area of interest and hands it to a tool —
needs all of this, and none of it is a small change.

Today a view can only send text. `js/mcp-view/src/host.ts` exposes exactly
`onData(handler)` and `sendMessage(text)`, and the latter builds
`{role: "user", content: [{type: "text", text}]}`. So a draw-an-AOI app can either
stringify a 2000-vertex polygon into a chat message — precisely the failure
session state exists to prevent — or send prose and lose the geometry.

Supporting it properly needs four things:

1. **A structured channel in the bridge** — `sendData(key, value)` alongside
   `sendMessage`. This changes the published npm package, so it has the longest
   tail. Worth first checking what `@modelcontextprotocol/ext-apps` already
   offers; the 58-line wrapper here is narrower than the SDK.
2. **A write route** — `PUT /threads/{id}/state/{key}`, mirroring the read.
3. **A whitelist**, naming which keys a client may write, rather than a blanket
   door into `tool_state`.
4. **A synthetic message** — `HumanMessage("Manually selected data for field
   aoi")`, so drawing on a map is a visible conversational act while the geometry
   stays in state.

**The kind must be declared on the write, not detected.** `detect_kind` reports a
FeatureCollection as `GEOJSON_FOOTPRINT`, never `GEOJSON_AREA_OF_INTEREST` —
"both are FeatureCollections and the bytes cannot tell them apart". A pushed AOI
relying on detection would silently fail to match an
`aoi: Kind(GEOJSON_AREA_OF_INTEREST)` parameter and degrade to a handle at best.

Done properly, the payoff is that the existing machinery needs no changes: the
tool's tagged parameter is FILLed from `client/aoi`, the geometry never enters the
transcript, and the tool step reads `aoi: ← client/aoi · … · from <the app>` — the
user sees that their own drawing is what the tool ran on.

## Open

Nothing here blocks starting.

- **Whether to port `_split_markdown_safe`** from `ecmwf/digital-twin-assistant`
  (`backend/src/destine_digital_twin_assistant/api/streaming.py`) — ~150 lines
  holding back incomplete links, headings, tables, fenced code and emphasis so a
  streaming UI never flashes half-rendered markdown. Portable, and hard to get
  right twice. Defer until a UI shows the need.
- **Whether to advertise `AgentCapabilities`.** Decided against for v1: a custom
  frontend has to know how to render `state.consumed` and `mcp.view` regardless,
  so discovering their names buys nothing, and a `custom` block naming them is
  one more contract to keep true as they change. The case for it appears when a
  second agent sits behind the same UI, or when the view route moves — at which
  point `getCapabilities()` is the seam already waiting.
- **`forwarded_props` carries the gateway session handle** when that lands.
