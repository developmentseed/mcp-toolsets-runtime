# Consuming `mcp-toolsets-runtime`

This is what a repo that installs the runtime has to do. There are four
personas — most repos are the first, and the rest combine freely:

1. **Serving tools** — you have toolsets and want to expose them as MCP servers.
   You need `mcp_runtime` and the plugin contract. That's it.
2. **Rendering tool results as UI views** — a view follows the **MCP Apps**
   standard, so *what* you render in decides how much you do
   ([UI views](#3-ui-views-rendered-by-any-mcp-apps-host)). An external host
   (Claude.ai, ChatGPT, Goose, VS Code) needs
   [3a](#3a-declare--build-the-bundle)–[3b](#3b-the-view-side-bridge-developmentseedmcp-view)
   and nothing else. The bundled Chainlit host renders them in its side panel;
   it needs the `[web]` extra and the host element installed at build time
   ([3c](#3c-only-if-you-also-run-the-bundled-chainlit-agent)). **Your own
   frontend** is a host like any other — it implements the host end of the same
   `ui/*` bridge, or embeds views with a standard MCP Apps client
   ([3d](#3d-rendering-views-in-your-own-frontend)). Custom-built views want the
   `@developmentseed/mcp-view` npm bridge either way.
3. **Running your own agent** — you drive MCP tools from your own LangGraph
   agent rather than the bundled chat host. You need the `[state]` extra to keep
   large tool values out of the model's context
   ([Session state](#4-session-state-keeping-large-values-out-of-the-model)).
4. **Serving an agent over HTTP** — you want a chat *backend*, not a chat UI:
   your own frontend, AG-UI over SSE between them. You need the `[api]` extra
   ([Serving the agent over HTTP](#5-serving-the-agent-over-http-mcp_agent_api)).
   It layers over 3 but does not require it — the bundled agent serves as it is,
   with no agent code of your own.

Personas 2 and 3 are independent: your own frontend can talk to the bundled
agent, and your own agent can serve an external host. Doing both is
[3a](#3a-declare--build-the-bundle)–[3b](#3b-the-view-side-bridge-developmentseedmcp-view)
plus [4](#4-session-state-keeping-large-values-out-of-the-model), and none of
[3c](#3c-only-if-you-also-run-the-bundled-chainlit-agent).

The web host (`mcp-agent-web`) is **bring-your-own-model**: it holds no provider
key. Each user sets a `provider:model` + their API key in the chat's ⚙ settings
(env `PROVIDER_MODEL` / `PROVIDER_API_KEY` only *pre-fill* for local use), so a
hosted deployment stores no secret. The `[web]` extra stays provider-agnostic —
install the provider package your users need at image-build time (e.g.
`uv pip install langchain-anthropic`). Deployment scaffolding (a `Dockerfile` and
Helm chart for the hosted chat) is a **consumer** concern — see
[`mcp-toolsets`](https://github.com/developmentseed/mcp-toolsets)'
`Dockerfile.chat` / `charts/mcp-chat` as the reference.

---

## 1. Install

[![PyPI](https://img.shields.io/pypi/v/mcp-toolsets-runtime?label=PyPI)](https://pypi.org/project/mcp-toolsets-runtime/)

It's on PyPI — an ordinary dependency, no source override:

```toml
# pyproject.toml
dependencies = [
    "mcp-toolsets-runtime",          # base: mcp_runtime + mcp_cli
    # "mcp-toolsets-runtime[state]", # your own agent (see "Session state")
    # "mcp-toolsets-runtime[agent]", # build_agent/run_turn + host helpers
    # "mcp-toolsets-runtime[web]",   # the bundled Chainlit host, on top
    # "mcp-toolsets-runtime[api]",   # the agent over HTTP, beside [web] not under it
]
```

```bash
uv lock && uv sync
```

Imports are unchanged from the old in-repo workspace packages — `from
mcp_runtime.server import build_server`, `from mcp_agent.main import ...`, etc.
`uv.lock` pins the exact version that resolved, so that — not a number in this
document — is what makes your builds reproducible. **Upgrade** with `uv lock
--upgrade-package mcp-toolsets-runtime`.

The package is pre-1.0, where a minor release may break. If you'd rather take
those deliberately, bound the dependency at the next minor in your own
`pyproject.toml`.

Available console scripts: `mcp-serve`, `mcp-serve-local`, `mcp-index` (base);
`mcp-cli` (base); `mcp-agent` (needs `[agent]`); `mcp-agent-web` (needs `[web]`).
The HTTP API has no script of its own — it is an ASGI application, served with
`uvicorn mcp_agent_api.app:app` (needs `[api]`).

---

## 2. Author a toolset (the plugin contract)

Scaffold one with the bundled generator (run from your repo root) — it lays down
the package, tests, and (with `--with-ui`) a Vite view wired to
`@developmentseed/mcp-view`, then `uv add`s it to the workspace:

```bash
mcp-toolset new my-toolset            # or: mcp-toolset new my-toolset --with-ui
```

The rest of this section is what that scaffold contains. `mcp_runtime` discovers
a toolset by convention. Given `TOOLSET=my-toolset`, it imports `my_toolset.tools`
and reads module-level exports:

```python
# my_toolset/tools/__init__.py
from langchain_core.tools import tool
from mcp_runtime.tool_result import ToolResult


@tool
def search(query: str) -> ToolResult:
    """One-line docstring — becomes the tool's MCP description (required)."""
    return ToolResult(message=f"results for {query}", ...)


TOOLS = [search]                      # required: non-empty list of tools
# VIEWS = {"search": "gallery"}       # optional: tool_name -> view_id (see "UI views")
# CREDENTIAL_HEADERS = ["X-My-Token"] # optional: headers the tools read off the transport
```

- **`TOOLS`** — every tool must return a `ToolResult` (its annotations become the
  MCP output schema; `build_server` rejects tools that don't at startup).
- **`CREDENTIAL_HEADERS`** — names of headers the tools read off the transport.
  The runtime derives the server `instructions` from these so the model is told
  a credential rides the connection and it shouldn't ask the user for it. The
  credential never enters the model context.
- **`VIEWS`** — see [UI views](#3-ui-views-rendered-by-any-mcp-apps-host).

A tool may also tag a parameter `NotAuthored` — see
[Session state](#4-session-state-keeping-large-values-out-of-the-model). Which
half of that is optional depends on which end you control:

- **The tag is optional.** An agent built on `mcp_state` keeps large values out
  of the model's context whether or not you use it; tagging makes it cheaper and
  safer.
- **The agent is not.** Keeping a value out of the context is entirely
  client-side work. Serve the same toolset to Claude.ai, ChatGPT, or any other
  MCP client and none of it happens — the tag is advertised in `_meta`, nothing
  reads it, and large values land in the transcript as they always would.

So tag for the benefit of agents that understand it, and size your tool returns
on the assumption that a client might not.

Serve it:

```bash
TOOLSET=my-toolset mcp-serve         # serves this toolset's tools over MCP
```

### Serving all of them at once, locally

`mcp-serve` is one toolset per process, which is what production wants — every
toolset its own service, behind one entry point, with `mcp-index` presenting
them as one directory. Locally you have neither, so `mcp-serve-local` builds
that same shape in a single process:

```bash
mcp-serve-local                      # from your repo root
```

Each toolset is mounted at `/<toolset>`, so `/<toolset>/mcp` and
`/<toolset>/health` are the paths production serves, and `/` returns the same
directory document `mcp-index` returns. `mcp-cli`, `mcp-agent` and a plain
`MultiServerMCPClient` all consume that already:

```bash
mcp-cli call forecast city=Lisbon --url http://localhost:8000/weather/mcp
mcp-agent chat http://localhost:8000/     # every toolset, one agent
```

With no `TOOLSETS` set it serves every directory under `./toolsets` that has a
`pyproject.toml` — the layout `mcp-toolset new` produces. That only supplies the
*names*; each one is still resolved by importing `<name>.tools`, so they must be
installed in the environment you run from. Name them explicitly (`TOOLSETS=a,b`)
and the directory is never read, which is what to do if your toolsets live
somewhere else — or point `TOOLSETS_DIR` at them. `HOST` and `PORT` work as they
do for `mcp-serve`.

### Serving the directory in production (`mcp-index`)

`mcp-index` is the deployed counterpart of that `/` route: one service beside
the toolsets that asks the platform which of them are **running**, asks each
one's `/health` for its tool names, and serves the aggregate. It asks every
time rather than reading a list fixed at deploy time, so a toolset that failed
to start is absent from the directory instead of advertised.

```bash
PUBLIC_URL=https://mcp.example.com mcp-index
```

| Variable | Default | Meaning |
| --- | --- | --- |
| `PUBLIC_URL` | *required* | External base URL the toolsets are served under |
| `HOST` / `PORT` | `127.0.0.1` / `8000` | Where the index itself listens |
| `MCP_INDEX_DISCOVERY` | `kubernetes` | `kubernetes` or `ecs` |
| `MCP_ECS_CLUSTER` | — | Cluster to list; required when discovery is `ecs` |
| `MCP_TOOLSET_PORT` | `8000` | Port to address a toolset on when its registration names none (`ecs` only) |

**`kubernetes`** lists Services in the index's own namespace carrying the
`mcp-toolsets/toolset` label, and addresses each by its Service name. It needs
nothing installed and no credentials beyond the pod's service account, which
must be allowed to `get` and `list` services in that namespace.

**`ecs`** lists services on `MCP_ECS_CLUSTER` tagged `mcp-toolsets/toolset`,
and addresses each by the Cloud Map registration ECS made for it — so a service
must be both tagged and registered to appear. Install the extra
(`mcp-toolsets-runtime[aws]`) and give the index's task role:

```
ecs:ListServices, ecs:DescribeServices,
servicediscovery:GetService, servicediscovery:GetNamespace
```

Selecting `ecs` without the extra, or without a cluster, fails at startup
rather than serving an empty directory with a 200.

### How the runtime finds your toolset

Discovery is a **Python import**, not a directory scan. `TOOLSET=<name>` is
converted kebab→snake and the runtime imports `<name>.tools` (e.g.
`my-toolset` → `import my_toolset.tools`). So the one requirement is that your
toolset package is **installed in the environment** you serve from — on the
Python import path, not at any particular folder. `mcp-toolset new` handles this
by running `uv add`, which adds the toolset to your deps so `uv sync` installs it
editable.

A typical consumer repo:

```text
your-repo/
├── pyproject.toml            # deps: mcp-toolsets-runtime
│                             # [tool.uv.workspace] members = ["toolsets/*"]
├── uv.lock
└── toolsets/
    └── my-toolset/
        ├── pyproject.toml    # name = "my-toolset"; deps: mcp-toolsets-runtime
        └── src/
            └── my_toolset/   # importable package ("my-toolset" → "my_toolset")
                ├── __init__.py
                └── tools.py   # exports TOOLS (+ optional VIEWS / CREDENTIAL_HEADERS)
```

The `src/` layout is just the standard Python "src layout" — the folder path is
irrelevant to discovery; only the installed, importable package name is. That's
why a toolset can live wherever you like (a nested path, or even a separate
distribution) as long as it ends up on the import path.

**Custom module or non-conventional layout.** To point the runtime at a module
that doesn't match the `<name>.tools` convention, set `TOOLSET_MODULE`:

```bash
TOOLSET=my-toolset TOOLSET_MODULE=my_pkg.mcp_tools mcp-serve
```

### Testing the contract

You don't write the assertions — **`build_server` is the conformance check**.
Every clause is validated there, and each raises at startup rather than on a
user's call:

| Checked | Raised by |
| --- | --- |
| the module imports, and exports a non-empty `TOOLS` | `load_tools` |
| every entry is a LangChain `BaseTool` | `load_tools` |
| every tool returns a `ToolResult`, with a required `message` | `to_fastmcp` |
| `CREDENTIAL_HEADERS` is a list of header names | `load_credential_headers` |
| every `VIEWS` entry names a real tool, with a built bundle on disk | `load_views` |
| every `NotAuthored` sits on a parameter that exists | `with_state_meta` |

What the runtime *can't* ship is the **enumeration**. Discovery is by Python
import, not a directory scan, so only your repo knows which toolsets exist and
where their names come from. That makes the sweep about three lines:

```python
import pytest
from mcp_runtime.server import build_server

TOOLSETS = ["my-toolset", "other-toolset"]  # or read [tool.uv.workspace] members


@pytest.mark.parametrize("toolset", TOOLSETS)
def test_toolset_conforms(toolset: str) -> None:
    build_server(toolset)  # raises if the toolset breaks the contract
```

`build_server` binds no port and starts nothing, so this is a fast unit test.
[`mcp-toolsets`](https://github.com/developmentseed/mcp-toolsets) has a
`test_contract.py` doing exactly this over its workspace members.

---

## 3. UI views (rendered by any MCP Apps host)

A view is a pre-built HTML bundle the runtime serves as an MCP resource
`ui://<toolset>/<id>` (MIME `text/html;profile=mcp-app`) and stamps onto the
owning tool's `_meta`. Because it follows the **MCP Apps** standard
(`modelcontextprotocol/ext-apps`), it renders inline in **any MCP Apps host** —
Claude.ai, ChatGPT, Goose, VS Code — with no host-specific work from you: the
runtime already serves the spec MIME and speaks the `ui/*` postMessage bridge.
Views are progressive enhancement — a tool's `message` + structured content
still stand alone in a plain MCP client that can't render them.

For the common case (your server connected to Claude.ai / ChatGPT), you only do
**[3a](#3a-declare--build-the-bundle) +
[3b](#3b-the-view-side-bridge-developmentseedmcp-view)**.
[3c](#3c-only-if-you-also-run-the-bundled-chainlit-agent) is a special case,
needed *only* for the bundled Chainlit agent;
[3d](#3d-rendering-views-in-your-own-frontend) is for when the host is your own
frontend.

### 3a. Declare + build the bundle

Declare `VIEWS = {tool_name: view_id}` and ship a built bundle at
`<package>/views/<view_id>.html`. `build_server` validates the wiring at startup
(unknown tool or missing bundle aborts). How you build the bundle (Vite, etc.) is
your repo's concern — see the `toolsets/*/ui/` setup in
[`mcp-toolsets`](https://github.com/developmentseed/mcp-toolsets).

### 3b. The view-side bridge (`@developmentseed/mcp-view`)

[![npm](https://img.shields.io/npm/v/%40developmentseed%2Fmcp-view?label=npm)](https://www.npmjs.com/package/@developmentseed/mcp-view)

Your bundle talks to whatever host embeds it through the standard `ui/*`
protocol. Import that bridge from the npm package instead of vendoring `host.ts`:

```ts
import { onData, sendMessage } from "@developmentseed/mcp-view";

onData((data) => render(data));       // the tool's structuredContent, from the host
button.onclick = () => sendMessage("run the next thing"); // a user turn back to the chat
```

This is **host-agnostic** — the exact same bundle works in Claude.ai, ChatGPT,
and the Chainlit agent below. It's a public package on npm, so it needs no
registry configuration or auth, in your repo or in CI. From your `ui/` project:

```bash
npm install @developmentseed/mcp-view
```

It shares its version with the Python package, so the two move together.

### 3c. Only if you also run the bundled Chainlit agent

Claude.ai and ChatGPT are MCP Apps hosts already, so they render your views with
nothing beyond [3a](#3a-declare--build-the-bundle) and
[3b](#3b-the-view-side-bridge-developmentseedmcp-view). The bundled Chainlit
chat host (`mcp-agent-web`) is **not** an MCP Apps host out of the box, so the
package ships a host-side element (`McpView.jsx`) that implements the *host* end
of the same `ui/*` bridge. Install it into the Chainlit app root at build time —
do **not** rely on any runtime copy:

```dockerfile
# In your Dockerfile, after `uv sync`, before the runtime image:
RUN mcp-agent install-elements          # writes ./public/elements/McpView.jsx
# or an explicit dir: RUN mcp-agent install-elements path/to/public/elements
```

Deterministic and idempotent — a package upgrade + rebuild refreshes the element,
and nothing writes to the filesystem at runtime (so it works on a read-only root
filesystem). If you launch `mcp-agent-web` without it, the agent still starts but
prints a warning and views won't render. External hosts need none of this.

### 3d. Rendering views in your own frontend

Your own frontend is a host like any other. You have two ways in, and neither
changes your toolsets or their bundles:

- **Embed with a standard MCP Apps client**, which handles the `ui/*` protocol
  for you.
- **Implement the host end yourself** — fetch the `ui://<toolset>/<id>` resource,
  render it in an iframe, and speak the `ui/*` postMessage protocol back.
  `mcp-agent install-elements` writes `McpView.jsx`, which is 89 lines of
  exactly this for Chainlit; read it as the reference implementation even if you
  are not using Chainlit.

Either way you do
**[3a](#3a-declare--build-the-bundle)–[3b](#3b-the-view-side-bridge-developmentseedmcp-view)
and not [3c](#3c-only-if-you-also-run-the-bundled-chainlit-agent)** —
`install-elements` targets the Chainlit app root specifically. If your frontend
also drives its own agent, add
[Session state](#4-session-state-keeping-large-values-out-of-the-model).

---

## 4. Session state (keeping large values out of the model)

Some tool inputs and outputs are too large for a model to be handling: a clip
geometry, an item collection, a raster footprint. `mcp_state` moves such a value
from the tool that produced it to the tool that needs it, through your agent's
state, without it entering the conversation.

All of this is **client-side**, and the asymmetry matters:

- **Any server works, unmodified** — including ones that know nothing about this
  runtime. [Tagging](#4c-tagging-a-tool-optional-and-worth-it) makes it cheaper
  and safer; it is not what makes it work.
- **Only an `mcp_state` client works** — the bundled agent, or your own wired up
  as below. An external MCP host does none of it, so a toolset served to
  Claude.ai or ChatGPT gets no benefit from being tagged.

The full contract — two decision flowcharts and six worked scenarios — is in
[SESSION-STATE.md](./SESSION-STATE.md), with a runnable version in
[`examples/session-state/`](../examples/session-state/).

**The bundled agent has this on already.** `mcp-agent` and `mcp-agent-web` wire
everything in 4b for you, so pointing either at your toolsets is the fastest way
to see tagging pay off. Set `MCP_AGENT_STATE=0` (environment or `.env`) to build
the plain agent instead — every value through the transcript, no capture, no
injection — which is what you want if your host renders tool results straight
off the message, or installs middleware of its own.

It also checkpoints conversations, so both the transcript and `tool_state`
belong to a `thread_id` rather than to the caller:

| `MCP_AGENT_CHECKPOINT` | Store | Use it for |
| --- | --- | --- |
| unset / `memory` | in-process | local dev, demos, a single replica nobody expects to resume |
| a `postgres://` URL | PostgreSQL, via the `[checkpointing-postgres]` extra | anything that restarts, or runs more than one replica |

Embedding it in your own process instead? `build_agent(..., checkpointer=...)`
takes any LangGraph saver and makes no assumptions about it — configure the
connection pool, schema and lifecycle however you already do, and the
environment variable never comes into it. Omit it and each call gets a fresh
in-process saver, which is right for building one agent and wrong for building
several: two agents with separate savers cannot see each other's threads. If
you rebuild agents and expect conversations to survive, hold one `Checkpointing`
(an async context manager) and pass its `saver()` every time.

### 4a. Extending the bundled agent

Before assembling your own (below), check whether the seams on `build_agent`
cover you. A host with its own system prompt, its own local tools, or its own
callbacks does not need to fork anything:

```python
from mcp_state import SESSION_STATE_PROMPT

built = await build_agent(
    url,
    model,
    api_key,
    # Replaces the bundled default. It is used verbatim, so append the
    # session-state fragment yourself when state is on: it is what tells the
    # model how breadcrumbs, @state:<key> handles and host-filled parameters
    # work, and asks it to carry the provenance the state notes record into
    # its answers.
    system_prompt=MY_PROMPT + "\n\n" + SESSION_STATE_PROMPT,
    extra_tools=[load_skill],  # your own tools, added as given
    middleware=[TracingMiddleware()],  # runs after StateCaptureMiddleware
)
```

The bundled default composes exactly this way — `mcp_agent.main.BASE_PROMPT`
plus `SESSION_STATE_PROMPT` — and the plain agent (`MCP_AGENT_STATE=0`) gets
`BASE_PROMPT` alone, so the prompt never describes machinery that is not
wired. Assembling your own agent instead ([4b](#4b-wiring-it-into-your-own-agent))?
The fragment is host-agnostic: append it to your prompt whenever the three
state pieces are installed.

`extra_tools` are yours, not MCP tools: they are neither bound to session state
nor rewritten. `middleware` layers over `StateCaptureMiddleware` rather than
replacing it, so capture and handles keep working.

It returns a `BuiltAgent` — `agent`, `connections`, `tools` (as loaded, before
binding), and `required`, the per-toolset credential-header
declaration discovered alongside the connections. Take `required` from here
rather than looking it up again: a second lookup can disagree with what the
agent was actually wired with.

`run_turn` returns a `TurnResult` — `history`, `new_messages`, `answer`,
`sidecar` (the thread's `tool_state`) and `citations` (ids the model put on
`reference` content blocks). It also takes a `config`, merged into the runnable
config passed to `ainvoke`, for attaching per-turn callbacks or metadata;
`thread_id` always wins over anything set in its `configurable`.

**Streaming the same turn.** `mcp_agent.streaming.stream_turn` takes the same
arguments and is an async generator, for a surface that shows a token before the
turn ends. It yields, in order of arrival:

| event | what it carries |
| --- | --- |
| `AnswerChunk` | `text` — a piece of the answer |
| `ToolStarted` | `id`, `name`, `arguments` the model wrote |
| `ToolFinished` | `id`, `name`, `content`, `artifact`, plus `received` (receipts for what session state supplied) and `published` (`{field: state key}` for what it stored) |
| `StateChanged` | `state` — session state after a tool wrote to it, accumulated |
| `TurnFinished` | `result` — the identical `TurnResult`, always last |

```python
async for event in stream_turn(agent, "clip chirps to my area", thread_id):
    match event:
        case AnswerChunk(text):
            print(text, end="", flush=True)
        case ToolFinished(name=name, received=received) if received:
            print(f"\n{name} was handed {list(received)} from state")
        case TurnFinished(result):
            sources = result.citations
```

A caller wanting one answer can consume the stream and keep only `TurnFinished`
— which is what makes this a superset of `run_turn` rather than a fork of it.

Three things it does that a loop of your own would have to get right: tool
results reach the token channel as well as the update channel, so answer text is
`AIMessageChunk` from the model node and nothing else; receipts ride
`ToolMessage.artifact` rather than content; and an update's `tool_state` names
only what that node wrote, so `StateChanged` carries a running total and the
turn's final state is read back from the checkpointer.

### 4b. Wiring it into your own agent

Only needed if you are *not* using the bundled agent. Needs the `[state]` extra.
Three pieces, all three required:

```python
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp_state import (
    SESSION_STATE_PROMPT,
    StateCaptureMiddleware,
    bind_all_injected,
    make_inspect_state,
    owners,
    publications,
    state_keys,
    with_server_name,
)

# Loaded per server, so each tool records where it came from. The adapter
# takes a `server_name` and stamps it nowhere; without this an undeclared
# capture cannot be keyed `<toolset>/<tool>/<field>` like a declared one.
client = MultiServerMCPClient(connections)
tools = [
    with_server_name(tool, server)
    for server in connections
    for tool in await client.get_tools(server_name=server)
]
published = publications(tools)

agent = create_agent(
    model,
    [*bind_all_injected(tools), make_inspect_state(state_keys(published))],
    system_prompt=MY_PROMPT + "\n\n" + SESSION_STATE_PROMPT,
    middleware=[StateCaptureMiddleware(published, owners=owners(tools))],
)
```

- **`StateCaptureMiddleware`** — moves large values out of tool returns into
  `tool_state`, leaving a `[state updated: …]` breadcrumb in their place. It
  declares `mcp_state.AgentState` as its `state_schema`, so adding it is what
  puts the `tool_state` namespace and its reducer on the graph — you do not
  pass `state_schema` yourself. That reducer bounds the namespace at
  `MAX_TOOL_STATE_BYTES` (8 MB of stored values), evicting the oldest writes;
  nothing else does, and capture writes on every tool call.
- **`bind_all_injected`** — rewrites tool schemas so a model can name a stored
  value, substitutes it at call time, and refuses the calls it cannot serve.
- **`make_inspect_state`** — an `inspect_state` tool, for when the *model* needs
  to read a stored value rather than pass it on. Everything in `tool_state` is
  readable; `state_keys(published)` is passed so a read that misses can say
  "declared, but no tool has published it yet" instead of "no such key".

The fourth piece is soft but do not skip it: append `SESSION_STATE_PROMPT` to
your system prompt. The machinery works without it, but the model then meets
breadcrumbs, `@state:<key>` handles and handle-only parameters with no
explanation — and nothing asks it to carry the provenance into its answers.

**Rendering a tool call in your own host.** A handle *is* in the arguments the
model produced, but only as the `@state:<key>` string — which names a value
without saying what it held or which tool published it, and a reader cannot
expand it.

`receipts_of(message.artifact)` returns `{parameter: Receipt}` for everything
session state supplied — the key it came from and the tool that published it.
The entry that key holds adds one thing more: `inputs`, where each argument of
the call that *produced* it came from, either another key or `"model"`.
`authored(entry)` narrows that to the parameters the model wrote, which is the
part worth showing — "this value rests on something the model chose" is what
decides how much to trust a result. `mcp_agent.host.step_input` is the worked
example; that module holds the host-side helpers and imports no UI framework,
so it is reachable from a base install.

`inputs` is recorded and never enforced. Nothing refuses a call on it, and
`NotAuthored` does not consult it — a value the model wrote, laundered through
a tool that echoed it, still resolves. What changes is that you can see it.

**If your agent has state of its own**, pass a `state_schema` subclassing
`mcp_state.AgentState`. LangChain merges a middleware's schema with the one you
pass, so both sets of channels are present:

```python
class HostState(AgentState):
    run_id: str


agent = create_agent(model, tools, state_schema=HostState, middleware=[...])
```

> **The one thing that bites.** Handle resolution is a LangGraph
> `InjectedState` mechanism, so it runs wherever a tool executes — but
> **capture is agent middleware**. Assemble a bare `StateGraph`/`ToolNode`
> instead of `create_agent` and you get resolution with no capture: nothing is
> ever stored, so there is never anything to name, and no error tells you.

> **Compile with a checkpointer.** `tool_state` lives on graph state, so
> without one every turn starts empty — capture runs, the next turn finds
> nothing to name, and no error says so. With one, state and transcript both belong to the
> `thread_id` and persist for free. `create_agent(..., checkpointer=...)`;
> `InMemorySaver` is enough for local dev, `AsyncPostgresSaver` for anything
> that restarts or scales past one replica.

> **If you also render UI views.** Capture moves the payload off the tool
> message, so rebuild each view's data with
> `restore_structured(message.artifact, tool_state)` rather than reading
> `ToolMessage.artifact` directly. It is a no-op on an uncaptured message, so
> call it unconditionally. `mcp-agent-web` does exactly this — see
> `view_props` in `mcp_agent/host.py`, and "Sharp edges and limits" in
> [SESSION-STATE.md](./SESSION-STATE.md).

Capture is by size (`DEFAULT_CAPTURE_BYTES`, 2 kB) as well as by declaration.
`StateCaptureMiddleware(published, capture_undeclared=None)` turns the size path
off if you want capture strictly as declared.

### 4c. Naming things, and the one tag worth adding

**A `ToolResult` data key is a public name.** Every field but `message` is
captured into session state under `<toolset>/<tool>/<field>`, and that key is
what the *next* toolset's model reads when it decides which stored value a call
should use. Nothing else crosses between two toolsets — no shared vocabulary,
no imports, no registry.

So name for what the value is, not what type it is:

```python
class SearchResult(ToolResult):
    area_of_interest: NotRequired[dict]  # not `geometry`
```

`geometry` is a poor name because a coverage footprint is also a geometry, and
the two are identical JSON. A model handed the wrong one produces confident
nonsense, and nothing in this system will notice.

**`NotAuthored` is the one tag.** It says a model must not write this
parameter's value — nothing about types, nothing about session state:

```python
from typing import Annotated

from langchain_core.tools import tool
from mcp_runtime.declarations import NotAuthored
from mcp_runtime.tool_result import ToolResult


@tool
async def clip_raster(
    dataset_id: str,
    aoi: Annotated[dict, NotAuthored()],
) -> ToolResult: ...
```

Use it where a plausible-looking invention is worse than no answer: a
2000-vertex catchment boundary, an item collection, a bounding box that has to
be *the* one under discussion. A 2000-vertex boundary and four numbers are both
"geometry", and only you know which of them a model could produce.

It degrades in three steps rather than requiring anything:

| client | effect |
| --- | --- |
| ignores `_meta` | the parameter behaves normally; the model fills it |
| reads the description | advisory — the served schema says the value must already exist |
| implements `mcp_state` | the schema accepts only `@state:<key>` |

Tagging something that is not a parameter fails at `build_server` rather than
going unnoticed until a client connects. Toolsets advertise what they publish
and what they will not author on `/health` (`state.produces`,
`state.not_authored`), and the index aggregates them, so you can see a
deployment's data flow without speaking MCP.

---

## 5. Serving the agent over HTTP (`mcp_agent_api`)

Needs the `[api]` extra, which is `[agent]` plus `ag-ui-protocol` — and not
`[web]`, so an API deployment installs no UI framework.

Your frontend, this runtime's agent, and
[AG-UI](https://github.com/ag-ui-protocol/ag-ui) over Server-Sent Events between
them. Three layers, and you enter at the one you already have an application at:

| Module | What it is | What you supply |
| --- | --- | --- |
| `mcp_agent_api.app` | a `FastAPI`, agent connected in its lifespan | nothing, or a build factory |
| `mcp_agent_api.routes` | an `APIRouter` over an already-built agent | your app, lifespan and middleware |
| `mcp_agent_api.events` | one turn as AG-UI events | your transport |

Each is the one below it plus a decision — `create_app` calls `create_router`,
which calls `agui_events` — so entering at the top costs nothing you cannot undo
by dropping a layer later.

A runnable version of all three, with a React client on `@ag-ui/client` and a
backend laid out the way a deployment is, is in
[`examples/agui-events/`](../examples/agui-events/).

**This one holds the provider key.** The Chainlit host is bring-your-own-model
because its users have a settings dialog to type into; an API has no dialog and
its client is yours, so `PROVIDER_MODEL` and `PROVIDER_API_KEY` are read from the
environment at startup and belong to the deployment.

### 5a. The whole service

```bash
uv run uvicorn mcp_agent_api.app:app --port 8000
```

`MCP_URL`, `PROVIDER_MODEL` and `PROVIDER_API_KEY` come from the environment or
a `.env`; `MCP_AGENT_STATE` and `MCP_AGENT_CHECKPOINT` mean here exactly what
they mean in [4](#4-session-state-keeping-large-values-out-of-the-model), so an
API deployment gets session state and resumable threads on the same terms as the
bundled host. `MCP_AGENT_API_CORS_ORIGINS` is the one setting the API adds,
comma-separated
(`https://a,https://b`) rather than JSON because that is what someone writes in a
Helm values file. Unset — a UI on the same origin, or reached through a dev
server's proxy — adds no CORS middleware at all.

Building the agent yourself instead (extra tools, your own system prompt, a
checkpointer you already own) is a factory:

```python
from mcp_agent.main import build_agent
from mcp_agent_api.app import create_app


async def build():
    return await build_agent(url, model, key, extra_tools=[load_skill])


app = create_app(build)
```

**The factory is async and awaited during startup**, not called by `create_app`:
connecting to MCP servers needs a running loop and there is none at import. Until
it returns the routes answer `503`, and a failure inside it stops the process —
a container that starts, reports healthy and answers 503 forever is harder to
notice than one that refuses to start.

That shapes what you report to an orchestrator. Readiness should reflect the
agent being built; liveness deliberately should not, or a restart lands on a
process that was merely still connecting. `create_app` hands back a `FastAPI` and
it is still yours, so both are yours to add:

```python
app = create_app(build)


@app.get("/health/readiness")
async def readiness():
    if getattr(app.state, "built", None) is None:
        raise HTTPException(503, "the agent is still connecting")
    return {"status": "ready"}
```

### 5b. Mounting the routes into your own application

A deployment with its own FastAPI application — its own auth, its own
middleware, its own everything — takes the router and keeps all of it:

```python
from mcp_agent_api.routes import create_router

app.include_router(create_router(lambda: app.state.built, prefix="/agent"))
```

**The agent arrives through a callable**, called per request rather than passed
once, for the same reason `create_app` takes a factory: the router is built and
mounted before anything has connected. Raising from it is how you say "not yet",
and that surfaces as `503`. An agent rebuilt behind it — on a model change, on a
reconnect — is picked up without remounting.

It may return anything with `.agent`, `.connections`, `.tools` and
`.required`, which `build_agent`'s `BuiltAgent` already is. Those are read by
attribute rather than unpacked because `BuiltAgent` is a `NamedTuple` and a
consumer's equivalent may order its fields differently — positional unpacking
would silently pair the wrong ones.

**`turn_context` is where what wraps a run goes** — tracing callbacks, a
correlation id, per-request metadata. A context manager, entered inside the
stream and given the request and both ids; whatever it yields becomes the
turn's runnable config:

```python
@contextmanager
def traced(request, thread_id, run_id):
    trace = uuid.uuid4().hex
    with correlation_id(trace):  # a ContextVar your httpx hook reads
        yield {"callbacks": [handler(trace)], "metadata": {"thread": thread_id}}


app.include_router(create_router(provider, turn_context=traced))
```

A context manager rather than a config factory because the two things a host
wants here differ in kind: a config is a *value* handed to the turn, while a
correlation id stamped onto outgoing MCP calls is a **context variable**, which
has to be set for the duration. Both need to be in force *while the turn runs*,
not while the handler is on the stack — by the time the first tool is called,
the handler has long returned. That is why this is entered beside
`user_credentials` rather than around the route. `create_app` takes the same
argument and passes it straight through.

**The AG-UI types come from AG-UI.** `messages` on `POST /runs`, and the
transcript `GET /threads/{id}` hands back, are `ag_ui.core.Message` — the
protocol's own discriminated union, which includes the `activity` role this
server emits, so a client echoing its history back validates. The stream is
documented as `ag_ui.core.Event`: all 33 event types, discriminated on `type`,
under `text/event-stream` rather than a nominal `application/json`. Nothing is
re-described by hand, so none of it can drift from the protocol.

One consequence worth knowing: the protocol requires an `id` on every message,
so a client that omits one now gets a `422`. Any string does — a fresh uuid per
message is the obvious choice — and nothing here reads it: history is the
server's, and the id you post is discarded rather than stored.

**The read routes document their own shapes.** `ThreadResponse`, `TurnsResponse`,
`StateValueResponse` and the `StateEntryInfo` all three share are exported from
`mcp_agent_api`, so a Python client validates against them rather than
re-declaring them, and the generated OpenAPI carries them instead of a bare
`object`. They are attached through FastAPI's `responses=` rather than
`response_model=`, deliberately: a response model would re-serialise, and
`seq` is *omitted* from a state entry until it is known — a client sorting by
it must never be sorting nulls — while `tool: null` is meaningful and has to
stay. Documenting without re-serialising keeps both.

### 5c. The routes

| | |
| --- | --- |
| `POST /runs` | one turn, streamed as AG-UI SSE — the whole conversation is here |
| `GET /threads/{id}` | the thread's messages and activities, so a page reload restores it |
| `GET /threads/{id}/turns` | its turns, and what session state held at the end of each |
| `GET /threads/{id}/state/{key}` | one session-state value in full; `?turn=N` for the value as of then |
| `GET /views/{toolset}/{view}` | the HTML for a `ui://` bundle a tool declared |

`POST /runs` accepts a `RunAgentInput` from `@ag-ui/client` as posted, and also a
hand-written `{"threadId": …, "messages": […]}`: the fields that model requires
but this endpoint never reads are optional here. A client that omits `threadId`
learns the one it was given from `RUN_STARTED`, which is always the first event.

**History is the server's.** Only the trailing user message of the request is
read; the rest of `messages` is ignored and the checkpointer's transcript is the
truth. That diverges from AG-UI's client-is-authoritative convention on purpose —
the values session state exists to keep out of the model's context would
otherwise have to live in the browser and be posted back every turn. The stream
says so rather than leaving a client to discover it: a `MESSAGES_SNAPSHOT`
closes every run with the thread as the server holds it. It closes rather than
opens the run because a snapshot drops every local message it does not name, and
one sent before the turn was checkpointed would take the user's own question off
their screen. Ids line up — the question keeps the client's `id`, the answer
carries the id the thread will store — so a client reconciles in place rather
than rebuilding its list.

**The read routes are what the stream deliberately leaves out.** The state
channel carries `{tool, bytes, inputs}` per key and never the payload, so a client that
has decided it wants the 38 kB geometry comes to `/state/{key}` for it. The key
is qualified by its publishing toolset and tool
(`dataset-search/search_datasets/area_of_interest`) and those slashes are part
of the key, not path separators.

**Per-turn state needs no new storage.** The state channel is cumulative — the
patches take a client to every key the thread holds — so neither "which keys did *this*
turn add" nor "what did this turn run on" is answerable from the stream. Both
are answerable from what LangGraph already keeps: an immutable checkpoint per
super-step, every past value retained. `/turns` and `?turn=N` read that back.
`/turns` reports `turns` (retained) alongside `total` (asked), because a missing
turn has two meanings and a client cannot otherwise tell them apart: one the
thread never had is a `404`, and one the checkpointer has since pruned is a
`410`.

**A thread id is the only credential on a state read.** `/threads/{id}/state/{key}`
returns whatever that thread published, to whoever names the thread. That is what
lets an anonymous conversation draw its own map, and it means the id must be
treated as a secret — it leaks through logs, referrers and browser history like
any other URL component. If a thread belongs to a user, put your own dependency
in front of the router ([5b](#5b-mounting-the-routes-into-your-own-application)).

### 5d. The event stream

`mcp_agent_api.events.agui_events` maps one turn onto AG-UI. It imports no
FastAPI and opens no socket, so a consumer with its own transport can use it
directly:

```python
from ag_ui.encoder import EventEncoder
from mcp_agent.streaming import stream_turn
from mcp_agent_api.events import agui_events

encoder = EventEncoder()
turn = stream_turn(built.agent, question, thread_id)
async for event in agui_events(
    turn,
    thread_id=thread_id,
    run_id=run_id,
    tools={tool.name: tool for tool in built.tools},
):
    yield encoder.encode(event)  # already `data: {...}\n\n`
```

AG-UI covers tokens, tool calls and state natively. It has no vocabulary for the
things this runtime exists to make visible, so those ride `ACTIVITY_*` — an
activity *is* a message in AG-UI, so a client rendering messages in order shows
them in the right place with no correlation code:

| `activityType` | content |
| --- | --- |
| `state.consumed` | `toolCallId`, `tool`, `received` — each receipt's fields plus a `display` line |
| `state.published` | `toolCallId`, `tool`, `published` |
| `mcp.view` | `toolCallId`, `tool`, `uri` — the `ui://` bundle, fetched separately and cached |
| `answer.citations` | `ids` — after the answer, since that is where they belong |

Every activity carries a `display` string as well as its fields. A minimal client
prints it; a bespoke one styles the fields and ignores it. For receipts that
string is `mcp_agent.host.step_input`'s output, so the wire says exactly what the
bundled Chainlit host shows:

```
@state:dataset-search/search_datasets/area_of_interest · 1 feature(s), 4 vertices · from search_datasets · query written by the model
```

**Read the fields, never that string.** A receipt is `{key, tool}` — the state
key the value came from, and the tool that published it. The trailing clause
appears only when the producing call was itself given a model-authored
argument, and it names the parameter rather than repeating its value.

`state.published` carries `{toolCallId, tool, published: {field: key}}`, which is
enough to link a key in a state panel back to the call that wrote it without any
bookkeeping of your own — the example UI's cross-highlighting is that mapping and
nothing else.

`STATE_DELTA` carries metadata only — `tool`, `bytes`, `inputs`, and `seq`
once known — never the stored value, which a frontend fetches when it actually wants
to draw it. It sits under `toolState` inside AG-UI's state object.

**Patched rather than snapshotted, and that is the point.** A `STATE_SNAPSHOT`
replaces the whole `state` object, and that object is shared: everything the
client keeps in it would go every time a tool wrote. Every operation sent here
names a path under `toolState` instead, so the client's own keys are never
touched. Each run opens with one `add` of the whole namespace — the
resynchronisation point, carrying the thread's state and not merely this turn's
writes — and what follows is one `add` or `remove` per key that moved. State
that has not moved sends no event at all.

Two details a client meets. **Escaping**: a state key is `toolset/name` and `/`
is JSON Pointer's own separator, so `gazet/candidates` arrives as
`/toolState/gazet~1candidates` (RFC 6901, `~` as `~0`). **`seq`**: assigned when
a write is merged, so mid-turn entries omit it and the delta closing the turn
adds it.

Four protocol rules shape the event order, each checked against `@ag-ui/client`'s
own verifier rather than read off the specification: nothing may precede
`RUN_STARTED`; **a message's position is fixed when it is created**, so
everything belonging to a tool call is emitted before the answer's text message
opens; activity deltas fail silently, so only snapshots are sent; and activity
content is always a JSON object, never a bare list.

### 5e. Credentials

**From the environment.** `resolve_credentials(required, flags, dotenv_extra)`
resolves each header a connected toolset advertises: an explicit flag first, then
`X_DEMO_TOKEN` for `x-demo-token` in the process environment, then the same key
from a `.env`. The CLI exposes it as repeatable `--header NAME=VALUE`, and the
web host uses it to skip asking for anything the deployment can already supply.

**From the request.** `POST /runs` calls `credentials_for(request.headers,
built.required)`, which takes off the request only those headers a connected
toolset actually declared, and falls back to the environment for the rest.
Forwarding everything would put this request's `Authorization` — which belongs to
*your* API, not to a toolset — onto an outbound MCP call, and the MCP
authorization spec is explicit that a server must not transit a token addressed
to somebody else.

A deployment that sets the environment fallback gives every caller that account.
That is right for local development and for a single shared service credential,
and wrong for per-user access: such a deployment leaves the variable unset and
lets each request carry its own header.

---

## 6. Migrating off the in-repo workspace

If your repo currently vendors `packages/mcp-runtime`, `packages/mcp-cli`,
`packages/mcp-agent`:

1. Delete those three directories.
2. Drop them from root `pyproject.toml` `dependencies` and `[tool.uv.sources]`;
   add the `mcp-toolsets-runtime` dependency from [Install](#1-install). It
   needs no `[tool.uv.sources]` entry of its own — it resolves from PyPI.
3. Remove `packages/*` from `[tool.uv.workspace] members` if nothing else lives
   there.
4. In each toolset `ui/`, delete the vendored `src/host.ts` and depend on
   `@developmentseed/mcp-view`
   ([3b](#3b-the-view-side-bridge-developmentseedmcp-view)). Delete the repo-root
   `public/elements/McpView.jsx` — it now comes from `mcp-agent install-elements`
   ([3c](#3c-only-if-you-also-run-the-bundled-chainlit-agent)).
5. `uv lock`, run your lint/tests, and smoke-test a toolset server + the web host.

Imports don't change, so application code is untouched — this is a dependency and
build-wiring change only.
