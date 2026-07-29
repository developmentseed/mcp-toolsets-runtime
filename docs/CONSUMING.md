# Consuming `mcp-toolsets-runtime`

This is what a repo that installs the runtime has to do. There are three
personas — most repos are the first, some are also one of the others:

1. **Serving tools** — you have toolsets and want to expose them as MCP servers.
   You need `mcp_runtime` and the plugin contract. That's it.
2. **Running the web agent with UI views** — you also run the bundled Chainlit
   chat host and want tool results to render as views in its side panel. You
   need the `[agent]` extra, the host element installed at build time, and (if
   your views are custom-built) the `@developmentseed/mcp-view` npm bridge.
3. **Running your own agent** — you drive MCP tools from your own LangGraph
   agent rather than the bundled chat host. You need the `[state]` extra to keep
   large tool values out of the model's context (see
   [Session state](#4-session-state-keeping-large-values-out-of-the-model)).

The web host (`mcp-agent-web`) is **bring-your-own-model**: it holds no provider
key. Each user sets a `provider:model` + their API key in the chat's ⚙ settings
(env `PROVIDER_MODEL` / `PROVIDER_API_KEY` only *pre-fill* for local use), so a
hosted deployment stores no secret. The `[agent]` extra stays provider-agnostic —
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
    # "mcp-toolsets-runtime[agent]", # add [agent] if you run the web host
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

Available console scripts: `mcp-serve`, `mcp-index` (base); `mcp-cli` (base);
`mcp-agent`, `mcp-agent-web` (need `[agent]`).

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

A tool may also tag a value with the `Kind` it is — see
[Session state](#4-session-state-keeping-large-values-out-of-the-model). That
tag is **optional**: an agent keeps large values out of the model's context
whether or not you use it.

Serve it:

```bash
TOOLSET=my-toolset mcp-serve         # serves this toolset's tools over MCP
```

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

A per-toolset conformance sweep (walk `toolsets/`, import each, assert the
contract) belongs in **your** repo — it isn't shipped by the runtime. Copy the
`test_contract.py` pattern from
[`mcp-toolsets`](https://github.com/developmentseed/mcp-toolsets) if you want it.

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
**[3a](#3a-declare--build-the-bundle)** +
**[3b](#3b-the-view-side-bridge-developmentseedmcp-view)**.
[3c](#3c-only-if-you-also-run-the-bundled-chainlit-agent) is a special case,
needed *only* for the bundled Chainlit agent.

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

> A future first-party React app is just another host: it implements the host end
> of the same `ui/*` bridge (as `McpView.jsx` does), or embeds views with a
> standard MCP Apps client — no change to your toolsets or their bundles.

---

## 4. Session state (keeping large values out of the model)

Some tool inputs and outputs are too large for a model to be handling: a clip
geometry, an item collection, a raster footprint. `mcp_state` moves such a value
from the tool that produced it to the tool that needs it, through your agent's
state, without it entering the conversation.

**You do not have to change your tools for this.** It works against any MCP
server, including ones that know nothing about this runtime.
[Tagging](#4b-tagging-a-tool-optional-and-worth-it) makes it cheaper and safer;
it is not what makes it work.

The full contract — two decision flowcharts and six worked scenarios — is in
[SESSION-STATE.md](./SESSION-STATE.md), with a runnable version in
[`examples/session-state/`](../examples/session-state/).

### 4a. Wiring it into your own agent

Needs the `[state]` extra. Four pieces, all four required:

```python
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp_state import (
    AgentState,
    StateCaptureMiddleware,
    bind_all_injected,
    make_inspect_state,
    partition_usable,
    publications,
    state_keys,
)

tools = await MultiServerMCPClient(connections).get_tools()
published = publications(tools)

# Drop tools that can never be called, and say which and why.
agent_tools, withheld = partition_usable(bind_all_injected(tools))
for item in withheld:
    log.warning("withholding %s", item)

agent = create_agent(
    model,
    [*agent_tools, make_inspect_state(state_keys(published))],
    state_schema=AgentState,
    middleware=[StateCaptureMiddleware(published)],
)
```

- **`state_schema=AgentState`** — adds the `tool_state` namespace and its
  reducer. Subclass it if your agent has state of its own.
- **`StateCaptureMiddleware`** — moves large values out of tool returns into
  `tool_state`, leaving a `[state updated: …]` breadcrumb in their place.
- **`bind_all_injected`** — rewrites tool schemas so a stored value can reach a
  parameter, and fills it at call time.
- **`make_inspect_state`** — an `inspect_state` tool, for when the *model* needs
  to read a stored value rather than pass it on.

> **The one thing that bites.** Injection is a LangGraph `InjectedState`
> mechanism, so it runs wherever a tool executes — but **capture is agent
> middleware**. Assemble a bare `StateGraph`/`ToolNode` instead of
> `create_agent` and you get injection with no capture: nothing is ever stored,
> so nothing is ever injected, and there is no error to tell you.

Capture is by size (`DEFAULT_CAPTURE_BYTES`, 2 kB) as well as by declaration.
`StateCaptureMiddleware(published, capture_undeclared=None)` turns the size path
off if you want capture strictly as declared.

### 4b. Tagging a tool (optional, and worth it)

Tag a value with the `Kind` it is — on a `ToolResult` data key to say what the
tool publishes, on a parameter to say what it takes:

```python
from typing import Annotated, NotRequired

from mcp_runtime.declarations import Kind
from mcp_runtime.kinds import GEOJSON_AREA_OF_INTEREST
from mcp_runtime.tool_result import ToolResult


class SearchResult(ToolResult):
    geometry: NotRequired[Annotated[dict, Kind(GEOJSON_AREA_OF_INTEREST)]]


@tool
async def clip_raster(
    dataset_id: str,
    aoi: Annotated[dict, Kind(GEOJSON_AREA_OF_INTEREST)],
) -> ToolResult: ...
```

A kind names *what a value is* and nothing else. Matching is by kind, so the
producing and consuming toolsets can live in different repos on different
servers and neither names the other — the string is the entire contract, which
is why kinds live in `mcp_runtime.kinds` and are added by PR.

Without the tag, the model points a parameter at a stored value by name
(`@state:<key>`) — about ten tokens. With it, the parameter is **removed from
the model's schema entirely** and filled by the client: no tokens, no turn spent
choosing, and no way for the model to get it wrong or inline a bad value. You
also get the wiring checked at connect and typos caught at `build_server`.

`Kind` takes one option, for the judgement only you can make:

```python
aoi: Annotated[dict, Kind(GEOJSON_AREA_OF_INTEREST, model_generatable=False)]
```

A 2000-vertex catchment boundary and a four-number bounding box are both
"geometry"; only the tool author knows which a model could plausibly produce.
It defaults to `True`, so a parameter whose kind nothing publishes stays visible
to the model and the tool keeps working. Set it `False` and the tool is withheld
instead — but only do that when the value's producer is one of *your* toolsets,
since the connect-time check reads declarations and cannot see a third-party
server's output coming.

Toolsets advertise both halves on `/health` (`state.produces`, `state.consumes`)
and the index aggregates them, so you can see a deployment's data flow without
speaking MCP.

---

## 5. Migrating off the in-repo workspace

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
