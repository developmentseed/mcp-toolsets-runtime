# Consuming `mcp-toolsets-runtime`

This is what a repo that installs the runtime has to do. There are two personas —
most repos are the first, some are also the second:

1. **Serving tools** — you have toolsets and want to expose them as MCP servers.
   You need `mcp_runtime` and the plugin contract. That's it.
2. **Running the web agent with UI views** — you also run the bundled Chainlit
   chat host and want tool results to render as inline views. You need the
   `[agent]` extra, the host element installed at build time, and (if your views
   are custom-built) the `@developmentseed/mcp-view` npm bridge.

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

Pin a tag; `uv.lock` records the exact commit, so upgrades are a one-line tag bump.

```toml
# pyproject.toml
dependencies = [
    "mcp-toolsets-runtime",          # base: mcp_runtime + mcp_cli
    # "mcp-toolsets-runtime[agent]", # add [agent] if you run the web host
]

[tool.uv.sources]
mcp-toolsets-runtime = { git = "https://github.com/developmentseed/mcp-toolsets-runtime.git", tag = "v0.1.0" }
```

```bash
uv lock && uv sync
```

Imports are unchanged from the old in-repo workspace packages — `from
mcp_runtime.server import build_server`, `from mcp_agent.main import ...`, etc.
**Upgrade** by changing the `tag` and running `uv lock`.

Available console scripts: `mcp-serve`, `mcp-index` (base); `mcp-cli` (base);
`mcp-agent`, `mcp-agent-web` (need `[agent]`).

---

## 2. Author a toolset (the plugin contract)

`mcp_runtime` discovers a toolset by convention. Given `TOOLSET=my-toolset`, it
imports `my_toolset.tools` and reads module-level exports:

```python
# my_toolset/tools/__init__.py
from langchain_core.tools import tool
from mcp_runtime.tool_result import ToolResult


@tool
def search(query: str) -> ToolResult:
    """One-line docstring — becomes the tool's MCP description (required)."""
    return ToolResult(message=f"results for {query}", ...)


TOOLS = [search]                      # required: non-empty list of tools
# VIEWS = {"search": "gallery"}       # optional: tool_name -> view_id (see §3)
# CREDENTIAL_HEADERS = ["X-My-Token"] # optional: headers the tools read off the transport
```

- **`TOOLS`** — every tool must return a `ToolResult` (its annotations become the
  MCP output schema; `build_server` rejects tools that don't at startup).
- **`CREDENTIAL_HEADERS`** — names of headers the tools read off the transport.
  The runtime derives the server `instructions` from these so the model is told
  a credential rides the connection and it shouldn't ask the user for it. The
  credential never enters the model context.
- **`VIEWS`** — see §3.

Serve it:

```bash
TOOLSET=my-toolset mcp-serve         # serves this toolset's tools over MCP
```

A per-toolset conformance sweep (walk `toolsets/`, import each, assert the
contract) belongs in **your** repo — it isn't shipped by the runtime. Copy the
`test_contract.py` pattern from
[`mcp-toolsets`](https://github.com/developmentseed/mcp-toolsets) if you want it.

---

## 3. UI views (only if you serve views through the Chainlit host)

A view is a pre-built HTML bundle served as an MCP resource `ui://<toolset>/<id>`
and rendered inline. Wiring it up has three parts:

### 3a. Declare + build the bundle

Declare `VIEWS = {tool_name: view_id}` and ship a built bundle at
`<package>/views/<view_id>.html`. `build_server` validates the wiring at startup
(unknown tool or missing bundle aborts). How you build the bundle (Vite, etc.) is
your repo's concern — see the `toolsets/*/ui/` setup in
[`mcp-toolsets`](https://github.com/developmentseed/mcp-toolsets).

### 3b. Install the Chainlit host element at build time

The web host renders views through a Chainlit `CustomElement` named `McpView`,
which Chainlit loads from `<app-root>/public/elements/`. The element ships **with
the package**; install it into place as an explicit build step — do **not** rely
on any runtime copy:

```dockerfile
# In your Dockerfile, after `uv sync`, before the runtime image:
RUN mcp-agent install-elements          # writes ./public/elements/McpView.jsx
# or target an explicit dir: RUN mcp-agent install-elements path/to/public/elements
```

This is deterministic and idempotent — a package upgrade + rebuild refreshes the
element, and nothing writes to the filesystem at runtime (so it works on a
read-only root filesystem). If you launch `mcp-agent-web` without having run it,
the agent still starts but prints a warning and views won't render.

### 3c. The view-side bridge (`@developmentseed/mcp-view`)

If you build your own view bundles, import the `ui/*` postMessage bridge from the
npm package instead of vendoring `host.ts`:

```ts
import { onData, sendMessage } from "@developmentseed/mcp-view";

onData((data) => render(data));       // the tool's structuredContent
button.onclick = () => sendMessage("do the next thing");
```

It's published to **GitHub Packages**, so the consuming UI needs an `.npmrc`:

```
# js .npmrc (repo root or the ui/ dir)
@developmentseed:registry=https://npm.pkg.github.com
//npm.pkg.github.com/:_authToken=${GITHUB_TOKEN}
```

```jsonc
// ui/package.json
"dependencies": { "@developmentseed/mcp-view": "^0.1.0" }
```

In CI, set `GITHUB_TOKEN` (the default `secrets.GITHUB_TOKEN` has `read:packages`
against repos in the same org).

---

## 4. Migrating off the in-repo workspace

If your repo currently vendors `packages/mcp-runtime`, `packages/mcp-cli`,
`packages/mcp-agent`:

1. Delete those three directories.
2. Drop them from root `pyproject.toml` `dependencies` and `[tool.uv.sources]`;
   add the `mcp-toolsets-runtime` dependency + source from §1.
3. Remove `packages/*` from `[tool.uv.workspace] members` if nothing else lives
   there.
4. In each toolset `ui/`, delete the vendored `src/host.ts` and depend on
   `@developmentseed/mcp-view` (§3c). Delete the repo-root
   `public/elements/McpView.jsx` — it now comes from `mcp-agent install-elements`.
5. `uv lock`, run your lint/tests, and smoke-test a toolset server + the web host.

Imports don't change, so application code is untouched — this is a dependency and
build-wiring change only.
```
