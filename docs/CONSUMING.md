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
**§3a + §3b**. §3c is a special case, needed *only* for the bundled Chainlit agent.

### 3a. Declare + build the bundle

Declare `VIEWS = {tool_name: view_id}` and ship a built bundle at
`<package>/views/<view_id>.html`. `build_server` validates the wiring at startup
(unknown tool or missing bundle aborts). How you build the bundle (Vite, etc.) is
your repo's concern — see the `toolsets/*/ui/` setup in
[`mcp-toolsets`](https://github.com/developmentseed/mcp-toolsets).

### 3b. The view-side bridge (`@developmentseed/mcp-view`)

Your bundle talks to whatever host embeds it through the standard `ui/*`
protocol. Import that bridge from the npm package instead of vendoring `host.ts`:

```ts
import { onData, sendMessage } from "@developmentseed/mcp-view";

onData((data) => render(data));       // the tool's structuredContent, from the host
button.onclick = () => sendMessage("run the next thing"); // a user turn back to the chat
```

This is **host-agnostic** — the exact same bundle works in Claude.ai, ChatGPT,
and the Chainlit agent below. It's published to **GitHub Packages**, so the
consuming UI needs an `.npmrc`:

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

### 3c. Only if you also run the bundled Chainlit agent

Claude.ai and ChatGPT are MCP Apps hosts already, so they render your views with
nothing beyond §3a–§3b. The bundled Chainlit chat host (`mcp-agent-web`) is
**not** an MCP Apps host out of the box, so the package ships a host-side element
(`McpView.jsx`) that implements the *host* end of the same `ui/*` bridge. Install
it into the Chainlit app root at build time — do **not** rely on any runtime copy:

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

## 4. Migrating off the in-repo workspace

If your repo currently vendors `packages/mcp-runtime`, `packages/mcp-cli`,
`packages/mcp-agent`:

1. Delete those three directories.
2. Drop them from root `pyproject.toml` `dependencies` and `[tool.uv.sources]`;
   add the `mcp-toolsets-runtime` dependency + source from §1.
3. Remove `packages/*` from `[tool.uv.workspace] members` if nothing else lives
   there.
4. In each toolset `ui/`, delete the vendored `src/host.ts` and depend on
   `@developmentseed/mcp-view` (§3b). Delete the repo-root
   `public/elements/McpView.jsx` — it now comes from `mcp-agent install-elements`
   (§3c).
5. `uv lock`, run your lint/tests, and smoke-test a toolset server + the web host.

Imports don't change, so application code is untouched — this is a dependency and
build-wiring change only.
```
