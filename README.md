# mcp-toolsets-runtime

The shared runtime for [MCP Toolsets](https://github.com/developmentseed/mcp-toolsets).
Both `developmentseed/mcp-toolsets` and downstream repos generated from it
install this package instead of each carrying their own copy of the runtime.

## What's in here

One Python distribution (`mcp-toolsets-runtime`) exposing three top-level
modules, plus the view-side JS bridge:

| Module | What it is |
| --- | --- |
| `mcp_runtime` | Discovers a toolset's LangChain tools (`TOOLS`) and serves them as an MCP server; serves UI views (`VIEWS`) as `ui://` resources; derives server `instructions` from `CREDENTIAL_HEADERS`. Entry points: `mcp-serve`, `mcp-index`. |
| `mcp_cli` | Typer CLI to list and call tools on a running MCP service. Entry point: `mcp-cli`. |
| `mcp_toolset` | Scaffolds a new toolset in a consumer repo (`mcp-toolset new [--with-ui] <name>`), wired to this package + the npm view bridge. |
| `mcp_agent` | Example Chainlit chat agent that discovers MCP servers behind an index URL and drives their tools. Ships the Chainlit host element `elements/McpView.jsx`. Entry points: `mcp-agent`, `mcp-agent-web`. Requires the `[agent]` extra. |
| `@developmentseed/mcp-view` (`js/mcp-view`) | The view-side `ui/*` postMessage bridge a toolset UI imports (`onData` / `sendMessage`). Published to npm separately. |

### The toolset plugin contract

`mcp_runtime` discovers a toolset purely by convention — a `<toolset>.tools`
module exporting:

- `TOOLS` — a non-empty list of LangChain tools that return a `ToolResult`.
- `VIEWS` *(optional)* — `{tool_name: view_id}`, with a built bundle at
  `<package>/views/<view_id>.html`.
- `CREDENTIAL_HEADERS` *(optional)* — header names the tools read off the
  transport; used to derive the model-facing auth hint.

Treat these, `ToolResult`, and the `ui/*` wire protocol as **public API**.

## Install

```bash
# base: runtime + cli (lean, for tool-serving images)
pip install "mcp-toolsets-runtime @ git+https://github.com/developmentseed/mcp-toolsets-runtime.git@v0.1.0"

# with the Chainlit web agent
pip install "mcp-toolsets-runtime[agent] @ git+https://github.com/developmentseed/mcp-toolsets-runtime.git@v0.1.0"
```

With uv, as a consumer:

```toml
dependencies = ["mcp-toolsets-runtime[agent]"]

[tool.uv.sources]
mcp-toolsets-runtime = { git = "https://github.com/developmentseed/mcp-toolsets-runtime.git", tag = "v0.1.0" }
```

Imports are unchanged from the old workspace packages: `from mcp_runtime.server
import build_server`, etc. Bump the runtime by changing the `tag` and running
`uv lock`.

**Consuming this package** — the plugin contract, serving toolsets, wiring up UI
views (including `mcp-agent install-elements` and the npm bridge), and migrating
off the in-repo workspace: see **[docs/CONSUMING.md](./docs/CONSUMING.md)**.

## Develop

```bash
uv sync --all-extras   # install with [agent] + dev tools
./scripts/lint         # ruff check + ruff format --check + mypy (config in pyproject)
./scripts/test         # pytest
./scripts/build-js     # typecheck + build + vitest for js/mcp-view (needs node)
```

## Releases

Versioning and `CHANGELOG.md` are managed by
[release-please](https://github.com/googleapis/release-please) from Conventional
Commits. See [CONTRIBUTING.md](./CONTRIBUTING.md) — in short, **your PR title is
the changelog entry**, and CI fails a PR whose title isn't a valid conventional
commit. The Python package and the JS bridge share one version (linked).
