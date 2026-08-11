# mcp-toolsets-runtime

[![PyPI](https://img.shields.io/pypi/v/mcp-toolsets-runtime?label=PyPI)](https://pypi.org/project/mcp-toolsets-runtime/)
[![npm](https://img.shields.io/npm/v/%40developmentseed%2Fmcp-view?label=npm)](https://www.npmjs.com/package/@developmentseed/mcp-view)

The shared runtime for [MCP Toolsets](https://github.com/developmentseed/mcp-toolsets).
Both `developmentseed/mcp-toolsets` and downstream repos generated from it
install this package instead of each carrying their own copy of the runtime.

## What's in here

One Python distribution (`mcp-toolsets-runtime`) exposing five top-level
modules, plus the view-side JS bridge:

| Module | What it is |
| --- | --- |
| `mcp_runtime` | Discovers a toolset's LangChain tools (`TOOLS`) and serves them as an MCP server; serves UI views (`VIEWS`) as `ui://` resources; derives server `instructions` from `CREDENTIAL_HEADERS`; advertises what each tool publishes into and takes from session state (`Kind`). Entry points: `mcp-serve` (one toolset), `mcp-serve-local` (several at once, for local dev), `mcp-index`. |
| `mcp_state` | Session state for *any* agent driving MCP tools: the `tool_state` namespace, `StateCaptureMiddleware` (moves large payloads out of the transcript), `inspect_state` (the model reads one on demand), and `bind_injected` (fills declared parameters from state, and offers `@state:<key>` handles on the rest). A filled parameter leaves a receipt, so a value the model never saw can still be traced to the tool that published it. Works against unmodified third-party servers. Requires the `[state]` extra. |
| `mcp_cli` | Typer CLI to list and call tools on a running MCP service. Entry point: `mcp-cli`. |
| `mcp_toolset` | Scaffolds a new toolset in a consumer repo (`mcp-toolset new [--with-ui] <name>`), wired to this package + the npm view bridge. |
| `mcp_agent` | Example Chainlit chat agent that discovers MCP servers behind an index URL and drives their tools, with `mcp_state` wired in (`MCP_AGENT_STATE=0` to opt out). Conversations are checkpointed per `thread_id` — in-process by default, PostgreSQL via `MCP_AGENT_CHECKPOINT` + the `[checkpointing-postgres]` extra. Ships the Chainlit host element `elements/McpView.jsx`. Entry points: `mcp-agent`, `mcp-agent-web`. `mcp_agent.main` (`build_agent`, `run_turn`) and `mcp_agent.host` — the UI-framework-free helpers a host of its own needs (view bundles and props, and the tool-step arguments session state filled in) — need the `[agent]` extra. `mcp_agent.web`, the Chainlit host, needs `[web]` on top. |
| `@developmentseed/mcp-view` (`js/mcp-view`) | The view-side `ui/*` postMessage bridge a toolset UI imports (`onData` / `sendMessage`). Published to npm separately. |

### The toolset plugin contract

`mcp_runtime` discovers a toolset purely by convention — a `<toolset>.tools`
module exporting:

- `TOOLS` — a non-empty list of LangChain tools that return a `ToolResult`.
- `VIEWS` *(optional)* — `{tool_name: view_id}`, with a built bundle at
  `<package>/views/<view_id>.html`.
- `CREDENTIAL_HEADERS` *(optional)* — header names the tools read off the
  transport; used to derive the model-facing auth hint.

A tool may additionally tag a value with the `Kind` it is — on a `ToolResult`
data key to say what it publishes, on a parameter to say what it takes. The
tag is advertised in the tool's `_meta`, and lets an `mcp_state` client move a
large value — a geometry, an item collection — from the tool that produced it
to the tool that needs it *without the model generating or reading it*.
Resolution is by kind, so producer and consumer may be different toolsets on
different servers. See `mcp_runtime.kinds` for the shared vocabulary.

Keeping a value out of the context is client-side work, so an external MCP host
does none of it: served to Claude.ai or ChatGPT, a tagged toolset behaves like
any other. Tag for the agents that understand it, and size tool returns for the
clients that don't.

Tagging is an accelerator, not a requirement: `mcp_state` moves values across
**unmodified third-party MCP servers** too, by capturing large returns on size
and letting the model point a parameter at one with an `@state:<key>` handle.
What the tag buys is that the parameter leaves the model's schema entirely.

Treat `ToolResult`, `Kind`, and the `ui/*` wire protocol as **public API**. The
state contract, worked through as sequence diagrams — including the trust
assumption it rests on — is in
**[docs/SESSION-STATE.md](./docs/SESSION-STATE.md)**, with a runnable version
of the whole thing, against a third-party server included, in
**[examples/session-state/](./examples/session-state/)** (`uv run python
examples/session-state/demo.py` — no API key needed).

## Install

From PyPI — see the badge above for the current release:

```bash
# base: runtime + cli (lean, for tool-serving images)
pip install mcp-toolsets-runtime

# session state, for wiring it into an agent of your own
pip install "mcp-toolsets-runtime[state]"

# the agent — build_agent, run_turn and the host helpers, no UI framework
pip install "mcp-toolsets-runtime[agent]"

# the bundled Chainlit web host, on top of the agent
pip install "mcp-toolsets-runtime[web]"
```

Each extra includes the one before it, so name only the outermost you need.

With uv, as a consumer — an ordinary dependency, no source override:

```toml
dependencies = ["mcp-toolsets-runtime[web]"]
```

Imports are unchanged from the old workspace packages: `from mcp_runtime.server
import build_server`, etc. `uv.lock` pins whatever resolved, so upgrading is
`uv lock --upgrade-package mcp-toolsets-runtime`. The package is pre-1.0, where
a minor release may break — bound it at the next minor in your own
`pyproject.toml` if you'd rather take those deliberately.

**Consuming this package** — the plugin contract, serving toolsets, wiring up UI
views (including `mcp-agent install-elements` and the npm bridge), wiring session
state into your own agent, and migrating off the in-repo workspace: see
**[docs/CONSUMING.md](./docs/CONSUMING.md)**.

## Develop

```bash
uv sync --all-extras   # install every extra ([web] included) + dev tools
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
