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
| `mcp_runtime` | Discovers a toolset's LangChain tools (`TOOLS`) and serves them as an MCP server; serves UI views (`VIEWS`) as `ui://` resources; derives server `instructions` from `CREDENTIAL_HEADERS`; advertises what each tool publishes into session state, and which parameters a model may not write (`NotAuthored`). Entry points: `mcp-serve` (one toolset), `mcp-serve-local` (several at once, for local dev), `mcp-index`. |
| `mcp_state` | Session state for *any* agent driving MCP tools: the `tool_state` namespace, `StateCaptureMiddleware` (moves large payloads out of the transcript), `inspect_state` (the model reads one on demand), and `bind_injected` (fills declared parameters from state, and offers `@state:<key>` handles on the rest). A filled parameter leaves a receipt, so a value the model never saw can still be traced to the tool that published it. Works against unmodified third-party servers. Requires the `[state]` extra. |
| `mcp_cli` | Typer CLI to list and call tools on a running MCP service. Entry point: `mcp-cli`. |
| `mcp_toolset` | Scaffolds a new toolset in a consumer repo (`mcp-toolset new [--with-ui] <name>`), wired to this package + the npm view bridge. |
| `mcp_agent` | Example Chainlit chat agent that discovers MCP servers behind an index URL and drives their tools, with `mcp_state` wired in (`MCP_AGENT_STATE=0` to opt out). Conversations are checkpointed per `thread_id` — in-process by default, PostgreSQL via `MCP_AGENT_CHECKPOINT` + the `[checkpointing-postgres]` extra. Ships the Chainlit host element `elements/McpView.jsx`. Entry points: `mcp-agent`, `mcp-agent-web`. `mcp_agent.main` (`build_agent`, `run_turn`), `mcp_agent.streaming` (`stream_turn`, the same turn yielded as it happens) and `mcp_agent.host` — the UI-framework-free helpers a host of its own needs (view bundles and props, and the tool-step arguments session state filled in) — need the `[agent]` extra. `mcp_agent.web`, the Chainlit host, needs `[web]` on top. |
| `mcp_agent_api` | The agent over HTTP. `mcp_agent_api.events` turns one turn into [AG-UI](https://github.com/ag-ui-protocol/ag-ui) events — tokens, tool calls, and the two things AG-UI has no vocabulary for: where each tool's arguments came from and which `ui://` view renders its result, both as `ACTIVITY_*` messages carrying a rendered `display` line beside their fields. Imports no FastAPI. `mcp_agent_api.routes` is an `APIRouter` over a built agent — `POST /runs` streams that turn as SSE, and four read routes serve what the stream deliberately leaves out: the thread's transcript, its turns with the state each ended holding, a session-state payload in full (`?turn=N` for the value as it stood then, which the checkpointer has kept all along), and a `ui://` view bundle. `mcp_agent_api.app` closes the stack for a deployment with no application of its own: `create_app(build=…)` puts a lifespan, a checkpointer and CORS around those routes, and a module-level `app` serves under `uvicorn mcp_agent_api.app:app`. Requires the `[api]` extra. |
| `@developmentseed/mcp-view` (`js/mcp-view`) | The view-side `ui/*` postMessage bridge a toolset UI imports (`onData` / `sendMessage`). Published to npm separately. |

### The toolset plugin contract

`mcp_runtime` discovers a toolset purely by convention — a `<toolset>.tools`
module exporting:

- `TOOLS` — a non-empty list of LangChain tools that return a `ToolResult`.
- `VIEWS` *(optional)* — `{tool_name: view_id}`, with a built bundle at
  `<package>/views/<view_id>.html`.
- `CREDENTIAL_HEADERS` *(optional)* — header names the tools read off the
  transport; used to derive the model-facing auth hint.

Every data key of a `ToolResult` — every field but `message` — is a value the
tool publishes. An `mcp_state` client captures each into session state under
`<toolset>/<tool>/<field>` and lets a later tool be pointed at it by that key,
so a large value — a geometry, an item collection — moves from the tool that
produced it to the tool that needs it *without passing through the model*.
Producer and consumer may be different toolsets on different servers; the key
is the only thing they share, which is why **a data key is a public name**.

A tool may also tag a parameter `NotAuthored`, which says only that a model
must not write the value — no type, nothing for another toolset to agree with.
An `mcp_state` client narrows that parameter until the only thing it accepts is
a reference to a value some tool already produced; a client that has never
heard of any of this is unaffected.

Keeping a value out of the context is client-side work, so an external MCP host
does none of it: served to Claude.ai or ChatGPT, a toolset behaves like
any other. Tag for the agents that understand it, and size tool returns for the
clients that don't.

Tagging is an accelerator, not a requirement: `mcp_state` moves values across
**unmodified third-party MCP servers** too, by capturing large returns on size
and letting the model point a parameter at one with an `@state:<key>` handle.
What the tag buys is that the parameter leaves the model's schema entirely.

Treat `ToolResult`, `NotAuthored`, and the `ui/*` wire protocol as **public API**. The
state contract, worked through as sequence diagrams — including the trust
assumption it rests on — is in
**[docs/SESSION-STATE.md](./docs/SESSION-STATE.md)**, with a runnable version
of the whole thing, against a third-party server included, in
**[examples/session-state/](./examples/session-state/)** (`uv run python
examples/session-state/demo.py` — no API key needed). The same machinery on the
wire, driven from a small React chat client over HTTP, is in
**[examples/agui-events/](./examples/agui-events/)** — tokens streaming, tool
calls and receipts in the order they arrive, and a state panel whose values are
a fetch away rather than on the wire.

## Install

From PyPI — see the badge above for the current release:

```bash
# base: runtime + cli (lean, for tool-serving images)
pip install mcp-toolsets-runtime

# session state, for wiring it into an agent of your own
pip install "mcp-toolsets-runtime[state]"

# the agent — build_agent, run_turn, stream_turn and the host helpers
pip install "mcp-toolsets-runtime[agent]"

# the bundled Chainlit web host, on top of the agent
pip install "mcp-toolsets-runtime[web]"

# the agent over HTTP, as AG-UI events — an alternative to [web], not a layer
pip install "mcp-toolsets-runtime[api]"
```

`[state]`, `[agent]` and `[web]` are a chain, so name only the outermost you
need. `[api]` sits beside `[web]` on top of `[agent]`: a deployment serving the
API does not install Chainlit, and one serving the chat does not install AG-UI.

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
state into your own agent, serving that agent over HTTP, and migrating off the
in-repo workspace: see
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
