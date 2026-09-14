# mcp-toolsets-runtime

[![PyPI](https://img.shields.io/pypi/v/mcp-toolsets-runtime?label=PyPI)](https://pypi.org/project/mcp-toolsets-runtime/)
[![npm](https://img.shields.io/npm/v/%40developmentseed%2Fmcp-view?label=npm)](https://www.npmjs.com/package/@developmentseed/mcp-view)

The shared runtime for [MCP Toolsets](https://github.com/developmentseed/mcp-toolsets).
`developmentseed/mcp-toolsets` and the repos generated from it install this
package rather than each carrying its own copy.

## What's in here

One Python distribution, `mcp-toolsets-runtime`, containing six top-level
modules, plus a JS bridge published separately to npm.

| Module | What it is | Extra needed |
| --- | --- | --- |
| `mcp_runtime` | Serves a toolset's tools as an MCP server. | none |
| `mcp_cli` | A Typer CLI that lists and calls tools on a running service. | none |
| `mcp_toolset` | Scaffolds a new toolset in a consumer repo. | none |
| `mcp_state` | Session state, so large tool values stay out of the model's context. | `[state]` |
| `mcp_agent` | An agent that drives the tools behind an index URL. | `[agent]` |
| `mcp_agent_api` | That agent over HTTP, with a web client. | `[api]` |
| `@developmentseed/mcp-view` | The view-side bridge a toolset UI imports. | npm |

### `mcp_runtime`

Discovers a toolset's LangChain tools and serves them over MCP. It also:

- serves the toolset's UI views as `ui://` resources
- derives the server's `instructions` from `CREDENTIAL_HEADERS`, so the model
  is told a credential rides the connection
- advertises what each tool publishes into session state, and which parameters
  a model may not write

Entry points: `mcp-serve` for one toolset, `mcp-serve-local` for several at
once during development, and `mcp-index` for the directory service.

### `mcp_state`

Session state for any agent driving MCP tools, not only the bundled one. Four
pieces:

- the `tool_state` namespace, where values are kept
- `StateCaptureMiddleware`, which moves large payloads out of the transcript
- `inspect_state`, which the model calls to read one value on demand,
  including the value as it stood at an earlier turn
- `bind_injected`, which fills declared parameters from state and offers
  `@state:<key>` handles on the rest

A filled parameter leaves a receipt, so a value the model never saw can still
be traced back to the tool that published it. All of this works against
unmodified third-party servers.

### `mcp_agent`

`mcp-agent` is a terminal chat. It discovers every MCP server behind an index
URL and lets a model drive their tools, with `mcp_state` wired in. Set
`MCP_AGENT_STATE=0` to opt out.

Conversations are checkpointed per `thread_id`. The store is in-process by
default, or PostgreSQL through `MCP_AGENT_CHECKPOINT` and the
`[checkpointing-postgres]` extra.

The module splits three ways, all under `[agent]`:

| Import | What it gives you |
| --- | --- |
| `mcp_agent.main` | `build_agent`, `run_turn` |
| `mcp_agent.streaming` | `stream_turn`, the same turn yielded as it happens |
| `mcp_agent.host` | UI-framework-free helpers: view bundles and props, and the arguments session state filled in |

`mcp_agent` itself has no UI. To present a turn, serve the web client that
`[api]` ships, or build your own on `mcp_agent.host` or the HTTP API below.

### `mcp_agent_api`

The agent over HTTP. Three layers, each usable without the one above it.

**`mcp_agent_api.events`** turns one turn into
[AG-UI](https://github.com/ag-ui-protocol/ag-ui) events: tokens, tool calls,
and two things AG-UI has no vocabulary for. Those two are where each tool's
arguments came from, and which `ui://` view renders its result. Both travel as
`ACTIVITY_*` messages carrying a rendered `display` line beside their fields.
This module imports no FastAPI.

**`mcp_agent_api.routes`** is an `APIRouter` over a built agent, serving six
routes:

| Route | What it does |
| --- | --- |
| `POST /runs` | streams one turn as SSE |
| `GET /threads/{id}` | the thread's transcript |
| `GET /threads/{id}/turns` | its turns, with the state each ended holding |
| `GET /threads/{id}/state` | a session-state payload in full; `?turn=N` for the value as it stood then |
| `GET /views/...` | a `ui://` view bundle |
| `GET /connections` | what the agent connected to, and which credential headers it wants |

The four read routes serve what the stream deliberately leaves out.
`GET /connections` is what a client needs before there is a conversation at
all.

**`mcp_agent_api.app`** closes the stack for a deployment with no application
of its own. `create_app(build=…)` wraps those routes in a lifespan, a
checkpointer, CORS and two health probes. A module-level `app` means
`uvicorn mcp_agent_api.app:app` serves the lot.

### The toolset plugin contract

Writing a toolset with a coding agent? `uv run mcp-toolset skill --install`
puts the authoring skill in the repo. It covers this section as a procedure,
and adds a mapping table for porting an existing codebase into tools. It ships
in the wheel, so it matches the pinned version.

`mcp_runtime` finds a toolset by convention. Given the name `my-toolset` it
imports `my_toolset.tools` and reads three module-level exports:

- **`TOOLS`** (required) — a non-empty list of LangChain tools that return a
  `ToolResult`.
- **`VIEWS`** (optional) — `{tool_name: view_id}`, with a built bundle at
  `<package>/views/<view_id>.html`.
- **`CREDENTIAL_HEADERS`** (optional) — header names the tools read off the
  transport, used to derive the model-facing auth hint.

**A data key is a public name.** Every field of a `ToolResult` except
`message` is a value the tool publishes. An `mcp_state` client captures each
one into session state under `<toolset>/<tool>/<field>`, and a later tool can
be pointed at it by that key. A large value such as a geometry or an item
collection therefore moves from the tool that produced it to the tool that
needs it without passing through the model. The producer and the consumer may
be different toolsets on different servers, and the key is the only thing they
share.

A tool may also tag a parameter `NotAuthored`. That says one thing: a model
must not write this value. It says nothing about types, and gives another
toolset nothing to agree with. An `mcp_state` client narrows the parameter
until the only thing it accepts is a reference to a value some tool already
produced. A client that has never heard of any of this is unaffected.

Keeping a value out of the context is client-side work, so an external MCP
host does none of it. Served to Claude.ai or ChatGPT, a toolset behaves like
any other. Tag for the agents that understand it, and size tool returns for
the clients that do not.

Tagging is an accelerator, not a requirement. `mcp_state` also moves values
across unmodified third-party MCP servers, by capturing large returns on size
and letting the model point a parameter at one with an `@state:<key>` handle.
What the tag buys is that the parameter leaves the model's schema entirely.

Treat `ToolResult`, `NotAuthored` and the `ui/*` wire protocol as public API.

Further reading:

- [docs/SESSION-STATE.md](./docs/SESSION-STATE.md) — the state contract as
  sequence diagrams, including the trust assumption it rests on.
- [examples/session-state/](./examples/session-state/) — a runnable version of
  the whole thing, a third-party server included. Run
  `uv run python examples/session-state/demo.py`; no API key needed.
- [examples/agui-events/](./examples/agui-events/) — the same machinery over
  HTTP, driven by the client the wheel ships. Tokens stream, tool calls and
  receipts arrive in order, and a state panel fetches values rather than
  receiving them on the wire.

## The bundled web client

The `[api]` extra installs a page as well as an API. `mcp_agent_api.app`
serves it at the root, so a container running `uvicorn mcp_agent_api.app:app`
is a working chat over the toolsets behind `MCP_URL`. It shows the transcript,
tool calls and receipts as they happen, the session-state panel, and `ui://`
views in their frames. No Node runs in the image, and no front end is copied
into the deployment.

A deployment sets its text and one colour from the environment at startup:

| Variable | What it sets |
| --- | --- |
| `MCP_AGENT_UI_TITLE` | the name in the header and the browser tab |
| `MCP_AGENT_UI_TAGLINE` | one line beside it |
| `MCP_AGENT_UI_GREETING` | the opening paragraph; unset, the page says what `GET /connections` reports |
| `MCP_AGENT_UI_EXAMPLES` | questions offered as buttons, one per line or a JSON array |
| `MCP_AGENT_UI_ACCENT` | a CSS colour |

Anything structural is a change to the client itself, whose source is
[`js/agent-ui`](./js/agent-ui). The client talks to the six routes in
`mcp_agent_api.routes` and nothing else. A host that mounts `create_router`
into its own application therefore serves the same client with
`mount_ui(app, api="/api")`. What forces a fork is diverging from those
routes, not from the application around them. `create_app(ui=False)` turns the
page off for a deployment with a front end of its own.

## Install

From PyPI. The badge above shows the current release.

```bash
# base: runtime + cli, for a lean tool-serving image
pip install mcp-toolsets-runtime

# session state, to wire into an agent of your own
pip install "mcp-toolsets-runtime[state]"

# the agent: build_agent, run_turn, stream_turn and the host helpers
pip install "mcp-toolsets-runtime[agent]"

# the agent over HTTP as AG-UI events, plus the web client that renders them
pip install "mcp-toolsets-runtime[api]"
```

`[state]`, `[agent]` and `[api]` form a chain, so name only the outermost one
you need.

As a consumer with uv, this is an ordinary dependency with no source override:

```toml
dependencies = ["mcp-toolsets-runtime[api]"]
```

Imports are unchanged from the old workspace packages, so
`from mcp_runtime.server import build_server` still works. `uv.lock` pins
whatever resolved, and upgrading is
`uv lock --upgrade-package mcp-toolsets-runtime`.

The package is pre-1.0, where a minor release may break. Bound it at the next
minor in your own `pyproject.toml` if you would rather take those deliberately.

**[docs/CONSUMING.md](./docs/CONSUMING.md)** covers the rest: the plugin
contract, serving toolsets, wiring up UI views and the npm bridge, wiring
session state into your own agent, serving that agent over HTTP, and migrating
off the in-repo workspace.

## Develop

```bash
uv sync --all-extras   # every extra, plus dev tools
./scripts/lint         # ruff check + ruff format --check + mypy
./scripts/test         # pytest
./scripts/build-js     # the npm view bridge, and the web client, which builds
                       # into src/mcp_agent_api/ui (needs node)
```

Lint and type rules live in `pyproject.toml`. The scripts pass no rule flags,
so your editor, `./scripts/format` and CI all agree.

## Releases

[release-please](https://github.com/googleapis/release-please) manages the
version and `CHANGELOG.md` from Conventional Commits. **Your PR title is the
changelog entry**, and CI fails a PR whose title is not a valid conventional
commit. The Python package and the JS bridge share one version. See
[CONTRIBUTING.md](./CONTRIBUTING.md).
