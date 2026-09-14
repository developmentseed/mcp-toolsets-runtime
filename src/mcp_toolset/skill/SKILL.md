---
name: writing-mcp-toolsets
description: Write a toolset for mcp-toolsets-runtime, or port existing Python into one. Covers the plugin contract, typed returns, the async rule, session state, per-user credentials and UI views. Use when adding a toolset, turning a library or script into tools, or working out why a toolset will not serve.
---

# Writing a toolset

A toolset is a Python package that exports a list of LangChain tools. The
runtime imports it by name and serves it over MCP.

This file is the procedure. The reference is `docs/CONSUMING.md` in
mcp-toolsets-runtime, and the section numbers below point into it. Read the
reference for detail; do not reproduce it here.

## Get these right first

They are cheap now and expensive later.

**1. The consuming repo owns no runtime code.** `mcp_runtime`, `mcp_state`,
`mcp_cli`, `mcp_agent`, `mcp_agent_api` and `mcp_toolset` come from the
`mcp-toolsets-runtime` package. Never add a module under one of those names,
and never patch runtime behaviour locally. If the runtime is wrong, fix it in
the runtime, release it, and bump the pin.

**2. Data key names are public.** Every key in a tool's return except
`message` is stored as `<toolset>/<tool>/<field>`. A *different* toolset's
model reads that name when it decides which stored value to pass to the next
call. Nothing else crosses between toolsets: no shared types, no imports, no
registry. So name the thing, not its type. `area_of_interest` is a good key;
`geometry` is a bad one, because a coverage footprint is also a geometry and
the two are identical JSON. A model handed the wrong one produces confident
nonsense and nothing will notice. (§4c)

**3. Scaffold, never hand-roll.** `mcp-toolset new` writes the package, the
test, the pyproject and whatever deployment config the repo declares. A
hand-made directory misses files you will not notice until deploy.

## Steps

### 1. Scaffold

```bash
uv run mcp-toolset new my-toolset          # --with-ui to add a React view
```

This writes `toolsets/my-toolset/` containing `src/my_toolset/tools.py`, a
test, a pyproject, and the deployment file the repo's root pyproject declares
under `[tool.mcp-toolset] deployment-config`.

### 2. Write the tools

`tools.py` must export `TOOLS`, a non-empty list. Two exports are optional:
`VIEWS` maps a tool name to a view id, and `CREDENTIAL_HEADERS` lists the HTTP
headers the tools read. (§2)

Each tool:

- is decorated `@tool`
- has a docstring, which becomes the description the model reads; an empty one
  fails at startup
- is `async def` if it does any I/O. Use `def` only for pure computation,
  because the runtime hands a sync tool to a thread pool and holds a thread
  for the length of the call
- returns a `ToolResult` subclass, unioned with `ToolError` if it can fail

```python
class SearchResult(ToolResult):
    """Datasets matching the query."""

    datasets: NotRequired[list[Dataset]]


@tool
async def search(query: str) -> SearchResult | ToolError:
    """Search the catalogue for datasets matching a free-text query."""
```

The annotation becomes the MCP output schema. A tool that does not convert is
refused at startup, not on a user's call.

### 3. Handle errors in the right place

| Situation | What to do |
| --- | --- |
| the call failed in a way the model should see and can act on | return `ToolError(error="upstream_error", detail="...")` |
| a credential header is missing | let `MissingCredentialError` propagate |
| a bug in the tool | let it raise |

Return `ToolError` rather than raising for anything you expect to happen. The
model reads `detail`, so write it for the model: what went wrong, and what to
do next.

### 4. Write the tests

Both this repo and mcp-toolsets set `asyncio_mode = "auto"`, so an async test
needs no decorator:

```python
async def test_search():
    result = await search.ainvoke({"query": "sentinel"})
    assert result["datasets"]
```

For a tool that reads a credential, `mcp_runtime.credentials.header_context`
supplies the headers without running a server.

### 5. Prove it works

```bash
./scripts/lint
./scripts/test        # includes the contract sweep over toolsets/
```

Then serve it and make a real call, because the tests do not exercise the MCP
schema the model actually sees:

```bash
uv run mcp-serve-local                     # every toolset, index at /
uv run mcp-cli list --url http://localhost:8000/my-toolset/mcp
uv run mcp-cli call search query=sentinel --url http://localhost:8000/my-toolset/mcp
```

Read the schema `list` prints. That is the tool as the model receives it, and
it is where a missing description or a wrong type shows up.

## Porting an existing codebase

Map the parts first. Do not translate file by file.

| What you have | What it becomes |
| --- | --- |
| a function that calls an HTTP API | one `async def` tool returning `Result \| ToolError` |
| a function returning a dict | a `ToolResult` subclass, one `NotRequired` field per key |
| `raise ValueError("no results")` | `return ToolError(error="no_results", detail=...)` |
| `requests`, `urllib` | `httpx.AsyncClient` |
| `os.environ["API_KEY"]` | `CREDENTIAL_HEADERS` plus `credential_from_header` (§5e) |
| a large return: GeoJSON, an item collection, a dataframe | a declared data key, with `NotAuthored` on the parameter that consumes it (§4) |
| a plot, a map, an HTML table | a view (§3) |

Do not port the CLI or argparse layer, the HTTP server, the authentication,
the caching, or the logging configuration. The runtime provides the server.
The model is the interface.

Split tools by what someone might ask for, not by how the original code was
organised. One tool per question.

### Large values

If a tool produces something the model should not be reading or writing back,
such as a 2000-vertex boundary or a full item collection, declare it as a data
key and tag the *consuming* tool's parameter `NotAuthored`. The value moves
between tools through session state without entering the conversation.

This is client-side work. The bundled agent does it; an external MCP host such
as Claude.ai or ChatGPT does not. So tag for the clients that understand it,
and still size the return so a client that ignores it survives. (§4)

## Where to read more

| Question | Where |
| --- | --- |
| the plugin contract in full | CONSUMING §2 |
| UI views | CONSUMING §3 |
| session state, with flowcharts and worked scenarios | `docs/SESSION-STATE.md` |
| per-user credentials | CONSUMING §5e |
| a worked async tool with views | `toolsets/stac-explorer` in mcp-toolsets |
| a worked credential tool | `toolsets/credential-demo` in mcp-toolsets |
