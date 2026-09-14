---
name: writing-mcp-toolsets
description: Write a toolset for mcp-toolsets-runtime, or port existing Python into one. Covers the plugin contract, typed returns, the async rule, session state, per-user credentials and UI views. Use when adding a toolset, turning a library or script into tools, or working out why a toolset will not serve.
---

# Writing a toolset

A toolset is a Python package that exports a list of LangChain tools. The
runtime imports it by name and serves it over MCP.

This file is the procedure. The reference is `docs/CONSUMING.md` in
mcp-toolsets-runtime; a marker like `(#4c)` below means that section of it.
Read the reference for detail. Do not reproduce it here.

## Get these right first

They are cheap now and expensive later.

**1. The consuming repo owns no runtime code.** `mcp_runtime`, `mcp_state`,
`mcp_cli`, `mcp_agent`, `mcp_agent_api` and `mcp_toolset` come from the
`mcp-toolsets-runtime` package. Never add a module under one of those names,
and never patch runtime behaviour locally. If the runtime is wrong, fix it in
the runtime, release it, and bump the pin.

**2. Data key names are public.** Every key in a tool's return except
`message` is stored as `<toolset>/<tool>/<field>`. The agent's model reads that
name when it decides which stored value to pass into a later call, and that
call may land in a different toolset. The key is the only thing the two share:
no types, no imports, no registry.

So name the thing, not its type. `area_of_interest` is a good key; `geometry`
is a bad one, because a coverage footprint is also a geometry and the two are
identical JSON. A model handed the wrong one produces confident nonsense and
nothing will notice. (#4c)

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
headers the tools read. (#2)

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
| `os.environ["API_KEY"]` | `CREDENTIAL_HEADERS` plus `credential_from_header` (#5e) |
| a function returning something big: GeoJSON, an item collection, a dataframe | a declared data key on the result, so it moves through session state (#4) |
| a parameter whose value must come from a real source rather than the model: a boundary, an item collection, the exact bbox under discussion | `Annotated[dict, NotAuthored()]` on that parameter (#4c) |
| a plot, a map, an HTML table | a view (#3) |

Do not port the CLI or argparse layer, the HTTP server, the authentication,
the caching, or the logging configuration. The runtime provides the server.
The model is the interface.

Split tools by what someone might ask for, not by how the original code was
organised. One tool per question.

### Values the model should not carry

Two mechanisms, one at each end. They are usually used together, but they
answer different questions and either can be used alone.

**Producing: declare a data key.** Every key in a return except `message` is
captured into session state, so a large value moves from the tool that made it
to the tool that needs it without passing through the conversation.

**Consuming: tag the parameter `NotAuthored`.** This says a model must not
write this value. It is about authorship, not size. A 2000-vertex boundary
qualifies, and so does a four-number bbox that has to be *the* one under
discussion, because a plausible invention is worse than no answer. The tag
says nothing about types and nothing about session state; a client that
implements `mcp_state` is what narrows the parameter to a stored value.

```python
from mcp_runtime.declarations import NotAuthored


@tool
async def clip_raster(
    dataset_id: str,
    aoi: Annotated[dict, NotAuthored()],
) -> ClipResult | ToolError: ...
```

Tagging something that is not a parameter fails at startup rather than going
unnoticed until a client connects.

Both are client-side work. The bundled agent does it; an external MCP host such
as Claude.ai or ChatGPT does not. So tag for the clients that understand it,
and still size the return so a client that ignores it survives. (#4)

## Where to read more

| Question | Where |
| --- | --- |
| the plugin contract in full | CONSUMING #2 |
| UI views | CONSUMING #3 |
| session state, with flowcharts and worked scenarios | `docs/SESSION-STATE.md` |
| per-user credentials | CONSUMING #5e |
| a worked async tool with views | `toolsets/stac-explorer` in mcp-toolsets |
| a worked credential tool | `toolsets/credential-demo` in mcp-toolsets |
