# @developmentseed/mcp-view

The view-side bridge for [MCP Apps](https://github.com/modelcontextprotocol/ext-apps)
UI views: the `ui/*` postMessage protocol as three small functions.

An MCP tool can ship a UI view — an HTML bundle the server exposes as a
`ui://<toolset>/<id>` resource, which the host renders inline in the chat
instead of raw JSON. The bundle runs in a sandboxed iframe and talks to its host
over `ui/*` JSON-RPC-over-postMessage. This package wraps the
[official SDK](https://www.npmjs.com/package/@modelcontextprotocol/ext-apps)
so your view never has to know which host it landed in.

```bash
npm install @developmentseed/mcp-view
```

## Use

```ts
import { onData, sendMessage } from "@developmentseed/mcp-view";

// The tool's structuredContent, delivered by the host once the tool has run.
onData((data) => render(data));

// A user turn back into the conversation, so the model calls the next tool.
button.onclick = () => sendMessage("clip this to the Severn catchment");
```

`onData` registers your handler and connects to the host; `sendMessage` posts
a message as if the user had typed it.

### Calling a tool from the view

When the view needs more than the tool's result to draw itself — a boundary
too large to put in the model's context, the next page of a table — it can
call a tool itself:

```ts
import { callTool, ToolError } from "@developmentseed/mcp-view";

try {
  const result = await callTool("places_get", {
    place_id: data.place_id,
    include: ["boundary"],
  });
  drawOutline(result.structuredContent);
} catch (error) {
  // ToolError: the tool ran and answered `isError: true` (its result is on
  // `error.result`). Any other Error: the host refused or could not deliver
  // the call.
}
```

This is MCP Apps' `tools/call`, proxied by the host to the server the view came
from. The result comes back to the view alone: nothing enters the conversation
and no model turn is spent, which is the difference from `sendMessage`. A host
only runs tools whose `_meta.ui.visibility` includes `"app"`; a tool that
declares none is visible to both the model and views, as the spec says.

`callTool` resolves with the host's `CallToolResult`, and rejects both when the
host answers an error and when the result has `isError: true` — unlike the
SDK's `callServerTool`, which resolves with an error result and leaves the
check to you.

Optionally identify your view during the `ui/initialize` handshake — call it
before the first `onData`, `sendMessage` or `callTool`:

```ts
import { configure } from "@developmentseed/mcp-view";

configure({ name: "dataset-gallery", version: "1.2.0" });
```

## Hosts

The same bundle works unchanged in any MCP Apps host — Claude.ai, ChatGPT,
Goose, VS Code — and in the web client bundled with
[`mcp-toolsets-runtime`](https://github.com/developmentseed/mcp-toolsets-runtime),
which implements the host end of this protocol.

Views are progressive enhancement: a tool's `message` and structured content
still stand on their own in a plain MCP client that can't render them, so
shipping a view never makes a tool unusable anywhere.

## Serving the view

Building and serving the bundle is the server's job, not this package's. If
you're using `mcp-toolsets-runtime`, declare `VIEWS = {tool_name: view_id}` in
your toolset and ship the built bundle at `<package>/views/<view_id>.html` — see
[CONSUMING.md](https://github.com/developmentseed/mcp-toolsets-runtime/blob/main/docs/CONSUMING.md).

## Notes

- ESM only, with TypeScript types included.
- Versioned in lockstep with the Python `mcp-toolsets-runtime` distribution, from
  the same repository.

MIT © Development Seed
