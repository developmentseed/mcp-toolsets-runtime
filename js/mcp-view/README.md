# @developmentseed/mcp-view

The view-side bridge for [MCP Apps](https://github.com/modelcontextprotocol/ext-apps)
UI views: the `ui/*` postMessage protocol as two small functions.

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

That's the whole surface. `onData` registers your handler and connects to the
host; `sendMessage` posts a message as if the user had typed it.

Optionally identify your view during the `ui/initialize` handshake — call it
before the first `onData` or `sendMessage`:

```ts
import { configure } from "@developmentseed/mcp-view";

configure({ name: "dataset-gallery", version: "1.2.0" });
```

## Hosts

The same bundle works unchanged in any MCP Apps host — Claude.ai, ChatGPT,
Goose, VS Code — and in the Chainlit chat agent bundled with
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
