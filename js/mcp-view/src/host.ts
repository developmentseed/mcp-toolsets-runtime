// The host bridge — the one seam a view shares no matter what framework it is
// built with. A view runs in a sandboxed iframe and talks to its host over the
// MCP Apps `ui/*` JSON-RPC-over-postMessage protocol, via the standard SDK
// (`@modelcontextprotocol/ext-apps`). That is what Claude, ChatGPT, Goose and
// VS Code speak, and what the web client bundled with mcp-toolsets-runtime
// implements the other end of.
//
// The SDK handles the `ui/initialize` handshake, delivers the tool's result
// (`ui/notifications/tool-result`), sends a chat message (`ui/message`) and
// calls a tool through the host (`tools/call`). This module wraps it in three
// small functions so the views stay host-agnostic:
//
//   onData(handler)        — the tool's structuredContent, once the host sends it
//   sendMessage(text)      — a user turn back into the chat, to run the next tool
//   callTool(name, args)   — one tool's result for the view alone, no model turn
//
// A third thing happens with no function for it: the SDK watches the document
// and reports its size to the host as `ui/notifications/size-changed`, so a
// view is sized by its content rather than by a box the host guessed. See
// `AUTO_RESIZE`.
import { App } from "@modelcontextprotocol/ext-apps";

// One App per iframe, connected once. Created lazily on first use so the
// tool-result handler is registered before connect() runs the handshake (the
// SDK warns if a handler is added after connect()).
let appPromise: Promise<App> | null = null;
let dataHandler: ((payload: unknown) => void) | null = null;
let appInfo: { name: string; version: string } = {
  name: "mcp-view",
  version: "0.12.5", // x-release-please-version
};

/** Tell the host how tall the view wants to be, and keep telling it.
 *
 * The SDK's own default, set here rather than left implicit: it is the whole
 * reason a view's height is not a number somebody picked, and a default that
 * decides that much is worth being able to read. A `ResizeObserver` on the
 * document sends `ui/notifications/size-changed` on every change.
 *
 * Nothing breaks where a host ignores it — the notification is advisory, and
 * a host that does not listen keeps whatever sizing it already had.
 */
const AUTO_RESIZE = { autoResize: true };

/**
 * Set the App identity reported to the host during `ui/initialize`. Optional —
 * call it before the first `onData`/`sendMessage`/`callTool`. Defaults to a
 * generic name.
 */
export function configure(info: { name: string; version: string }): void {
  appInfo = info;
}

function app(): Promise<App> {
  if (!appPromise) {
    const instance = new App(appInfo, {}, AUTO_RESIZE);
    // The host delivers the tool's structuredContent here; hand it to whatever
    // onData registered. Read dataHandler lazily so a later onData still wins.
    instance.ontoolresult = (params) => {
      if (params.structuredContent != null) dataHandler?.(params.structuredContent);
    };
    appPromise = instance.connect().then(() => instance);
  }
  return appPromise;
}

/** Register for the tool's structuredContent, and connect to the host. */
export function onData<T>(handler: (payload: T) => void): void {
  dataHandler = handler as (payload: unknown) => void;
  void app();
}

/** Send text back into the conversation, so the model calls the next tool. */
export function sendMessage(text: string): void {
  void app().then((instance) =>
    instance.sendMessage({ role: "user", content: [{ type: "text", text }] }),
  );
}

/** What the host answers a `tools/call` with: MCP's `CallToolResult`.
 *
 * Named off the SDK's own signature rather than imported from the MCP client
 * package, which is the SDK's dependency and not this one's: the type a view
 * gets is whatever the SDK it runs on returns.
 */
export type CallToolResult = Awaited<ReturnType<App["callServerTool"]>>;

/** A tool that ran and answered `isError: true`.
 *
 * Its own type so a view can tell "the tool said no" — a place that does not
 * exist, an argument out of range — from "the call never got there", which
 * rejects with the SDK's plain `Error`. The message is the result's text, which
 * is what a tool writes its failure in; the whole result is on `result`.
 */
export class ToolError extends Error {
  readonly result: CallToolResult;

  constructor(result: CallToolResult) {
    const text = (result.content ?? [])
      .map((block) => (block.type === "text" ? block.text : ""))
      .join("");
    super(text || "the tool returned an error");
    this.name = "ToolError";
    this.result = result;
  }
}

/**
 * Call a tool through the host, and get its result back in the view.
 *
 * This is MCP Apps' `tools/call`, sent on the same connection `onData` and
 * `sendMessage` use. The host runs the tool and answers the view alone: nothing
 * enters the conversation, and no model turn is spent. That is the difference
 * from `sendMessage`, which asks the model to call the tool and puts the
 * request and the result in the chat. Use it for what a view needs to draw
 * itself — a boundary too large for the model's context, the next page of a
 * table — rather than for what the conversation should know.
 *
 * A host only answers for tools whose `_meta.ui.visibility` includes `"app"`
 * (the spec's default when a tool declares none is `["model", "app"]`), and
 * only for tools of the server the view came from.
 *
 * Resolves with the host's `CallToolResult` when the tool succeeded. Rejects
 * in two cases, both of which are failures a view has to handle:
 *
 *   - the host answered a JSON-RPC error — the tool is not visible to views,
 *     the host does not proxy tool calls, the server could not be reached —
 *     with the SDK's `Error`;
 *   - the tool ran and its result has `isError: true`, with a {@link ToolError}
 *     carrying that result.
 *
 * Rejecting on `isError` is a choice this module makes over the SDK, whose
 * `callServerTool` resolves with such a result and leaves the check to every
 * caller: a view that forgets it would draw an error message as data.
 */
export async function callTool(
  name: string,
  args: Record<string, unknown>,
): Promise<CallToolResult> {
  const instance = await app();
  const result = await instance.callServerTool({ name, arguments: args });
  if (result.isError) throw new ToolError(result);
  return result;
}
