// The host bridge — the one seam a view shares no matter what framework it is
// built with. A view runs in a sandboxed iframe and talks to its host over the
// MCP Apps `ui/*` JSON-RPC-over-postMessage protocol, via the standard SDK
// (`@modelcontextprotocol/ext-apps`). That is what Claude, ChatGPT, Goose and
// VS Code speak, and what the web client bundled with mcp-toolsets-runtime
// implements the other end of.
//
// The SDK handles the `ui/initialize` handshake, delivers the tool's result
// (`ui/notifications/tool-result`), and sends a chat message (`ui/message`).
// This module wraps it in two tiny functions so the views stay host-agnostic:
//
//   onData(handler)   — the tool's structuredContent, once the host sends it
//   sendMessage(text) — a user turn back into the chat, to run the next tool
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
  version: "0.10.1", // x-release-please-version
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
 * call it before the first `onData`/`sendMessage`. Defaults to a generic name.
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
