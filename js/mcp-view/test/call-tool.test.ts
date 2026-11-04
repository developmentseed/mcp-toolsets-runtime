import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// `callTool` against the real SDK and a host that speaks the wire, rather than
// a mocked `App`: what is under test is the request the host receives and how
// its answer comes back, and a mock of `callServerTool` would only test that
// this module calls it.
//
// The host is a few lines of raw JSON-RPC — the same messages the agent UI's
// `View` answers in `js/agent-ui/src/chat.tsx`. The SDK's transport posts to
// `window.parent` and listens on `window` for messages whose `source` is
// `window.parent`, so a stand-in for each is all the "iframe" needs.

type Rpc = { jsonrpc: "2.0"; id?: number | string; method?: string; params?: any };
type Answer = { result: unknown } | { error: { code: number; message: string } };

let received: Rpc[] = [];
let answerToolCall: (request: Rpc) => Answer;

function installHost() {
  const view = new EventTarget();
  const parent = {
    postMessage(message: Rpc) {
      received.push(message);
      const reply = (body: Answer) =>
        // Async, as a real postMessage is: an answer delivered inside the
        // request's own send would arrive before the SDK is waiting for it.
        queueMicrotask(() => {
          const event = new Event("message");
          Object.assign(event, {
            data: { jsonrpc: "2.0", id: message.id, ...body },
            source: parent,
          });
          view.dispatchEvent(event);
        });
      if (message.method === "ui/initialize") {
        reply({
          result: {
            protocolVersion: "2026-01-26",
            hostInfo: { name: "fake-host", version: "0.0.0" },
            hostCapabilities: { serverTools: {} },
            hostContext: {},
          },
        });
      } else if (message.method === "tools/call") {
        reply(answerToolCall(message));
      }
    },
  };
  vi.stubGlobal("window", Object.assign(view, { parent, innerWidth: 640 }));
  // What the SDK's auto-resize touches once connected (see `AUTO_RESIZE`):
  // enough of a document to measure, and an observer that never fires.
  const element = { style: {}, getBoundingClientRect: () => ({ height: 120 }) };
  vi.stubGlobal("document", { documentElement: element, body: element });
  vi.stubGlobal(
    "ResizeObserver",
    class {
      observe() {}
      disconnect() {}
    },
  );
  vi.stubGlobal("requestAnimationFrame", (callback: () => void) =>
    queueMicrotask(callback),
  );
}

beforeEach(() => {
  // The bridge holds module-level singletons; reset between tests.
  vi.resetModules();
  received = [];
  answerToolCall = () => ({ result: { content: [] } });
  installHost();
});

afterEach(() => {
  vi.unstubAllGlobals();
});

const calls = () => received.filter((message) => message.method === "tools/call");

describe("callTool", () => {
  it("sends one tools/call with the arguments and resolves with the result", async () => {
    const result = {
      content: [{ type: "text", text: "Lisboa" }],
      structuredContent: { place_id: "pt-lisboa", boundary: { type: "Polygon" } },
    };
    answerToolCall = () => ({ result });
    const { callTool } = await import("../src/index");

    const answer = await callTool("places_get", {
      place_id: "pt-lisboa",
      include: ["boundary"],
    });

    expect(calls()).toHaveLength(1);
    expect(calls()[0].params).toMatchObject({
      name: "places_get",
      arguments: { place_id: "pt-lisboa", include: ["boundary"] },
    });
    expect(answer).toMatchObject(result);
  });

  it("rejects when the host answers an error", async () => {
    answerToolCall = () => ({
      error: { code: -32603, message: "places_get is not visible to views" },
    });
    const { callTool } = await import("../src/index");

    await expect(callTool("places_get", {})).rejects.toThrow(
      "places_get is not visible to views",
    );
  });

  it("rejects when the tool itself failed, with the result on the error", async () => {
    const result = {
      content: [{ type: "text", text: "no place pt-nowhere" }],
      isError: true,
    };
    answerToolCall = () => ({ result });
    const { callTool, ToolError } = await import("../src/index");

    const failure = await callTool("places_get", { place_id: "pt-nowhere" }).catch(
      (error: unknown) => error,
    );

    expect(failure).toBeInstanceOf(ToolError);
    expect((failure as InstanceType<typeof ToolError>).message).toBe(
      "no place pt-nowhere",
    );
    expect((failure as InstanceType<typeof ToolError>).result).toMatchObject(result);
  });

  it("shares one connection with onData and sendMessage", async () => {
    const { callTool, onData } = await import("../src/index");
    onData(() => {});
    await callTool("places_get", {});
    await callTool("places_get", {});

    // One handshake, however many calls: a second `ui/initialize` would be a
    // second App the host has to tell apart from the first.
    expect(received.filter((m) => m.method === "ui/initialize")).toHaveLength(1);
    expect(calls()).toHaveLength(2);
  });
});
