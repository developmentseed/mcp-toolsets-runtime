import { beforeEach, describe, expect, it, vi } from "vitest";

// Capture every App the bridge constructs so we can drive the host side.
const { instances } = vi.hoisted(() => ({ instances: [] as any[] }));

vi.mock("@modelcontextprotocol/ext-apps", () => {
  class App {
    ontoolresult: ((p: { structuredContent?: unknown }) => void) | null = null;
    sendMessage = vi.fn();
    connect = vi.fn().mockResolvedValue(undefined);
    constructor(public info: { name: string; version: string }) {
      instances.push(this);
    }
  }
  return { App };
});

beforeEach(() => {
  // The bridge holds module-level singletons; reset between tests.
  vi.resetModules();
  instances.length = 0;
});

const flush = () => new Promise((resolve) => setTimeout(resolve, 0));

describe("mcp-view host bridge", () => {
  it("delivers the tool's structuredContent to the onData handler", async () => {
    const { onData } = await import("../src/index");
    const received: unknown[] = [];
    onData((payload) => received.push(payload));

    // one App is created lazily and connect() runs the handshake
    expect(instances).toHaveLength(1);
    expect(instances[0].connect).toHaveBeenCalledOnce();

    instances[0].ontoolresult?.({ structuredContent: { hello: "world" } });
    expect(received).toEqual([{ hello: "world" }]);

    // null/absent structuredContent is ignored
    instances[0].ontoolresult?.({ structuredContent: null });
    instances[0].ontoolresult?.({});
    expect(received).toHaveLength(1);
  });

  it("sends a user text turn back to the host", async () => {
    const { onData, sendMessage } = await import("../src/index");
    onData(() => {});
    sendMessage("next please");
    await flush();

    expect(instances[0].sendMessage).toHaveBeenCalledWith({
      role: "user",
      content: [{ type: "text", text: "next please" }],
    });
  });

  it("reports the configured app identity to the host", async () => {
    const { configure, onData } = await import("../src/index");
    configure({ name: "custom-view", version: "9.9.9" });
    onData(() => {});
    expect(instances[0].info).toEqual({ name: "custom-view", version: "9.9.9" });
  });
});
