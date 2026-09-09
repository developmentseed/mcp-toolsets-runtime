/** What the deployment gets to say about this client, read out of the page.
 *
 * `mcp_agent_api.ui` rewrites the contents of a `<script type="application/json">`
 * element in `index.html` at mount time. Reading it from the document rather
 * than fetching it means the first render already has the real title and the
 * real example questions — a client that fetched its own configuration would
 * render placeholder text first, and every deployment would wear the flash.
 *
 * The values below are what `npm run dev` uses, where nothing has rewritten
 * anything: `/api` is the path the dev server proxies to the service.
 */
/** Where a credential the visitor types is kept between renders.
 *
 * `local` outlives the browser closing, `session` is forgotten with the tab,
 * and `none` never leaves memory. Which one is right depends on whose machine
 * the page is opened on, which only the deployment knows — see `UiConfig` in
 * `mcp_agent_api/ui.py`.
 */
export type CredentialStore = "local" | "session" | "none";

const STORES: readonly string[] = ["local", "session", "none"];

export type Config = {
  /** Base the six routes hang off, as the browser reaches them. */
  api: string;
  title: string;
  tagline: string;
  /** The opening paragraph. Empty means: describe what is connected. */
  greeting: string;
  /** Questions offered as buttons before the conversation starts. */
  examples: string[];
  /** A CSS colour, or empty to keep the client's own. */
  accent: string;
  credentials: CredentialStore;
};

const DEFAULTS: Config = {
  api: "/api",
  title: "MCP Toolsets",
  tagline: "",
  greeting: "",
  examples: [],
  accent: "",
  credentials: "local",
};

function read(): Config {
  const element = document.getElementById("mcp-agent-ui-config");
  if (!element?.textContent) return DEFAULTS;
  try {
    const parsed = JSON.parse(element.textContent) as Partial<Config>;
    // Field by field over the defaults: a server that adds a field this build
    // does not know about must not remove one it does.
    const merged = { ...DEFAULTS, ...parsed };
    // The server refuses to start on a value outside the set, so reaching this
    // means the page was written by something else. Fall back rather than
    // trust it: `credentials` decides where a key is put, and an unreadable
    // answer is not grounds for choosing the most durable option.
    if (!STORES.includes(merged.credentials)) {
      console.warn(
        `mcp-agent-ui: unknown credential store ${merged.credentials}, using ${DEFAULTS.credentials}`,
      );
      merged.credentials = DEFAULTS.credentials;
    }
    return merged;
  } catch {
    // A malformed configuration is a deployment bug, and a chat that still
    // works with default text is a better way to find out than a blank page.
    console.warn("mcp-agent-ui: unreadable configuration, using defaults");
    return DEFAULTS;
  }
}

export const config: Config = read();

/** One of the API's routes, as this deployment serves it. */
export function apiUrl(path: string): string {
  return `${config.api}${path}`;
}
