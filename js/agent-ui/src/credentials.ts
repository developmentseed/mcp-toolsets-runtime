/** Credential headers, kept in the browser and sent with every run.
 *
 * A toolset can declare that its tools need a header — an API key for the
 * service behind them — and the agent forwards only headers a toolset actually
 * declared. `GET /connections` says which those are, and whether the server
 * already holds a value from its own environment; this is the other half, for
 * the ones it does not.
 *
 * Browser storage rather than a cookie or the URL: the value is the visitor's
 * own key, and it should not travel to the server except as the header it is
 * for. *Which* store is the deployment's to choose (`config.credentials`),
 * because the answer depends on whose machine the page is opened on — a laptop
 * with one owner and a workstation in a public building want different things,
 * and the client cannot tell them apart. Whichever is used, the value is
 * readable by anything running on this origin.
 */
import { config } from "./config";

const STORE = "mcp-agent-ui:credentials";

export type Declared = {
  /** The header's name, which is what the API expects it under. */
  header: string;
  /** The server already has one, so nothing needs to be asked for. */
  supplied: boolean;
  /** The toolsets that declared it. */
  toolsets: string[];
};

/** The store this deployment chose, or `null` to keep nothing.
 *
 * Reading the property is itself what throws where a browser blocks site data,
 * so every caller does this inside its own `try`.
 */
function chosen(): Storage | null {
  if (config.credentials === "none") return null;
  return config.credentials === "session" ? sessionStorage : localStorage;
}

/** Drop what the stores this deployment does *not* use are still holding.
 *
 * Tightening the setting has to reach the browsers that were there before it:
 * without this, moving a deployment to `session` or `none` leaves every
 * existing visitor's key sitting in `localStorage`, which is the one thing
 * those settings exist to prevent. Cheap, and it runs once per page.
 */
function forgetUnused(): void {
  try {
    const keep = chosen();
    for (const store of [localStorage, sessionStorage]) {
      if (store !== keep) store.removeItem(STORE);
    }
  } catch {
    /* nothing to do: see load() */
  }
}

export function load(): Record<string, string> {
  forgetUnused();
  try {
    const raw = chosen()?.getItem(STORE);
    return raw ? (JSON.parse(raw) as Record<string, string>) : {};
  } catch {
    // A private window, or storage the browser refuses. The chat still runs;
    // the headers are simply not remembered between reloads.
    return {};
  }
}

export function save(values: Record<string, string>): void {
  try {
    const store = chosen();
    if (store === null) return; // held in the page's own state and nowhere else
    // Empty values are absences, not credentials: keeping them would send an
    // empty header, which reads to a toolset as a wrong key rather than none.
    const kept = Object.fromEntries(
      Object.entries(values).filter(([, value]) => value.trim()),
    );
    store.setItem(STORE, JSON.stringify(kept));
  } catch {
    /* nothing to do: see load() */
  }
}

/** Headers to send, which is the stored ones a toolset actually asked for. */
export function headersFor(
  declared: Declared[],
  values: Record<string, string>,
): Record<string, string> {
  const wanted = new Set(declared.map((each) => each.header.toLowerCase()));
  return Object.fromEntries(
    Object.entries(values).filter(
      ([header, value]) => value.trim() && wanted.has(header.toLowerCase()),
    ),
  );
}

/** The ones a visitor still has to supply. */
export function outstanding(
  declared: Declared[],
  values: Record<string, string>,
): Declared[] {
  return declared.filter(
    (each) => !each.supplied && !values[each.header]?.trim(),
  );
}
