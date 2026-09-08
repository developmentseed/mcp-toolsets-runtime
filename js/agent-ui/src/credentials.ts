/** Credential headers, kept in the browser and sent with every run.
 *
 * A toolset can declare that its tools need a header — an API key for the
 * service behind them — and the agent forwards only headers a toolset actually
 * declared. `GET /connections` says which those are, and whether the server
 * already holds a value from its own environment; this is the other half, for
 * the ones it does not.
 *
 * `localStorage` rather than a cookie or the URL: the value is the visitor's
 * own key, it should not travel to the server except as the header it is for,
 * and it should survive a reload the way the thread does. That does mean it is
 * readable by anything running on this origin, which is the same trust the
 * bundled Chainlit host's settings panel asks for.
 */
const STORE = "mcp-agent-ui:credentials";

export type Declared = {
  /** The header's name, which is what the API expects it under. */
  header: string;
  /** The server already has one, so nothing needs to be asked for. */
  supplied: boolean;
  /** The toolsets that declared it. */
  toolsets: string[];
};

export function load(): Record<string, string> {
  try {
    const raw = localStorage.getItem(STORE);
    return raw ? (JSON.parse(raw) as Record<string, string>) : {};
  } catch {
    // A private window, or storage the browser refuses. The chat still runs;
    // the headers are simply not remembered between reloads.
    return {};
  }
}

export function save(values: Record<string, string>): void {
  try {
    // Empty values are absences, not credentials: keeping them would send an
    // empty header, which reads to a toolset as a wrong key rather than none.
    const kept = Object.fromEntries(
      Object.entries(values).filter(([, value]) => value.trim()),
    );
    localStorage.setItem(STORE, JSON.stringify(kept));
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
