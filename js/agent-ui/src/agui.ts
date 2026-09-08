/** The routes AG-UI does not cover.
 *
 * The run itself goes through `@ag-ui/client`'s `HttpAgent` — see `chat.tsx`.
 * This is the other half: the reads that exist *because* the stream carries a
 * description of a value rather than the value, plus the one route that
 * describes the deployment rather than a conversation.
 */
import { apiUrl } from "./config";

import type { Declared } from "./credentials";

/** One session-state value, as the state route returns it. */
export type StateValue = {
  key: string;
  tool?: string;
  /** Where each argument of the producing call came from: another state key,
   * or "model" for one the model wrote itself. */
  inputs?: Record<string, string> | null;
  seq?: number | null;
  /** The turn it was read at, or `null` for "as state stands now". */
  turn: number | null;
  value: unknown;
};

/** What the stream says about one stored value, per key.
 *
 * Never the value: it is in session state because it was too big for the
 * transcript. Enough to decide whether to fetch it from
 * `GET /threads/{id}/state/{key}`.
 */
export type StateSummary = Record<
  string,
  { tool?: string; bytes?: number; inputs?: Record<string, string> }
>;

/** One session-state value in full — the payload the stream left out.
 *
 * `STATE_SNAPSHOT` carries `{tool, bytes, inputs}` per key. This is the route a
 * client follows once it has decided it wants the 39 kB geometry, and it is
 * outside the AG-UI vocabulary entirely: the protocol has a state channel but
 * no notion of a value too large to put on it.
 *
 * `turn` asks for the value **as it stood at the end of that turn** rather
 * than now. State holds one value per key, so a key a later turn overwrote
 * would otherwise read back as the later value — which is the wrong answer to
 * "what did this turn run on". The checkpoints have kept every version.
 */
export async function readState(
  threadId: string,
  key: string,
  turn?: number,
): Promise<StateValue> {
  const at = turn === undefined ? "" : `?turn=${turn}`;
  const response = await fetch(apiUrl(`/threads/${threadId}/state/${key}${at}`));
  if (!response.ok) {
    // The API's own wording, which distinguishes a turn that never existed
    // (404) from one the checkpointer has pruned (410) — a difference worth
    // showing a reader rather than flattening into "failed".
    const detail = await response
      .json()
      .then((body) => body?.detail)
      .catch(() => null);
    throw new Error(detail || `${response.status} ${response.statusText}`);
  }
  return (await response.json()) as StateValue;
}

/** A thread's messages, for a client that reloaded.
 *
 * The conversation, and nothing around it: receipts, views and the rest are
 * activities, and the server does not rebuild past turns' activities. So a
 * restored thread shows what was said but not where each tool's arguments came
 * from — see the README.
 */
export async function readThread(threadId: string) {
  const response = await fetch(apiUrl(`/threads/${threadId}`));
  // 404 is the ordinary answer for a thread id that has never run, which is
  // what a hand-edited URL produces. The caller starts fresh instead.
  if (response.status === 404) return null;
  if (!response.ok) throw new Error(`${response.status}`);
  return (await response.json()) as {
    threadId: string;
    messages: { id: string; role: string; content?: string | null }[];
    state: StateSummary;
  };
}

/** A thread's turns, and what session state held at the end of each.
 *
 * The live client builds turns from the events as they arrive; this is how one
 * that reloaded gets them back, and it is the route the panel's per-turn view
 * is really made of.
 */
export async function readTurns(threadId: string) {
  const response = await fetch(apiUrl(`/threads/${threadId}/turns`));
  if (!response.ok) throw new Error(`${response.status}`);
  return (await response.json()) as {
    threadId: string;
    turns: number;
    total: number;
    history: {
      turn: number;
      question: string;
      checkpointId: string | null;
      state: StateSummary;
    }[];
  };
}

/** What the agent is connected to, before anyone has asked it anything.
 *
 * The opening screen is built from this rather than from a sentence written
 * into the client: which toolsets a deployment connected is exactly what the
 * client cannot know, and a hardcoded greeting is wrong in every repository
 * that installs this one.
 *
 * It carries the credential headers too. A header no toolset declared is
 * dropped rather than forwarded, so asking a visitor for one is only ever
 * right when the server has said it wants it.
 */
export async function readConnections(): Promise<{
  toolsets: { name: string; credentials: { header: string; supplied: boolean }[] }[];
  tools: { name: string; description: string }[];
}> {
  const response = await fetch(apiUrl("/connections"));
  if (!response.ok) throw new Error(`${response.status}`);
  return await response.json();
}

/** The credential headers, folded across toolsets into one row per header.
 *
 * Two toolsets can want the same header, and a visitor types it once.
 */
export function declaredCredentials(
  toolsets: { name: string; credentials: { header: string; supplied: boolean }[] }[],
): Declared[] {
  const found = new Map<string, Declared>();
  for (const toolset of toolsets) {
    for (const credential of toolset.credentials) {
      const existing = found.get(credential.header);
      if (existing) {
        existing.toolsets.push(toolset.name);
        // Supplied for one toolset is supplied for all of them: the value
        // comes from the server's environment, which is not per-toolset.
        existing.supplied ||= credential.supplied;
      } else {
        found.set(credential.header, {
          header: credential.header,
          supplied: credential.supplied,
          toolsets: [toolset.name],
        });
      }
    }
  }
  return [...found.values()];
}
