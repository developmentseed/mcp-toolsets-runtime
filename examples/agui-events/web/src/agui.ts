/** The routes AG-UI does not cover.
 *
 * The run itself goes through `@ag-ui/client`'s `HttpAgent` — see `chat.tsx`.
 * This is the other half: the reads that exist *because* the stream carries a
 * description of a value rather than the value.
 */

/** One session-state value in full — the payload the stream left out.
 *
 * `STATE_SNAPSHOT` carries `{kind, tool, bytes}` per key. This is the route a
 * client follows once it has decided it wants the 39 kB geometry, and it is
 * outside the AG-UI vocabulary entirely: the protocol has a state channel but
 * no notion of a value too large to put on it.
 */
export async function readState(threadId: string, key: string) {
  const response = await fetch(`/api/threads/${threadId}/state/${key}`);
  if (!response.ok) throw new Error(`${response.status}`);
  return (await response.json()) as {
    key: string;
    kind: string | null;
    value: unknown;
  };
}
