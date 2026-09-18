/** What the client does when the API stops recognising it.
 *
 * A deployment that puts a session in front of this page — an OIDC proxy that
 * signs the visitor in and adds their token to each request — outlives the
 * page's first load. An access token measured in minutes against a
 * conversation that can run for an hour means the session lapses *between*
 * fetches, and from then on every route answers `401`.
 *
 * The client cannot sign anyone in: it holds no credential and knows nothing
 * about what sits in front of it. What it can do is reload, because a page
 * navigation is the one request the proxy answers with its own redirect to the
 * sign-in page. Without this, a lapsed session reads as a broken chat — a
 * thread that silently starts empty, or a run that ends in `client error`.
 */

/** Where the time of the last reload for a `401` is kept, for the tab. */
const RELOADED_AT = "mcp-agent-ui:reloaded-for-401";

/** How long after one reload another `401` is taken as not a lapsed session.
 *
 * A page whose routes answer `401` straight after a fresh load was not fixed
 * by reloading, and reloading again would loop. Past this, a `401` is a new
 * lapse and gets its reload.
 */
const GRACE_MS = 30_000;

/** Reload, unless this tab has just reloaded for a `401` already.
 *
 * When it does not — a `401` that a reload did not cure, or storage the
 * browser will not let the page use — the caller's error handling runs as it
 * would have, which at least shows the status.
 */
function reloadOnce(): void {
  try {
    const last = Number(sessionStorage.getItem(RELOADED_AT) ?? 0);
    if (Date.now() - last < GRACE_MS) return;
    sessionStorage.setItem(RELOADED_AT, String(Date.now()));
  } catch {
    // Without somewhere to record it, a reload cannot tell whether it is the
    // second, so it does not happen.
    return;
  }
  location.reload();
}

/** `fetch`, for the API's routes: a `401` reloads the page.
 *
 * The response is still returned, so the caller's own handling runs while the
 * reload is under way; nothing waits on it.
 */
export async function apiFetch(
  input: RequestInfo | URL,
  init?: RequestInit,
): Promise<Response> {
  const response = await fetch(input, init);
  if (response.status === 401) reloadOnce();
  return response;
}
