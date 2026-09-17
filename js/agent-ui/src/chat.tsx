import {
  buildResumeArray,
  HttpAgent,
  type Interrupt,
  type Message,
  type ResumeEntry,
} from "@ag-ui/client";
import { Fragment, useEffect, useMemo, useRef, useState } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";

import {
  declaredCredentials,
  readConnections,
  readState,
  readThread,
  readTurns,
} from "./agui";
import { apiUrl, config, type CredentialStore } from "./config";
import {
  type Declared,
  headersFor,
  load as loadCredentials,
  outstanding,
  save as saveCredentials,
} from "./credentials";

/** What `GET /connections` says this deployment is. */
type Connected = Awaited<ReturnType<typeof readConnections>>;

/** Session state as the stream describes it: no payloads, one line per key. */
type StateEntry = {
  tool?: string;
  bytes?: number;
  seq?: number;
  /** Parameter -> the state key it came from, or "model". Absent when the
   * producing call took no arguments. */
  inputs?: Record<string, string>;
  /** The turn this value was written in. */
  turn?: number;
  /** How many turns hold a value for this key, this one included. Absent when
   * it is one, so its presence *is* the signal: a key shows one value, and
   * without this a panel cannot say an earlier turn holds another. Each turn
   * it counts is one `?turn=N` can fetch. */
  turnsWritten?: number;
};

/** Every argument of the call that produced an entry, in a stable order.
 *
 * Both halves, not just the model's. The listing a *refusal* shows the model
 * names only what the model wrote, because every line there costs context and
 * "this one came from state" is the unremarkable case. A panel has neither
 * constraint and a reader has no memory of the call, which by now has scrolled
 * away — dropping half the record here would leave the chain unreadable from
 * the one surface built to show it.
 *
 * One level, deliberately: this reads the call that produced the entry and
 * follows nothing further. The reader follows it by clicking, since every
 * state-sourced input names a key that is itself a row in this panel. */
function producedBy(entry: StateEntry): [string, string][] {
  // Model-authored first: it is the caveat, and a reader scanning a column of
  // these is looking for it rather than for the unremarkable half.
  return Object.entries(entry.inputs ?? {}).sort(
    ([a, from], [b, other]) =>
      Number(other === "model") - Number(from === "model") ||
      a.localeCompare(b),
  );
}

/** `state key -> the arguments of the call that produced it`.
 *
 * The value a model wrote is deliberately *not* on the wire: `inputs` carries
 * parameter names and state keys and nothing else, because an argument can be
 * arbitrarily large and the state channel is re-sent every turn. A client does
 * not need it to be — it already holds the call. `state.published` names the
 * `toolCallId`, the transcript holds that call, and this is the join.
 *
 * Read across every message rather than one turn's, so a key published three
 * turns ago still resolves.
 */
function producedArguments(
  all: readonly Message[],
): Record<string, Record<string, unknown>> {
  const calls: Record<string, Record<string, unknown>> = {};
  for (const message of all) {
    for (const call of (message as any).toolCalls ?? []) {
      try {
        calls[call.id] = JSON.parse(call.function.arguments || "{}");
      } catch {
        calls[call.id] = {};
      }
    }
  }
  const found: Record<string, Record<string, unknown>> = {};
  for (const message of all) {
    if ((message as any).activityType !== "state.published") continue;
    const content = (message as any).content;
    const args = calls[content?.toolCallId];
    if (!args) continue;
    for (const key of Object.values<string>(content?.published ?? {})) {
      found[key] = args;
    }
  }
  return found;
}

/** How much of a model-authored value fits on a line before it is folded. */
const INLINE = 56;

/** Whether a value is a string holding JSON — an object or an array.
 *
 * Providers differ on whether a structured argument arrives as an object or
 * as the text of one, and the server coerces either. Rendering the text form
 * with `JSON.stringify` escapes it a second time, which helps nobody. Only
 * objects and arrays qualify: a model that wrote the string "4" wrote a
 * string, and quoting it is the honest rendering. */
function isJsonText(value: unknown): value is string {
  if (typeof value !== "string") return false;
  try {
    const parsed: unknown = JSON.parse(value);
    return typeof parsed === "object" && parsed !== null;
  } catch {
    return false;
  }
}

/** A value as the tool received it, indented. */
function pretty(value: unknown): string {
  return isJsonText(value)
    ? JSON.stringify(JSON.parse(value), null, 2)
    : JSON.stringify(value, null, 2);
}

/** A value with every stringified-JSON field inside it parsed back.
 *
 * `isJsonText` spots the trick for one value; a tool call's arguments can
 * nest it at any depth (`{"terms": "[\"fire\"]"}`), because a provider that
 * encodes one structured argument as text does it wherever one appears. So
 * this walks the whole tree rather than the top level, and a reader sees the
 * list the model wrote instead of the escaping it arrived in.
 */
function unescaped(value: unknown): unknown {
  if (isJsonText(value)) return unescaped(JSON.parse(value));
  if (Array.isArray(value)) return value.map(unescaped);
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value).map(([key, entry]) => [key, unescaped(entry)]),
    );
  }
  return value;
}

/** A call's own arguments, indented and unescaped.
 *
 * Malformed JSON falls back to the raw string: a provider that wrote it badly
 * is a thing to see rather than to hide behind an empty object, and the tool's
 * own result says what was wrong with it.
 */
function prettyArgs(argumentsJson: string): string {
  try {
    return JSON.stringify(unescaped(JSON.parse(argumentsJson || "{}")), null, 2);
  } catch {
    return argumentsJson;
  }
}

/** The value the model wrote, shown whole or folded.
 *
 * The case worth seeing is the expensive one — a model inlining a large
 * literal into an untagged parameter — and that is exactly the case that
 * would fill the panel. `<details>` because collapsing is what the element is
 * for, and the keyboard and screen-reader behaviour comes with it.
 */
function Wrote({ value }: { value: unknown }) {
  if (value === undefined) {
    return <span className="authored">written by the model</span>;
  }
  // Quoted, so a string reads as a value rather than as a second identifier
  // beside the parameter — except where it is the text of an object, which
  // `JSON.stringify` would escape twice.
  const whole = isJsonText(value) ? value : JSON.stringify(value);
  if (whole.length <= INLINE) {
    return (
      <>
        <code className="wrote">{whole}</code>
        <span className="authored"> · written by the model</span>
      </>
    );
  }
  return (
    <details className="folded">
      <summary>
        <code className="wrote">{whole.slice(0, INLINE)}…</code>
        <span className="authored"> · written by the model</span>
        <span className="dim"> · {whole.length} chars</span>
      </summary>
      <pre>{pretty(value)}</pre>
    </details>
  );
}

/** A state key that may wrap, preferring its own separators.
 *
 * `<toolset>/<tool>/<field>` has no spaces, so a narrow column breaks it
 * mid-word — `datas / ets` — unless it is told where the seams are. `<wbr>`
 * marks them; `overflow-wrap: break-word` remains the fallback for a segment
 * too long to fit on its own. */
function Key({ value }: { value: string }) {
  const parts = value.split("/");
  return (
    <>
      {parts.map((part, index) => (
        <Fragment key={index}>
          {index > 0 ? (
            <>
              /<wbr />
            </>
          ) : null}
          {part}
        </Fragment>
      ))}
    </>
  );
}

/** Every key the thread holds, which is what the state channel describes. */
type Snapshot = Record<string, StateEntry>;

/**
 * Key inside AG-UI's `state` object holding session-state metadata. The rest
 * of that object belongs to the client, and every operation the server sends
 * names a path inside this one key — so whatever a client keeps beside it
 * survives a run untouched.
 */
const TOOL_STATE = "toolState";

/** One JSON Patch operation, as `STATE_DELTA` carries them. */
type Operation = { op: string; path: string; value?: StateEntry };

/**
 * Our `toolState`, moved on by one delta.
 *
 * Short because the server only ever sends two shapes: `add` of the whole
 * namespace, which opens every run and is the resynchronisation point, and
 * `add`/`remove` of one key under it. RFC 6901 escaping has to be undone —
 * state keys are `toolset/name`, and `/` is the pointer's own separator.
 */
function applyDelta(state: Snapshot, delta: Operation[]): Snapshot {
  let next = state;
  for (const { op, path, value } of delta) {
    if (path === `/${TOOL_STATE}`) {
      next = { ...((value ?? {}) as unknown as Snapshot) };
      continue;
    }
    const key = path
      .slice(`/${TOOL_STATE}/`.length)
      .replace(/~1/g, "/")
      .replace(/~0/g, "~");
    next = { ...next };
    if (op === "remove") delete next[key];
    else if (value) next[key] = value;
  }
  return next;
}

/** The activity messages naming one tool call — `mcp.view` and both state
 * halves alike.
 *
 * A turn's `published` origins reach only `state.published`, because that is
 * the one activity keyed by what it wrote. A view is published under no key
 * and a `state.consumed` writes none, so neither is in that map. Every
 * activity carries the `toolCallId` it belongs to, and an activity is a
 * message, so reading the messages is what covers all three.
 */
function activitiesOf(all: readonly Message[], toolCallId: string): string[] {
  return all
    .filter(
      (message) =>
        (message as any).role === "activity" &&
        (message as any).content?.toolCallId === toolCallId,
    )
    .map((message) => String(message.id));
}

/** Where a key came from, read off the `state.published` that announced it.
 *
 * The announcing activity is not held here: both hover paths resolve
 * activities through `activitiesOf`, which finds all three rather than only
 * the one that named a key. */
type Origin = { toolCallId: string; tool: string };

/** One question and what it did.
 *
 * The stream has no turn boundary in it — `RUN_STARTED` and `RUN_FINISHED`
 * bracket a run, but a client that reloads mid-thread never saw them. So a
 * turn is marked here, at the point the question is asked, and `from` is where
 * its messages begin.
 */
type Turn = {
  n: number;
  question: string;
  /** The question's own message id, so scrolling to a turn needs no index. */
  questionId: string;
  from: number;
  /** Cumulative, not a delta: everything in state as of this turn. */
  state: Snapshot;
  /** Only the keys *this* turn wrote, and which call wrote each. */
  published: Record<string, Origin>;
};

/** What a hover has lit up, on all three sides at once. */
type Linked = { keys: string[]; calls: string[]; activities: string[] };

const NOTHING: Linked = { keys: [], calls: [], activities: [] };

/** The line a minimal client prints. Every activity carries one, generated by
 * the `mcp_agent.host` helpers, so the wire and this client cannot drift. A
 * client with opinions reads the fields beside it instead. */
function shown(content: any): string {
  if (typeof content?.display === "string") return content.display;
  // state.consumed carries one receipt per parameter, each with its own line.
  if (content?.received) {
    return Object.entries<any>(content.received)
      .map(([parameter, receipt]) => `${parameter} ${receipt.display}`)
      .join("   ·   ");
  }
  return JSON.stringify(content);
}

/** A tool result on one line. The whole thing is in the thread route if a
 * client wants it; this is the glance. */
function summarise(result: string): string {
  const line = result.replace(/\s+/g, " ").trim();
  return line.length > 140 ? `${line.slice(0, 140)}…` : line;
}

function bytes(size?: number): string {
  if (size === undefined) return "";
  return size < 1024 ? `${size} B` : `${(size / 1024).toFixed(1)} kB`;
}

/** What one answer to a question is: a choice, or no answer at all. */
type Response =
  { status: "resolved"; payload: unknown } | { status: "cancelled" };

/** An `interrupt` call, drawn as the question it asks.
 *
 * Drawn from the call's own arguments rather than from the interrupt, so the
 * question still reads after it is answered and after a reload. The interrupt
 * only says whether it is still open, and `result` — the tool message for this
 * call — is the answer once there is one.
 *
 * `given` is this question's answer while it waits, disabled, for the others:
 * a run that asked two questions takes both answers in one resume. It is held
 * by the chat rather than here, so a resume the server refuses clears it and
 * the buttons come back.
 */
function Question({
  call,
  open,
  given,
  result,
  onAnswer,
}: {
  call: any;
  open: Interrupt | undefined;
  given: Response | undefined;
  result: string | undefined;
  onAnswer: (interruptId: string, response: Response) => void;
}) {
  const [picked, setPicked] = useState<string[]>([]);
  let args: {
    question?: string;
    options?: { value: string; label: string }[];
    multiple?: boolean;
  } = {};
  try {
    args = JSON.parse(call.function.arguments || "{}");
  } catch {
    // A call the model wrote badly draws as an empty question; the tool's own
    // result says what was wrong with it.
  }
  const active = Boolean(open) && result === undefined && !given;
  const answer = (response: Response) => {
    if (open) onAnswer(open.id, response);
  };
  return (
    <div className="question">
      <p>{args.question}</p>
      <div className="choices">
        {(args.options ?? []).map((option) => (
          <button
            key={option.value}
            className={picked.includes(option.value) ? "choice on" : "choice"}
            disabled={!active}
            aria-pressed={
              args.multiple ? picked.includes(option.value) : undefined
            }
            onClick={() =>
              args.multiple
                ? setPicked((held) =>
                    held.includes(option.value)
                      ? held.filter((value) => value !== option.value)
                      : [...held, option.value],
                  )
                : answer({
                    status: "resolved",
                    payload: { choice: option.value },
                  })
            }
          >
            {option.label}
          </button>
        ))}
      </div>
      {active ? (
        <div className="choices">
          {args.multiple ? (
            <button
              disabled={picked.length === 0}
              onClick={() =>
                answer({ status: "resolved", payload: { choice: picked } })
              }
            >
              send
            </button>
          ) : null}
          <button
            className="link"
            onClick={() => answer({ status: "cancelled" })}
          >
            skip
          </button>
        </div>
      ) : null}
      {result !== undefined ? (
        <p className="dim">{result}</p>
      ) : given ? (
        <p className="dim">waiting for the other answers</p>
      ) : null}
    </div>
  );
}

/** ext-apps `LATEST_PROTOCOL_VERSION`, which the view's SDK checks. */
const UI_PROTOCOL_VERSION = "2026-01-26";

/** How tall a view is allowed to be, in pixels.
 *
 * `initial` is what it gets before it has said anything, so a view whose SDK
 * never reports a size still renders in a usable box. The floor is a line of
 * text: a view whose answer is one sentence — an empty result, a failed call —
 * should take one sentence's room rather than a panel's. The ceiling is what
 * keeps a 500-row table from owning the scrollback; past it the frame scrolls
 * inside itself.
 */
const HEIGHT = { initial: 240, min: 44, max: 1000 };

/** A tool's `ui://` bundle, mounted and driven over MCP Apps `ui/*`.
 *
 * This is the host end of the same JSON-RPC-over-postMessage protocol Claude,
 * ChatGPT, Goose and VS Code implement — the view's end is the standard
 * `@modelcontextprotocol/ext-apps` SDK, wrapped by `@developmentseed/mcp-view`.
 * The exchange is four messages:
 *
 *     view "ui/initialize"                 -> host info and capabilities
 *     view "ui/notifications/initialized"  -> host pushes tool-input, then
 *                                             tool-result (that order is
 *                                             required, and the view only
 *                                             needs the second)
 *     view "ui/message"                    -> a turn back into the chat
 *
 * A fifth message runs the other way and is not part of that sequence:
 *
 *     view "ui/notifications/size-changed" -> the height the view wants
 *
 * The SDK sends it on its own — `@modelcontextprotocol/ext-apps` observes the
 * document and reports every change, so a view built on
 * `@developmentseed/mcp-view` needs no code for this. A host may act on it or
 * ignore it; this one acts, within `HEIGHT`.
 *
 * `src` is the view route rather than `srcdoc`, so the bundle really is fetched
 * over HTTP — that route exists because a bundle is hundreds of kilobytes and
 * does not change between turns. `allow-scripts` without `allow-same-origin`
 * gives the frame an opaque origin: it can run, and it can reach nothing.
 */
function View({
  uri,
  data,
  onMessage,
}: {
  uri: string;
  data: unknown;
  onMessage: (text: string) => void;
}) {
  const frame = useRef<HTMLIFrameElement>(null);
  // What the view has asked for, starting at what it gets before it has asked.
  const [height, setHeight] = useState(HEIGHT.initial);
  const [toolset, view] = uri.replace("ui://", "").split("/");
  // The turn's data never changes once rendered, but the listener is mounted
  // once — so it reads through a ref rather than closing over the first value.
  const latest = useRef(data);
  latest.current = data;
  const reply = useRef(onMessage);
  reply.current = onMessage;

  useEffect(() => {
    const iframe = frame.current;
    if (!iframe) return;
    const post = (message: unknown) =>
      iframe.contentWindow?.postMessage(message, "*");

    function handle(event: MessageEvent) {
      if (event.source !== iframe?.contentWindow) return;
      const message = event.data;
      if (!message || message.jsonrpc !== "2.0") return;

      if (message.method === "ui/initialize") {
        post({
          jsonrpc: "2.0",
          id: message.id,
          result: {
            protocolVersion: UI_PROTOCOL_VERSION,
            hostInfo: { name: "mcp-agent-ui", version: "1.0.0" },
            hostCapabilities: { message: { text: {} } },
            hostContext: {},
          },
        });
      } else if (message.method === "ui/notifications/initialized") {
        post({
          jsonrpc: "2.0",
          method: "ui/notifications/tool-input",
          params: { arguments: {} },
        });
        post({
          jsonrpc: "2.0",
          method: "ui/notifications/tool-result",
          params: { content: [], structuredContent: latest.current },
        });
      } else if (message.method === "ui/notifications/size-changed") {
        const asked = message.params?.height;
        if (typeof asked === "number" && Number.isFinite(asked)) {
          setHeight(Math.min(Math.max(asked, HEIGHT.min), HEIGHT.max));
        }
      } else if (message.method === "ui/message" && message.id !== undefined) {
        const text = (message.params?.content ?? [])
          .filter((block: any) => block?.type === "text")
          .map((block: any) => block.text)
          .join("");
        if (text) reply.current(text);
        post({ jsonrpc: "2.0", id: message.id, result: {} });
      }
    }

    window.addEventListener("message", handle);
    return () => window.removeEventListener("message", handle);
  }, []);

  return (
    <iframe
      ref={frame}
      className="view"
      style={{ height }}
      title={uri}
      src={apiUrl(`/views/${toolset}/${view}`)}
      sandbox="allow-scripts"
    />
  );
}

/** How long one tool's name holds the indicator before the next takes it. */
const ROTATE = 1800;

/** What the run is doing, for as long as it runs.
 *
 * With the receipts folded away a run is otherwise silent between the question
 * and the answer, and a run is not quick: a tool call can fan out across a
 * dozen sources. Naming the call beats a bare spinner, and rotating is how
 * several of them fit in the space of one line.
 *
 * It says something at every phase, because a run has phases where no call is
 * in flight — the model has the turn, or it is writing the answer — and an
 * indicator that disappeared in those gaps would read as a run that stopped
 * rather than as one that is between calls.
 */
function Working({
  calls,
  writing,
}: {
  calls: { id: string; name: string }[];
  writing: boolean;
}) {
  const [at, setAt] = useState(0);
  useEffect(() => {
    if (calls.length < 2) return;
    const timer = setInterval(() => setAt((n) => n + 1), ROTATE);
    return () => clearInterval(timer);
  }, [calls.length]);

  // What is said, as against what is drawn. The two differ in one case: with
  // several calls in flight the name on screen changes every `ROTATE` so a
  // reader can scan them all, and a live region carrying that name would be
  // read aloud every `ROTATE` too. The count is the stable sentence, and it
  // changes only when the run does.
  const spoken =
    calls.length === 0
      ? writing
        ? "writing the answer"
        : "thinking"
      : calls.length === 1
        ? `calling ${calls[0].name}`
        : `calling ${calls.length} tools`;

  // No call to name. The caret in the answer already says text is arriving, so
  // the drawn half only has to say that the turn is still someone's.
  const showing = at % Math.max(calls.length, 1);
  const call = calls[showing];

  return (
    <div className="working">
      <span className="aloud" aria-live="polite">
        {spoken}
      </span>
      <i className="spin" aria-hidden="true" />
      <span className="dim" aria-hidden="true">
        {call ? "calling" : writing ? "writing the answer" : "thinking"}
      </span>
      {call ? (
        <>
          {/* Keyed on the call, so a swap animates rather than mutating in
              place under the reader. */}
          <code key={call.id} className="turning" aria-hidden="true">
            {call.name}
          </code>
          {calls.length > 1 ? (
            <span className="of" aria-hidden="true">
              {showing + 1} of {calls.length}
            </span>
          ) : null}
        </>
      ) : (
        <Dots />
      )}
    </div>
  );
}

/** Three dots, for a phase that has no name to show. Decoration, so it is
 * hidden from a screen reader — the label beside it is the message. */
function Dots() {
  return (
    <i className="dots" aria-hidden="true">
      <b />
      <b />
      <b />
    </i>
  );
}

/** The screen before anyone has asked anything.
 *
 * A deployment can write the paragraph and the example questions; failing
 * that, this says what the agent is actually connected to. A greeting written
 * into the client would be wrong in every repository that installs it, and
 * "connected to nothing" is worth seeing rather than hiding behind a welcome.
 */
function Opening({
  connected,
  missing,
  onAsk,
  onKeys,
}: {
  connected: Connected | null;
  missing: Declared[];
  onAsk: (text: string) => void;
  onKeys: () => void;
}) {
  const names = connected?.toolsets.map((each) => each.name) ?? [];
  return (
    <div className="opening">
      {config.greeting ? (
        <p>{config.greeting}</p>
      ) : names.length > 0 ? (
        <p className="dim">
          Connected to{" "}
          {names.map((name, index) => (
            <Fragment key={name}>
              {index ? ", " : ""}
              <code>{name}</code>
            </Fragment>
          ))}
          {connected ? ` · ${connected.tools.length} tools` : null}
        </p>
      ) : (
        <p className="dim">Ask something.</p>
      )}

      {missing.length > 0 ? (
        <p className="warn">
          {missing.length === 1
            ? "A connected toolset wants a key"
            : `Connected toolsets want ${missing.length} keys`}{" "}
          before those tools can run.{" "}
          <button className="link" onClick={onKeys}>
            add {missing.length === 1 ? "it" : "them"}
          </button>
        </p>
      ) : null}

      {config.examples.length > 0 ? (
        <div className="examples">
          {config.examples.map((each) => (
            <button key={each} className="example" onClick={() => onAsk(each)}>
              {each}
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}

/** How long what is typed here survives, in the visitor's terms.
 *
 * The deployment chooses the store; the person handing over a key is the one
 * who needs to know what was chosen, and they cannot see a container's
 * environment. Saying "kept in this browser" under every setting would be
 * false for `none` and would hide the tab-closing part of `session` — which is
 * the half of that setting a visitor on a shared machine actually cares about.
 */
const KEPT: Record<CredentialStore, string> = {
  local: "Remembered on this browser, including after it closes.",
  session: "Remembered until this tab closes.",
  none: "Kept only while this page is open, and never stored.",
};

/** One field per credential header a connected toolset declared.
 *
 * A header the server already holds is shown rather than hidden: a value
 * given here *wins* over the deployment's own, so a visitor with their own
 * account needs somewhere to say so, and someone wondering why a tool works
 * without a key needs to be able to see that one is already in force.
 */
function Keys({
  declared,
  values,
  onChange,
  onClose,
}: {
  declared: Declared[];
  values: Record<string, string>;
  onChange: (values: Record<string, string>) => void;
  onClose: () => void;
}) {
  return (
    <div className="panel">
      <p className="dim">
        Sent as headers with every question, and only ever to the toolset that
        asked for them. {KEPT[config.credentials]}
      </p>
      {declared.map((each) => (
        <label key={each.header}>
          <span>
            <code>{each.header}</code>
            <span className="dim">
              {" · "}
              {each.toolsets.join(", ")}
              {each.supplied ? " · the server has one" : ""}
            </span>
          </span>
          <input
            type="password"
            autoComplete="off"
            value={values[each.header] ?? ""}
            placeholder={each.supplied ? "using the server's" : "paste a key"}
            onChange={(changed) =>
              onChange({ ...values, [each.header]: changed.target.value })
            }
          />
        </label>
      ))}
      <button onClick={onClose}>done</button>
    </div>
  );
}

export function Chat() {
  // The reference client, not a parser of our own: `HttpAgent` POSTs a
  // `RunAgentInput`, runs the SSE through `verifyEvents`, and applies each
  // event to `messages` and `state`. If this server emitted anything the
  // protocol disallows, the run would fail here rather than render wrongly.
  // `?thread=` if the URL names one, so a reload comes back to the same
  // conversation rather than a fresh one — the thread lives in the
  // checkpointer, and the id is the only thing a client needs to keep.
  const [threadId, setThreadId] = useState(
    () =>
      new URLSearchParams(location.search).get("thread") || crypto.randomUUID(),
  );
  const agent = useMemo(
    () => new HttpAgent({ url: apiUrl("/runs"), threadId }),
    [threadId],
  );
  const log = useRef<HTMLDivElement>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [showing, setShowing] = useState(0);
  // Whether the turn selector is pinned to a past turn. A new turn steals focus
  // only while it is not — otherwise reading turn 1 is interrupted by turn 3.
  const pinned = useRef(false);
  const [opened, setOpened] = useState<{
    key: string;
    turn: number | null;
    value?: unknown;
    error?: string;
  } | null>(null);
  const [folded, setFolded] = useState(false);
  // Whether "clear" has been pressed once. Two presses, because the button
  // sits beside the conversation it ends and the id it abandons is not on
  // screen anywhere to type back in.
  const [confirming, setConfirming] = useState(false);
  // Raw payloads answer "is the view lying?", which is not a question most
  // turns raise — so they are behind this rather than under every message.
  // Nothing is discarded, only folded away.
  const [debug, setDebug] = useState(false);
  // Whether the state panel is open. It is a reference rather than the
  // output, so it starts as a spine to press rather than a fifth of the
  // window to read past.
  const [panel, setPanel] = useState(false);
  const [linked, setLinked] = useState<Linked>(NOTHING);
  const [running, setRunning] = useState(false);
  // The message currently receiving tokens, or null. Bracketed by the stream's
  // own TEXT_MESSAGE_START/END rather than inferred from the transcript: "the
  // newest assistant message" is a different claim, and it is wrong twice —
  // before this turn has written anything it names the last turn's answer, and
  // a tool call with no preamble is an assistant message with no text.
  const [writing, setWriting] = useState<string | null>(null);
  const busy = useRef(false);
  const [question, setQuestion] = useState("");
  // The questions the last run stopped on. Mirrors `agent.pendingInterrupts`,
  // which the library fills from RUN_FINISHED — held in state so the page
  // re-renders when it changes.
  const [pending, setPending] = useState<Interrupt[]>([]);
  // Answers given so far, by interrupt id. A resume must answer every open
  // question at once, so they are collected until the last one arrives.
  const [answers, setAnswers] = useState<Record<string, Response>>({});
  // What this deployment is connected to. Fetched once: it describes the
  // deployment rather than the conversation, and neither changes underneath a
  // running client.
  const [connected, setConnected] = useState<Connected | null>(null);
  const [keys, setKeys] = useState<Record<string, string>>(loadCredentials);
  const [askingKeys, setAskingKeys] = useState(false);

  // Rendered message elements, so a turn can be scrolled to by the question
  // that started it. Keyed by message id rather than index: ids are stable and
  // indices shift as a turn fills in beneath them.
  const nodes = useRef(new Map<string, HTMLElement>());

  useEffect(() => {
    // Not while a past turn is pinned: following the newest message would drag
    // the reader off the turn they went back to look at.
    if (pinned.current) return;
    log.current?.scrollTo({ top: log.current.scrollHeight });
  }, [messages]);

  useEffect(() => {
    document.title = config.title;
    if (config.accent) {
      document.documentElement.style.setProperty("--accent", config.accent);
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    readConnections()
      .then((found) => {
        if (!cancelled) setConnected(found);
      })
      // Not fatal, and deliberately not surfaced: this decorates the opening
      // screen and names the credential headers. A deployment needing none is
      // a working chat without it.
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, []);

  // Put the thread in the URL, so reloading the page restores it. Replace
  // rather than push: this is not a navigation, and a back button that stepped
  // through thread ids would be nonsense.
  useEffect(() => {
    const url = new URL(location.href);
    if (url.searchParams.get("thread") === threadId) return;
    url.searchParams.set("thread", threadId);
    history.replaceState(null, "", url);
  }, [threadId]);

  const declared = useMemo(
    () => (connected ? declaredCredentials(connected.toolsets) : []),
    [connected],
  );
  const missing = outstanding(declared, keys);

  // Assigned wholesale rather than merged: a header cleared here has to leave
  // the agent too, and `headersFor` has already dropped anything no toolset
  // declared — which the API would drop anyway, and for better reasons.
  useEffect(() => {
    agent.headers = headersFor(declared, keys);
  }, [agent, declared, keys]);

  /** Rebuild the conversation from the thread id alone.
   *
   * Two routes, because the stream has no turn boundary a reloaded client
   * could have seen: `/threads/{id}` is the transcript **and its activities**,
   * `/threads/{id}/turns` is what state held at the end of each turn. They are
   * joined on the question — turn *n* starts at the *n*th user message.
   *
   * The activities come back as messages, which is what an activity is in
   * AG-UI, so `origins` folds them into the same `key -> origin` map the live
   * client builds and the cross-highlighting works with no special case. Each
   * turn is bounded by the next one's start: unbounded, turn 1 would claim
   * every later turn's publications too.
   */
  useEffect(() => {
    let cancelled = false;
    (async () => {
      const thread = await readThread(threadId).catch(() => null);
      if (cancelled || !thread || thread.messages.length === 0) return;
      const past = await readTurns(threadId).catch(() => null);
      if (cancelled) return;

      const all = thread.messages as unknown as Message[];
      const starts = thread.messages
        .map((message, index) => ({ message, index }))
        .filter(({ message }) => message.role === "user");
      const restored: Turn[] = starts.map(({ message, index }, n) => ({
        n: n + 1,
        question: message.content || "",
        questionId: message.id,
        from: index,
        state: (past?.history[n]?.state ?? {}) as Snapshot,
        published: origins(
          all.slice(0, starts[n + 1]?.index ?? all.length),
          index,
        ),
      }));

      agent.setMessages(all);
      // A thread reloaded mid-question is still waiting for the answer, and
      // the library refuses to start a run that does not give one.
      agent.pendingInterrupts = thread.interrupts ?? [];
      setPending([...agent.pendingInterrupts]);
      setMessages([...agent.messages]);
      setTurns(restored);
      setShowing(Math.max(restored.length - 1, 0));
    })();
    return () => {
      cancelled = true;
    };
  }, [agent, threadId]);

  /** Start a new conversation, by abandoning this thread rather than by
   * deleting it.
   *
   * Nothing is removed server-side: there is no route that drops a
   * checkpointer record, and the old thread stays reachable by its own
   * `?thread=` URL. What a new id buys is a new *session state* — that is per
   * thread, so a stale value a tool published cannot follow you across.
   *
   * Everything the turn machinery holds is reset with it. `agent` is memoised
   * on the id, so a new id is a new `HttpAgent` with an empty transcript; what
   * would otherwise survive is this component's own state, and a leftover turn
   * list would index into messages that no longer exist. The open questions go
   * with it: they belong to a run in the thread being left behind.
   */
  function clearSession() {
    setThreadId(crypto.randomUUID());
    setMessages([]);
    setTurns([]);
    setShowing(0);
    setOpened(null);
    setLinked(NOTHING);
    setWriting(null);
    setPending([]);
    setAnswers({});
    setConfirming(false);
    pinned.current = false;
  }

  /** Put the question that started a turn at the top of the log.
   *
   * `scrollTo` on the log rather than `scrollIntoView` on the message, which
   * scrolls every scrollable ancestor as well — including, at narrow widths,
   * the page itself.
   */
  function scrollToTurn(turn: Turn) {
    const node = nodes.current.get(turn.questionId);
    if (!node || !log.current) return;
    // `offsetTop` counts from inside the log's padding, so a bare scroll puts
    // the question flush against the edge. The browser clamps at 0.
    log.current.scrollTo({ top: node.offsetTop - 12, behavior: "smooth" });
  }

  /** Fold this turn's `state.published` activities into `key -> origin`. */
  function origins(
    all: readonly Message[],
    from: number,
  ): Record<string, Origin> {
    const found: Record<string, Origin> = {};
    for (const message of all.slice(from)) {
      const content = (message as any).content;
      if (
        (message as any).role !== "activity" ||
        (message as any).activityType !== "state.published"
      ) {
        continue;
      }
      for (const key of Object.values<string>(content?.published ?? {})) {
        found[key] = { toolCallId: content.toolCallId, tool: content.tool };
      }
    }
    return found;
  }

  /** Run one turn. Called by the form, and by a view over `ui/message`.
   *
   * A ref rather than the `running` state for the guard: a view's button is
   * driven from a listener mounted once, so it closes over the first render's
   * value of anything held in state and would happily start a second turn on
   * top of the first.
   */
  async function run(text: string) {
    // Not while a question is open: the server refuses the message, so the
    // question is answered or skipped first.
    if (!text || busy.current || agent.pendingInterrupts.length > 0) return;
    busy.current = true;
    setRunning(true);

    const from = agent.messages.length;
    const questionId = crypto.randomUUID();
    agent.addMessage({ id: questionId, role: "user", content: text });
    setMessages([...agent.messages]);
    setTurns((held) => {
      const started: Turn = {
        n: held.length + 1,
        question: text,
        questionId,
        from,
        // Carried forward: state is cumulative, so a turn starts holding
        // everything the last one ended with.
        state: held[held.length - 1]?.state ?? {},
        published: {},
      };
      if (!pinned.current) setShowing(held.length);
      return [...held, started];
    });

    await drive();
  }

  /** One answer to an open question; the resume goes once all are in. */
  async function respond(interruptId: string, response: Response) {
    if (busy.current) return;
    const given = { ...answers, [interruptId]: response };
    setAnswers(given);
    const open = agent.pendingInterrupts;
    if (!open.every((each) => given[each.id])) return;
    busy.current = true;
    setRunning(true);
    // The answers carry on the turn that asked: no new question, no new turn.
    await drive(buildResumeArray(open, given));
  }

  /** Run the agent — for a question, or with `resume` for the answers. */
  async function drive(resume?: ResumeEntry[]) {
    const patch = (change: (turn: Turn) => Turn) =>
      setTurns((held) =>
        held.map((turn, index) =>
          index === held.length - 1 ? change(turn) : turn,
        ),
      );

    try {
      // One subscriber, called after each event is applied. Rendering from
      // `messages` rather than from the events is the point of the library:
      // an activity *is* a message, so it already sits where it belongs.
      await agent.runAgent(resume ? { resume } : undefined, {
        onEvent: ({ messages }) => {
          setMessages([...messages]);
          patch((turn) => ({
            ...turn,
            published: origins(messages, turn.from),
          }));
        },
        // Where the caret goes. The protocol brackets one assistant message's
        // text with these two, which is exactly what the caret claims.
        onTextMessageStartEvent: ({ event }) => setWriting(event.messageId),
        onTextMessageEndEvent: () => setWriting(null),
        // Session state arrives on AG-UI's standard `state` channel as
        // patches, every one of them under `toolState`. Each entry carries
        // `{tool, bytes, seq, inputs}`; see the README.
        //
        // Applied rather than merged: the operations say what changed,
        // including a key leaving, which a merge could not express. The one
        // that opens a run replaces the namespace whole — that is the
        // resynchronisation point, and it carries the thread's state, not just
        // this turn's writes.
        onStateDeltaEvent: ({ event }) => {
          patch((turn) => ({
            ...turn,
            state: applyDelta(turn.state, event.delta as Operation[]),
          }));
        },
      });
    } catch (error) {
      agent.addMessage({
        id: crypto.randomUUID(),
        role: "assistant",
        content: `client error: ${String(error)}`,
      });
      setMessages([...agent.messages]);
    } finally {
      busy.current = false;
      setRunning(false);
      // Filled from RUN_FINISHED, and left as it was by a run that failed —
      // so a refused answer can be given again.
      setAnswers({});
      setPending([...agent.pendingInterrupts]);
      // A run that fails between START and END never sends the END, which
      // would otherwise leave the caret blinking on a message nothing is
      // writing to.
      setWriting(null);
    }
  }

  function ask(submitted: React.FormEvent) {
    submitted.preventDefault();
    const text = question.trim();
    setQuestion("");
    void run(text);
  }

  const turn: Turn | undefined = turns[showing];
  const latest = showing === turns.length - 1;

  /** Fetch a key's value *as of the turn being shown*, not as of now.
   *
   * Passing the turn is the whole difference: a key a later turn overwrote
   * reads back as the later value without it, which is the wrong answer to
   * "what did this turn run on".
   */
  async function open(key: string) {
    const at = turn?.n;
    setFolded(false);
    try {
      const got = await readState(agent.threadId, key, at);
      setOpened({ key, turn: got.turn, value: got.value });
    } catch (error) {
      // A turn the checkpointer has pruned answers 410 with a sentence saying
      // so. Showing it beats a blank panel: "gone" and "never existed" are
      // different facts and the API has already told them apart.
      setOpened({
        key,
        turn: at ?? null,
        error: (error as Error).message,
      });
    }
  }

  /** Light a key, and with it the call and every activity about that call.
   *
   * The same set `litByCall` lights, deliberately: one relationship should
   * light identically whichever end of it is hovered, or the pair reads as two
   * coincidences rather than one link.
   */
  function litByKey(key: string) {
    const origin = turn?.published[key];
    setLinked(
      origin
        ? {
            keys: [key],
            calls: [origin.toolCallId],
            activities: activitiesOf(messages, origin.toolCallId),
          }
        : { ...NOTHING, keys: [key] },
    );
  }

  /** Light a call, and with it every key it wrote and every activity about it.
   *
   * Every activity, not only the ones announcing a key: a call's `mcp.view`
   * is the row hardest to attribute by eye, since several tools in a turn
   * each produce one and the rows are identical but for the URI.
   */
  function litByCall(toolCallId: string) {
    const wrote = Object.entries(turn?.published ?? {}).filter(
      ([, origin]) => origin.toolCallId === toolCallId,
    );
    setLinked({
      keys: wrote.map(([key]) => key),
      calls: [toolCallId],
      activities: activitiesOf(messages, toolCallId),
    });
  }

  /** Light a call and everything about it, from an activity of any shape.
   *
   * Whatever the activity is, not only `state.published`: the question a
   * reader has in front of a view or a consumed receipt is which call it
   * belongs to, and the `toolCallId` answering it is on all three. `keys`
   * stays empty for the two that publish nothing.
   */
  function litByActivity(content: any) {
    if (!content?.toolCallId) return;
    setLinked({
      keys: Object.values<string>(content.published ?? {}),
      calls: [content.toolCallId],
      activities: activitiesOf(messages, content.toolCallId),
    });
  }

  const entries = Object.entries(turn?.state ?? {}).sort(
    ([leftKey, left], [rightKey, right]) =>
      (left.seq ?? 0) - (right.seq ?? 0) || leftKey.localeCompare(rightKey),
  );
  // What the model actually wrote, recovered from the calls the transcript
  // holds. Nothing on the wire carries it; see `producedArguments`.
  const wroteFor = useMemo(() => producedArguments(messages), [messages]);
  // A call with no result yet is a call still running: the stream has no
  // "tool started" event, so the absence of its result is the only signal.
  //
  // Read from the newest turn rather than from the whole transcript, and the
  // newest rather than the *shown* one. A run that failed between a call and
  // its result leaves that call unsettled for good, and across the transcript
  // it would then be named as "calling" by every later run; scoped to the turn
  // in flight it is only ever a call this run made.
  const turnStart = turns[turns.length - 1]?.from ?? 0;
  const inFlight = useMemo(() => {
    const thisTurn = messages.slice(turnStart);
    const settled = new Set(
      thisTurn
        .filter((message) => message.role === "tool")
        .map((message) => (message as any).toolCallId),
    );
    return thisTurn.flatMap((message) =>
      ((message as any).toolCalls ?? [])
        .filter((call: any) => !settled.has(call.id))
        .map((call: any) => ({ id: String(call.id), name: call.function.name })),
    );
  }, [messages, turnStart]);
  // The `interrupt` calls in the transcript, whose results are their answers.
  const asked = useMemo(
    () =>
      new Set(
        messages.flatMap((message) =>
          ((message as any).toolCalls ?? [])
            .filter((call: any) => call.function.name === "interrupt")
            .map((call: any) => String(call.id)),
        ),
      ),
    [messages],
  );

  return (
    <main
      className={
        [opened ? (folded ? "folded" : "opened") : "", panel ? "" : "shut"]
          .filter(Boolean)
          .join(" ") || undefined
      }
    >
      <div className="chat">
        <header>
          <span className="name">
            <b>{config.title}</b>
            {config.tagline ? (
              <span className="dim"> · {config.tagline}</span>
            ) : null}
          </span>
          {/* The two colours are the whole point of the wire: blue is what
              AG-UI gives any client, amber is what this runtime adds on top
              of it. Naming them beats leaving a reader to infer it. */}
          <span className="legend">
            <i className="swatch tool" /> AG-UI
            <i className="swatch activity" /> receipts and views
            {declared.length > 0 ? (
              <button
                className="link"
                onClick={() => setAskingKeys(!askingKeys)}
                title="credential headers the connected toolsets declared"
              >
                keys{missing.length > 0 ? ` · ${missing.length} needed` : ""}
              </button>
            ) : null}
            <span className="dim">· thread {agent.threadId.slice(0, 8)}</span>
            <button
              className={confirming ? "toggle warn" : "toggle"}
              // Nothing is deleted — see `clearSession`. A run in flight is
              // still writing to the thread it started in, so wait for it.
              disabled={running}
              onClick={() => (confirming ? clearSession() : setConfirming(true))}
              onBlur={() => setConfirming(false)}
              title={
                confirming
                  ? "Press again to start a new thread"
                  : "Start a new thread — this one keeps its own URL, and its session state stays with it"
              }
            >
              {confirming ? "clear?" : "clear"}
            </button>
            <button
              className={debug ? "toggle on" : "toggle"}
              onClick={() => setDebug(!debug)}
              aria-pressed={debug}
              title="Show the raw JSON behind every result and receipt"
            >
              debug
            </button>
          </span>
        </header>

        {askingKeys ? (
          <Keys
            declared={declared}
            values={keys}
            onChange={(next) => {
              setKeys(next);
              saveCredentials(next);
            }}
            onClose={() => setAskingKeys(false)}
          />
        ) : null}

        <div className="log" ref={log}>
          {messages.map((message) =>
            message.role === "user" || message.role === "assistant" ? (
              <div
                key={message.id}
                className={`said ${message.role}`}
                ref={(node) => {
                  const id = String(message.id);
                  if (node) nodes.current.set(id, node);
                  else nodes.current.delete(id);
                }}
              >
                {/* Models answer in markdown whether or not you asked, and a
                    half-written stream is half-written markdown — an unclosed
                    ** or a table with one row. react-markdown re-parses each
                    delta, so it degrades to plain text rather than showing
                    syntax, and renders no raw HTML, which matters when the
                    text came from a model.

                    GFM because a model asked to compare things answers with a
                    table, and tables are not CommonMark — without this the
                    pipes are the output. Strikethrough and bare URLs come with
                    it. */}
                <Markdown remarkPlugins={[remarkGfm]}>
                  {String(message.content ?? "")}
                </Markdown>
                {(message as any).toolCalls?.map((call: any) =>
                  call.function.name === "interrupt" ? (
                    <Question
                      key={call.id}
                      call={call}
                      open={pending.find((each) => each.toolCallId === call.id)}
                      given={
                        answers[
                          pending.find((each) => each.toolCallId === call.id)
                            ?.id ?? ""
                        ]
                      }
                      result={
                        messages.find(
                          (each) =>
                            each.role === "tool" &&
                            (each as any).toolCallId === call.id,
                        )?.content as string | undefined
                      }
                      onAnswer={(id, response) => void respond(id, response)}
                    />
                  ) : (
                    // <details> rather than state: collapsing is what the
                    // element is for, and the keyboard and screen-reader
                    // behaviour comes with it.
                    <details
                      key={call.id}
                      className={`tool ${linked.calls.includes(call.id) ? "lit" : ""}`}
                      onMouseEnter={() => litByCall(call.id)}
                      onMouseLeave={() => setLinked(NOTHING)}
                    >
                      <summary>
                        <code>{call.function.name}</code>
                      </summary>
                      <pre>{prettyArgs(call.function.arguments)}</pre>
                    </details>
                  ),
                )}
                {message.id === writing ? <i className="caret" /> : null}
              </div>
            ) : message.role === "tool" &&
              // An answer is drawn inside its question, not again here.
              !asked.has(String((message as any).toolCallId)) ? (
              // Whatever a view is drawing, this is the same value as text.
              // Only worth the room when you are checking one against the
              // other, which is what `debug` is.
              debug ? (
                <details key={message.id} className="tool">
                  <summary>
                    <span className="dim">result</span>{" "}
                    {summarise(String(message.content ?? ""))}
                  </summary>
                  <pre>{String(message.content ?? "")}</pre>
                </details>
              ) : null
            ) : message.role === "activity" ? (
              (message as any).content?.uri ? (
                // A view is the tool's answer, so it sits where an answer
                // sits: a sibling of the messages at the same level, with no
                // summary to open and no receipt wrapped around it. It keeps
                // the hover link all the same — a view is the activity
                // hardest to attribute by eye, since a turn with three tools
                // in it draws three of them.
                <div
                  key={message.id}
                  className={`said shown-view ${
                    linked.activities.includes(String(message.id)) ? "lit" : ""
                  }`}
                  onMouseEnter={() => litByActivity((message as any).content)}
                  onMouseLeave={() => setLinked(NOTHING)}
                >
                  <View
                    uri={(message as any).content.uri}
                    data={(message as any).content.data}
                    // `ui/message` starts the turn rather than filling the
                    // box: a host may send it, and a button
                    // that only types for you is a view that cannot act.
                    onMessage={run}
                  />
                </div>
              ) : debug ? (
                <details
                  key={message.id}
                  className={`activity ${
                    linked.activities.includes(String(message.id)) ? "lit" : ""
                  }`}
                  onMouseEnter={() => litByActivity((message as any).content)}
                  onMouseLeave={() => setLinked(NOTHING)}
                >
                  <summary>
                    <em>{(message as any).activityType}</em>
                    {(message as any).content?.tool ? (
                      <>
                        {" "}
                        <code>{(message as any).content.tool}</code>
                      </>
                    ) : null}
                  </summary>
                  <span>{shown((message as any).content)}</span>
                  <pre className="dim">
                    {JSON.stringify((message as any).content, null, 2)}
                  </pre>
                </details>
              ) : null
            ) : null,
          )}
          {messages.length === 0 ? (
            <Opening
              connected={connected}
              missing={missing}
              onAsk={(text) => void run(text)}
              onKeys={() => setAskingKeys(true)}
            />
          ) : null}
        </div>

        {/* The indicator floats over the log rather than taking a row of its
            own, so starting a run does not shorten the log by its height and
            move the text a reader is in the middle of. It is anchored to the
            composer rather than to the column, so nothing here has to know how
            tall the composer is. */}
        <div className="composer">
          {running ? (
            <Working calls={inFlight} writing={writing !== null} />
          ) : null}

          <form onSubmit={ask}>
            <input
              value={question}
              onChange={(changed) => setQuestion(changed.target.value)}
              placeholder={
                pending.length > 0
                  ? "answer or skip the question above first"
                  : "ask something"
              }
              disabled={pending.length > 0}
              autoFocus
            />
            {/* The spinner covers the label rather than sitting beside it: the
                button cannot be pressed while a turn runs, so the word is not
                telling anyone anything they can act on. The label stays in the
                markup all the same — hidden, holding the width (see the CSS).
                `busy` overrides the disabled dimming, since a half-faded
                spinner reads as broken rather than as working, and `aria-busy`
                with a label keeps the state announced: a hidden span leaves
                the button with no accessible name of its own. */}
            <button
              className={running ? "busy" : undefined}
              disabled={running || pending.length > 0 || !question.trim()}
              aria-busy={running}
              aria-label={running ? "answering" : undefined}
            >
              <span className="label">send</span>
              {running ? <i className="spinner" aria-hidden="true" /> : null}
            </button>
          </form>
        </div>
      </div>

      <aside>
        {/* The heading is the collapse button: expanded, the two said the same
            words in the same column, and a title you can press is the shorter
            of the two ways to say it. */}
        <h2>
          <button
            className="shutter"
            onClick={() => setPanel(!panel)}
            aria-expanded={panel}
            title={panel ? "Collapse the state panel" : "Expand the state panel"}
          >
            {panel ? "›" : "‹"} <span className="edge">session state</span>
          </button>
        </h2>

        {turns.length > 0 ? (
          <>
            {/* State is cumulative, so a turn is a position in it rather than
                a slice of it. Switching turns rewinds the panel to what the
                thread held then. */}
            <div className="turns">
              {turns.map((each, index) => (
                <button
                  key={each.n}
                  className={index === showing ? "turn on" : "turn"}
                  title={each.question}
                  onClick={() => {
                    setShowing(index);
                    pinned.current = index !== turns.length - 1;
                    setLinked(NOTHING);
                    scrollToTurn(each);
                  }}
                >
                  {each.n}
                </button>
              ))}
            </div>
            <p className="asked dim">
              {turn?.question}
              {latest ? null : " · past turn"}
            </p>
          </>
        ) : (
          <p className="dim">
            What the tools exchanged without the model reading it. The stream
            carries this much per key and no payload; the value is a fetch away.
          </p>
        )}

        {entries.map(([key, entry]) => {
          const origin = turn?.published[key];
          return (
            <div
              key={key}
              className={`slot ${linked.keys.includes(key) ? "lit" : ""}`}
              onMouseEnter={() => litByKey(key)}
              onMouseLeave={() => setLinked(NOTHING)}
            >
              <div className="card">
                <button
                  className="key"
                  title={`GET /threads/…/state/${key}?turn=${turn?.n}`}
                  onClick={() => void open(key)}
                >
                  <code>
                    {origin ? <b className="new">new</b> : null}{" "}
                    <Key value={key} />
                  </code>
                  <span className="dim">
                    {bytes(entry.bytes)} · from {entry.tool}
                    {entry.turnsWritten ? (
                      // Only ever shown when more than one turn wrote the key,
                      // because the server omits the field otherwise. The
                      // panel shows the *current* value, so this is the one
                      // thing here saying an earlier turn holds another.
                      <>
                        {" · "}
                        <b className="rewritten">
                          written in {entry.turnsWritten} turns
                        </b>
                      </>
                    ) : null}
                  </span>
                </button>
                {producedBy(entry).length > 0 ? (
                  <>
                    <p className="inputs-label">
                      inputs to <code>{entry.tool}</code>
                    </p>
                    <ul className="inputs">
                      {producedBy(entry).map(([parameter, from]) => (
                        <li key={parameter}>
                          <code className="param">{parameter}</code>
                          <span className="rel">
                            {from === "model" ? " = " : " ← "}
                          </span>
                          {from === "model" ? (
                            <Wrote value={wroteFor[key]?.[parameter]} />
                          ) : (
                            <button
                              className="from"
                              title={`from ${from} — click to open it`}
                              onMouseEnter={() => litByKey(from)}
                              onClick={() => void open(from)}
                            >
                              <Key value={from} />
                            </button>
                          )}
                        </li>
                      ))}
                    </ul>
                  </>
                ) : null}
              </div>
            </div>
          );
        })}

        {turns.length > 0 && entries.length === 0 ? (
          <p className="dim">nothing published yet</p>
        ) : null}
      </aside>

      {opened ? (
        <section className="value">
          <header>
            <button
              className="fold"
              onClick={() => setFolded(!folded)}
              title={folded ? "expand" : "collapse"}
            >
              {folded ? "›" : "‹"}
            </button>
            {folded ? null : (
              <>
                <code>{opened.key}</code>
                <button className="fold" onClick={() => setOpened(null)}>
                  ✕
                </button>
              </>
            )}
          </header>
          {folded ? null : (
            <>
              <p className="dim">
                <b>
                  GET /threads/…/state/{opened.key}
                  {opened.turn === null ? "" : `?turn=${opened.turn}`}
                </b>
                <br />
                {/* Which turn this is, said plainly: several of these panels
                    over a conversation are otherwise indistinguishable, and
                    the value genuinely differs between turns. */}
                {opened.turn === null
                  ? "as state stands now"
                  : `as it stood at the end of turn ${opened.turn}`}
              </p>
              {opened.error ? (
                <p className="error">{opened.error}</p>
              ) : (
                <pre>{JSON.stringify(opened.value, null, 2)}</pre>
              )}
            </>
          )}
        </section>
      ) : null}
    </main>
  );
}
