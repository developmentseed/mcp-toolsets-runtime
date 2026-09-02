"""``inspect_state``: lets the model read the ``tool_state`` namespace on demand.

See :mod:`mcp_state.state` for the tool-side convention and the ``tool_state``
namespace this reads from. The stored values never enter the model's context
on their own — the model reads a value on demand with the tool
:func:`make_inspect_state` builds,
whole when small, otherwise narrowed with ``pattern`` (grep over flattened
``path = value`` lines) or ``path`` (drill into one sub-value).

This is the *pull* half of the pair. When a value's consumer is another tool
rather than the model, prefer the push half (:mod:`mcp_state.injection`) — a
parameter filled from state by the client, or an ``@state:<key>`` handle the
model passes without reading — which moves the same value without it entering
the context. ``inspect_state`` is for when the model itself needs to know
something: to summarise a result, or decide what to do next.

Image/binary blobs (data URIs, long base64) are redacted to a short marker
before any read, so the model learns an image exists without receiving bytes
it can neither use nor safely echo — the UI renders those from state directly.

A key holds one value, so a read answers "what is stored now" and a later tool
call can have made that a different value from the one an earlier answer was
built on. Two things address that, and the first is what makes the second get
used: a read says when its key has held other values, and ``turn=`` reads the
key as it stood at the end of an earlier turn — for a host that supplies a
:class:`~mcp_state.history.ThreadHistory`, and answering plainly that it
cannot for one that does not.
"""

import json
import re
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from mcp_state.handles import handle_key, is_handle
from mcp_state.history import ThreadHistory
from mcp_state.state import TOOL_STATE_KEY, StateEntry, versions

MAX_RESULT_CHARS = 4000  # bigger serialized values return an outline, not JSON
MAX_MATCHES = 50  # grep lines returned
MAX_LINE_CHARS = 160  # displayed value cap inside one match line
ALL_KEYS = "*"  # key sentinel: operate across every stored key

_BARE_SEGMENT = re.compile(r"[\w-]+")
_PATH_TOKEN = re.compile(r'\[(\d+)\]|\[("(?:[^"\\]|\\.)*")\]|([^.\[\]]+)')


def _dumps(value: Any) -> str:
    """The one serializer: unicode kept, non-JSON objects via ``str``."""
    return json.dumps(value, ensure_ascii=False, default=str)


def _is_blob(value: str) -> bool:
    """A ``data:`` URI — an embedded binary/image payload, not text to read.

    Deliberately narrow (prefix only): tools embed images as ``data:`` URIs
    (see cds.tools.preview_arco), and matching bare base64 by shape would
    false-positive on any long alphanumeric string.
    """
    return value.startswith("data:")


def _blob_marker(value: str) -> str:
    """Short stand-in for a redacted blob (kept < 40 chars so _shape prints it)."""
    kind = "image" if value.startswith("data:image/") else "blob"
    return f"<{kind}, {len(value) // 1024} KB, rendered in UI>"


def _redact_tree(value: Any) -> Any:
    """Copy a value, replacing image/binary blobs with a short marker.

    Applied to state before any read/grep/outline so the model can see that an
    image exists (and that the UI renders it) but never receives the bytes —
    which it would otherwise be tempted to echo back into its answer. A host
    prompt should say so explicitly; the redaction is the enforcement.
    """
    if isinstance(value, str):
        return _blob_marker(value) if _is_blob(value) else value
    if isinstance(value, dict):
        return {key: _redact_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_tree(item) for item in value]
    return value


def _path_key(key: str) -> str:
    """One path segment for a dict key: bare, or bracket-quoted if unsafe."""
    if _BARE_SEGMENT.fullmatch(key):
        return key
    return f"[{_dumps(key)}]"


def _join(prefix: str, segment: str) -> str:
    """Attach a segment to a path: dots between bare parts, brackets direct."""
    if not prefix:
        return segment
    if segment.startswith("["):
        return prefix + segment
    return f"{prefix}.{segment}"


def _flatten(value: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    """Depth-first gron-style walk yielding ``(path, scalar)`` pairs.

    Empty containers yield themselves so structure stays greppable; a root
    scalar gets the path ``"."``.
    """
    if isinstance(value, dict) and value:
        for key, item in value.items():
            yield from _flatten(item, _join(prefix, _path_key(str(key))))
    elif isinstance(value, list) and value:
        for i, item in enumerate(value):
            yield from _flatten(item, f"{prefix}[{i}]")
    else:
        yield prefix or ".", value


def _shape(value: Any) -> str:
    """A one-line summary of a value's structure."""
    if isinstance(value, dict):
        if not value:
            return "dict(empty)"
        keys = list(value)
        shown = ", ".join(keys[:8]) + (", …" if len(keys) > 8 else "")
        return f"dict({len(keys)} keys: {shown})"
    if isinstance(value, list):
        return f"list[{len(value)}] of {_shape(value[0])}" if value else "list[0]"
    if isinstance(value, str) and len(value) > 40:
        return f"str({len(value)} chars)"
    return _dumps(value)


def _structure(value: Any) -> str:
    """Top-level shapes of a value, for zero-match hints."""
    if isinstance(value, dict) and value:
        return ", ".join(f"{name}: {_shape(item)}" for name, item in value.items())
    return _shape(value)


def _summaries(
    populated: dict[str, Any], stored: Mapping[str, StateEntry] | None = None
) -> dict[str, str]:
    """``{key: shape}`` for every populated state key, rewrites called out.

    The shape line is the whole of what a model sees before deciding whether
    the value in front of it is the one an earlier turn ran on. A key holds
    one value, so a replacement is otherwise invisible from here: the read
    succeeds and the value is well-formed. Saying "rewritten" is what makes a
    turn-scoped read something the model knows to reach for.
    """
    lines = {}
    for name, value in populated.items():
        shape = _shape(value)
        held = versions((stored or {}).get(name))
        if held > 1:
            shape += f" — rewritten, {held} versions"
        lines[name] = shape
    return lines


def _rewrite_note(entry: StateEntry | None) -> str:
    """A trailing line saying this key has held other values, or ``""``.

    Appended after the value rather than wrapped around it, so a read still
    returns the payload and nothing has to be unwrapped — the same shape as
    the ``[state updated: …]`` breadcrumb the tool result already carries.

    A read of a single key is where the failure this guards against actually
    happens: the key listing marks rewrites, but a model told a key by a
    breadcrumb goes straight to the value and is shown a well-formed answer to
    a question it did not mean to ask.
    """
    held = versions(entry)
    if held < 2:
        return ""
    return (
        f"\n[this key has held {held} different values in this session; the "
        "above is the current one. Pass turn=<n> to read it as it stood at the "
        "end of an earlier turn, counting the user's questions from 1.]"
    )


def _excerpt(text: str, span: tuple[int, int] | None = None) -> str:
    """Cap a displayed value, windowing around the first match when known."""
    if len(text) <= MAX_LINE_CHARS:
        return text
    if span is None:
        return text[:MAX_LINE_CHARS] + f"…(+{len(text) - MAX_LINE_CHARS} chars)"
    start = max(0, min(span[0] - MAX_LINE_CHARS // 2, len(text) - MAX_LINE_CHARS))
    end = start + MAX_LINE_CHARS
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    return f"{prefix}{text[start:end]}{suffix}"


def _compile(pattern: str) -> tuple[re.Pattern[str], bool]:
    """A case-insensitive regex; invalid patterns fall back to literal text."""
    try:
        return re.compile(pattern, re.IGNORECASE), False
    except re.error:
        return re.compile(re.escape(pattern), re.IGNORECASE), True


def _search(
    pairs: Iterable[tuple[str, Any]], regex: re.Pattern[str]
) -> tuple[int, int, list[str]]:
    """Grep flattened pairs: (values scanned, total matches, capped lines).

    The regex runs against the full ``path = value`` line; only the display
    is excerpted, centered on the match so long strings show their context.
    """
    scanned = matches = 0
    lines: list[str] = []
    for path, scalar in pairs:
        scanned += 1
        rendered = _dumps(scalar)
        found = regex.search(f"{path} = {rendered}")
        if not found:
            continue
        matches += 1
        if len(lines) < MAX_MATCHES:
            offset = len(path) + 3  # skip "path = " so the span maps to rendered
            start = found.start() - offset
            span = (start, found.end() - offset) if start >= 0 else None
            lines.append(f"{path} = {_excerpt(rendered, span)}")
    return scanned, matches, lines


def _grep_response(
    pattern: str, where: str, pairs: Iterable[tuple[str, Any]], searched: Any
) -> str:
    """Plain-text grep result: header plus matching ``path = value`` lines."""
    regex, literal = _compile(pattern)
    scanned, matches, lines = _search(pairs, regex)
    note = " (pattern treated as literal text)" if literal else ""
    if not matches:
        return (
            f"no matches for {pattern!r} in {where} ({scanned} values){note}. "
            f"Structure: {_structure(searched)}"
        )
    header = f"{matches} match{'es' if matches != 1 else ''} for {pattern!r} in {where}"
    if matches > len(lines):
        header += f" (showing {len(lines)})"
    return f"{header}{note}:\n" + "\n".join(lines)


def _outline(key: str, value: Any, size: int) -> dict[str, Any]:
    """Structure summary returned instead of an over-sized value."""
    outline: Any
    if isinstance(value, list):
        first = value[0] if value else None
        outline = {
            "length": len(value),
            "items": _shape(first) if value else "empty",
            "first": first if len(_dumps(first)) <= 500 else _shape(first),
        }
    elif isinstance(value, dict):
        outline = {name: _shape(item) for name, item in value.items()}
    else:
        outline = _shape(value)
    return {
        "key": key,
        "size_chars": size,
        "type": type(value).__name__,
        "outline": outline,
        "hint": (
            "too large to return whole; pass pattern=<regex> to search it "
            "or path=<dotted.path> to drill down"
        ),
    }


def _strip_key_prefix(path: str, key: str) -> str:
    """Drop a redundant leading dot or ``key`` prefix from a drill path."""
    path = path.lstrip(".")
    if path == key:
        return ""
    if path.startswith(f"{key}."):
        return path[len(key) + 1 :]
    if path.startswith(f"{key}["):
        return path[len(key) :]
    return path


def _resolve_path(value: Any, path: str) -> tuple[Any, str | None]:
    """Walk a dot/bracket path: ``(sub_value, None)`` or ``(shape, bad_segment)``."""
    current = value
    for index, quoted, bare in _PATH_TOKEN.findall(path):
        segment: Any = int(index) if index else json.loads(quoted) if quoted else bare
        if isinstance(current, list):
            try:
                current = current[int(segment)]
                continue
            except (ValueError, IndexError):
                return _shape(current), str(segment)
        if isinstance(current, dict):
            if segment in current:
                current = current[segment]
                continue
            if str(segment) in current:
                current = current[str(segment)]
                continue
        return _shape(current), str(segment)
    return current, None


def read_state_key(
    key: str,
    state: dict[str, Any],
    *,
    allowed_keys: frozenset[str] | None = None,
    pattern: str | None = None,
    path: str | None = None,
) -> str:
    """One state key — whole JSON, grepped by ``pattern``, or drilled by ``path``.

    ``key="*"`` lists or greps every stored key; unknown keys report what is
    available; values whose JSON exceeds ``MAX_RESULT_CHARS`` come back as a
    structure outline instead.

    Always the state it is handed, which is the present when the caller passes
    live graph state and an earlier turn when the caller passes that turn's
    entries — this knows nothing of turns either way. What it does add is a
    trailing note on a key whose entry records earlier values, since a bare
    read is otherwise indistinguishable from a read of a key written once.

    ``allowed_keys`` names keys to expose *in addition to* whatever is actually
    stored, for a host that wants a declared-but-unwritten key to appear in the
    listing. It is not a filter: a key in ``tool_state`` is always readable,
    because capture is what put it there — it applied the secret backstop on
    the way in, and it wrote the ``[state updated: …]`` breadcrumb telling the
    model to read exactly this key. Filtering here would make that breadcrumb
    a lie for every value captured by size, which is the whole undeclared path.

    A ``key`` written as ``@state:<key>`` is read as ``<key>``. Here the key is
    the argument itself, so a handle can only mean the key inside it, and
    refusing one would be pedantry over a read that is perfectly well
    specified. Tolerated but deliberately not advertised — the tool's own
    description asks for a bare key, because ``@state:`` belongs to parameters
    that take a *value*, and teaching it here is what makes models reach for
    it everywhere.
    """
    if is_handle(key):
        key = handle_key(key)
    raw = state.get(TOOL_STATE_KEY)
    stored: dict[str, StateEntry] = raw if isinstance(raw, dict) else {}
    # Unwrap the StateEntry envelope — the model reads values, not the kind
    # and write-order metadata that injection resolves on. Redact blobs once,
    # up front, so every downstream path (whole read, drill, grep, outline,
    # key listing) operates on the marker, never the bytes.
    tool_state: dict[str, Any] = {
        name: _redact_tree(entry.get("value") if isinstance(entry, dict) else entry)
        for name, entry in stored.items()
    }
    populated = {
        name: value for name, value in sorted(tool_state.items()) if value is not None
    }

    if key in (ALL_KEYS, ""):
        if path:
            return _dumps(
                {
                    "error": "path needs a specific key",
                    "available_keys": _summaries(populated, stored),
                }
            )
        if pattern:
            pairs = (
                pair
                for name, value in populated.items()
                for pair in _flatten(value, _path_key(name))
            )
            return _grep_response(pattern, "all keys", pairs, populated)
        return _dumps({"available_keys": _summaries(populated, stored)})

    if key not in populated:
        response: dict[str, Any] = {
            "unknown_or_empty_key": key,
            "available_keys": _summaries(populated, stored),
        }
        # A key some tool declares but has not published is a different answer
        # from a key nobody has ever heard of: the model should run the
        # producing tool, not give up or invent the value.
        if allowed_keys and key in allowed_keys:
            response["hint"] = (
                "A tool declares this key but has not published it yet in this "
                "session — run the tool that produces it, then read it again."
            )
        return _dumps(response)

    target = tool_state[key]
    prefix = ""
    if path:
        sub_path = _strip_key_prefix(path, key)
        if sub_path:
            resolved, failed_at = _resolve_path(target, sub_path)
            if failed_at is not None:
                return _dumps(
                    {
                        "path_not_found": path,
                        "key": key,
                        "failed_at": failed_at,
                        "context": resolved,
                    }
                )
            target, prefix = resolved, sub_path

    where = _join(key, prefix) if prefix else key
    note = _rewrite_note(stored.get(key))
    if pattern:
        return _grep_response(pattern, where, _flatten(target, prefix), target) + note

    dumped = _dumps(target)
    if len(dumped) <= MAX_RESULT_CHARS:
        return dumped + note
    return _dumps(_outline(where, target, len(dumped))) + note


def thread_of(config: Any) -> str | None:
    """The ``thread_id`` a runnable config was invoked under, if any.

    A conversation without one is a conversation nothing is keeping, so this
    returning ``None`` and the host having no history are the same answer to
    the model.
    """
    if not isinstance(config, dict):
        return None
    identifier = (config.get("configurable") or {}).get("thread_id")
    return identifier if isinstance(identifier, str) else None


async def read_state_key_at_turn(
    key: str,
    turn: int,
    thread_id: str | None,
    history: ThreadHistory | None,
    *,
    pattern: str | None = None,
    path: str | None = None,
) -> str:
    """One state key as it stood at the end of ``turn``.

    Everything downstream of "which state" is unchanged: this resolves the
    turn to a ``{key: entry}`` mapping and hands it to :func:`read_state_key`,
    so narrowing, blob redaction, outlining and the unknown-key listing all
    behave exactly as they do for a read of the present.

    No ``allowed_keys``, deliberately. A key that misses here missed *at that
    turn*, and "a tool declares this but has not published it — run it" is
    advice about the present: acting on it would have the model call a tool to
    fill a key that may well be filled already. The listing this falls back to
    is the one thing that answers the question actually asked, which is what
    the turn held.

    Three ways it does not produce a value, and they mean different things:

    - the deployment retains no turn history at all, so the question cannot be
      asked here rather than having a negative answer;
    - the thread never had that turn, which is a miscount the model can fix;
    - the turn has been pruned. **The value existed and is gone.** Saying so is
      the point of the whole seam: "I cannot answer that, the earlier value is
      no longer retained" is a good answer, and silently comparing a value
      with itself is not.
    """
    if history is None or thread_id is None:
        return _dumps(
            {
                "no_turn_history": turn,
                "reason": (
                    "This deployment does not retain per-turn history, so only "
                    "the current value of a key can be read. Omit turn."
                ),
            }
        )
    snapshots = await history.snapshots(thread_id)
    if snapshots.never_had(turn):
        return _dumps(
            {
                "no_such_turn": turn,
                "turns_so_far": snapshots.total,
                "reason": (
                    "Turns are the user's questions, counted from 1. This "
                    f"conversation has had {snapshots.total}."
                ),
            }
        )
    state = snapshots.at(turn)
    if state is None:
        return _dumps(
            {
                "turn_no_longer_retained": turn,
                "retained_turns": sorted(snapshots.turns),
                "reason": (
                    "The value at that turn existed and has since been pruned. "
                    "It cannot be recovered — say so rather than answering "
                    "from the current value, which is a different value."
                ),
            }
        )
    return read_state_key(
        key, {TOOL_STATE_KEY: dict(state)}, pattern=pattern, path=path
    )


def make_inspect_state(
    allowed_keys: frozenset[str] | None = None,
    history: ThreadHistory | None = None,
) -> Any:
    """Build the ``inspect_state`` tool.

    Everything in ``tool_state`` is readable — see :func:`read_state_key` for
    why that is not a filter to configure. ``allowed_keys``, if given, is only
    used to tell a declared-but-unpublished key apart from an unknown one when
    a read misses; pass :func:`mcp_state.middleware.state_keys` over the same
    ``published`` map the middleware was built with.

    ``history`` is what lets the model read a key *as it stood at an earlier
    turn*, and is optional. Without one the tool behaves exactly as it always
    has and ``turn=`` answers that this deployment keeps no turn history, so a
    host embedding this without a checkpointer needs to change nothing. See
    :mod:`mcp_state.history`.
    """

    @tool
    async def inspect_state(
        key: str,
        runtime: ToolRuntime,
        pattern: str | None = None,
        path: str | None = None,
        turn: int | None = None,
    ) -> str:
        """Read or search a session-state key named in a '[state updated: ...]' note.

        Args:
            key: State key to read, or "*" to search across every stored key.
            pattern: Case-insensitive regex (invalid regex is treated as plain
                text) matched against flattened "path = value" lines; returns
                the matching lines.
            path: Sub-value to return, e.g. "parameters.variable.values" or
                "[0].id" — use the paths that pattern matches print.
            turn: Read the key as it stood at the *end* of this turn instead of
                now, counting the user's questions from 1. Use it when a key
                says it has been rewritten and you need the value an earlier
                answer was based on; a key holds one value, so without this you
                would be comparing the current value with itself.

        key alone returns the full JSON value, or a structure outline when it
        is large — then narrow with pattern or path.
        """
        if turn is not None:
            return await read_state_key_at_turn(
                key,
                turn,
                thread_of(runtime.config),
                history,
                pattern=pattern,
                path=path,
            )
        return read_state_key(
            key,
            runtime.state or {},
            allowed_keys=allowed_keys,
            pattern=pattern,
            path=path,
        )

    return inspect_state
