"""``inspect_state``: lets the model read the ``tool_state`` namespace on demand.

See :mod:`mcp_state.state` for the tool-side convention and the ``tool_state``
namespace this reads from. The stored values never enter the model's context
on their own — the model reads a value on demand with the tool
:func:`make_inspect_state` builds,
whole when small, otherwise narrowed with ``pattern`` (grep over flattened
``path = value`` lines) or ``path`` (drill into one sub-value).

This is the *pull* half of the pair. When a value's consumer is another tool
rather than the model, prefer the push half — a parameter the server marks
with :class:`mcp_runtime.injected.Injected`, filled by
:mod:`mcp_state.injection` — which moves the same value without the model
reading it at all. ``inspect_state`` is for when the model itself needs to
know something: to summarise a result, or decide what to do next.

Image/binary blobs (data URIs, long base64) are redacted to a short marker
before any read, so the model learns an image exists without receiving bytes
it can neither use nor safely echo — the UI renders those from state directly.
"""

import json
import re
from collections.abc import Iterable, Iterator
from typing import Any

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from mcp_state.state import TOOL_STATE_KEY, StateEntry

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


def _summaries(populated: dict[str, Any]) -> dict[str, str]:
    """``{key: shape}`` for every populated state key."""
    return {name: _shape(value) for name, value in populated.items()}


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
    structure outline instead. ``allowed_keys`` (``None`` skips the check)
    hides any key outside it, as if never stored.
    """
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
        name: value
        for name, value in sorted(tool_state.items())
        if value is not None and (allowed_keys is None or name in allowed_keys)
    }

    if key in (ALL_KEYS, ""):
        if path:
            return _dumps(
                {
                    "error": "path needs a specific key",
                    "available_keys": _summaries(populated),
                }
            )
        if pattern:
            pairs = (
                pair
                for name, value in populated.items()
                for pair in _flatten(value, _path_key(name))
            )
            return _grep_response(pattern, "all keys", pairs, populated)
        return _dumps({"available_keys": _summaries(populated)})

    if key not in populated:
        return _dumps(
            {"unknown_or_empty_key": key, "available_keys": _summaries(populated)}
        )

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
    if pattern:
        return _grep_response(pattern, where, _flatten(target, prefix), target)

    dumped = _dumps(target)
    if len(dumped) <= MAX_RESULT_CHARS:
        return dumped
    return _dumps(_outline(where, target, len(dumped)))


def make_inspect_state(allowed_keys: frozenset[str] | None = None) -> Any:
    """Build the ``inspect_state`` tool, restricted to ``allowed_keys``.

    Pass :func:`mcp_state.middleware.state_keys` over the same ``published``
    map the middleware was built with, so the model can only name a key some
    connected tool actually declares.
    """

    @tool
    def inspect_state(
        key: str,
        runtime: ToolRuntime,
        pattern: str | None = None,
        path: str | None = None,
    ) -> str:
        """Read or search a session-state key named in a '[state updated: ...]' note.

        Args:
            key: State key to read, or "*" to search across every stored key.
            pattern: Case-insensitive regex (invalid regex is treated as plain
                text) matched against flattened "path = value" lines; returns
                the matching lines.
            path: Sub-value to return, e.g. "parameters.variable.values" or
                "[0].id" — use the paths that pattern matches print.

        key alone returns the full JSON value, or a structure outline when it
        is large — then narrow with pattern or path.
        """
        return read_state_key(
            key,
            runtime.state or {},
            allowed_keys=allowed_keys,
            pattern=pattern,
            path=path,
        )

    return inspect_state
