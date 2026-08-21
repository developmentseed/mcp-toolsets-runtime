"""Agent graph state that MCP tools publish into and read back from.

Where a client keeps what tools exchange. :mod:`mcp_runtime.declarations` is
how a *server* describes what it publishes; this is the namespace those values
live in, and it fills up whether or not anything was declared.

A value lands here either because its tool declared it, or because it was too
large to leave in the transcript. Everything lands under one namespace, so
tools can never touch the agent's own channels (``messages`` etc.), and no
agent-side declaration is needed.

The stored values never enter the model's context: the tool message becomes
the ``message`` plus a ``[state updated: …]`` breadcrumb. From there a value
travels one of two ways — the model reads it on demand with ``inspect_state``,
or it points a tool parameter at the key with ``@state:<key>`` and the client
substitutes the value on the way out (:mod:`mcp_state.handles`), so the value
itself never passes through the transcript either way.

Keys are *qualified* — ``dataset-search/search_datasets/geometry`` rather than
``geometry`` (see :func:`mcp_runtime.declarations.qualified`), so one toolset's
write cannot overwrite another's, and so the key a model reads says which call
produced the value.

Values are wrapped in a :class:`StateEntry` rather than stored bare, because a
listing has to say where each value came from and which write was most recent.
Readers take ``entry["value"]``.
"""

import json
from typing import Annotated, Any, NotRequired, TypedDict

from langchain.agents.middleware import AgentState as _BaseAgentState

TOOL_STATE_KEY = "tool_state"

#: Serialised bytes of stored values to keep before evicting the oldest.
#: Nothing else bounds this namespace: capture writes on every tool call and a
#: single one can be tens of kB, so a long session would otherwise grow until
#: the process died — per user, in a hosted chat, and re-serialised every turn
#: by anyone running the agent under a checkpointer.
#:
#: Set high enough that an ordinary session never reaches it (hundreds of large
#: geometries), because eviction is not free: a handle names a key, and a key
#: that has been evicted resolves to nothing. Newest-first is the right order to
#: keep for exactly that reason.
MAX_TOOL_STATE_BYTES = 8 * 1024 * 1024


class StateEntry(TypedDict):
    """One published value, with what a reader needs to make sense of it."""

    value: Any
    tool: NotRequired[str]
    #: Monotonic write order, assigned by :func:`merge_tool_state`. What orders
    #: the listing a model chooses from, newest first.
    seq: NotRequired[int]


def merge_tool_state(
    current: dict[str, StateEntry] | None, update: dict[str, StateEntry] | None
) -> dict[str, StateEntry]:
    """Reducer for ``tool_state``: merge per key, later writes win.

    Without it langgraph would replace the whole dict on every write, so one
    tool's update would erase another's — and parallel tool calls in the same
    step would conflict instead of merging.

    Also stamps ``seq`` on entries that arrive without one, continuing from
    the highest already stored. Doing it here rather than in the middleware
    keeps the counter correct without anyone having to hold it: the reducer is
    the one place that sees old and new state together.
    """
    merged = {**(current or {})}
    if not update:
        return merged
    next_seq = max((entry.get("seq", 0) for entry in merged.values()), default=0) + 1
    for key, entry in update.items():
        if entry.get("seq") is None:
            entry = {**entry, "seq": next_seq}
            next_seq += 1
        merged[key] = entry
    return _within_budget(merged)


def _entry_size(entry: StateEntry) -> int:
    """The serialised size of one entry's value, or 0 if it will not serialise."""
    try:
        return len(json.dumps(entry.get("value"), default=str))
    except (TypeError, ValueError):
        return 0


def _within_budget(
    merged: dict[str, StateEntry], budget: int = MAX_TOOL_STATE_BYTES
) -> dict[str, StateEntry]:
    """``merged`` trimmed to ``budget`` bytes, dropping the oldest writes first.

    Cheap in the normal case: the total is only computed when the namespace has
    enough entries to be worth checking, and a session that never approaches
    the budget never loses anything.
    """
    if len(merged) < 2:
        return merged
    newest_first = sorted(
        merged.items(), key=lambda item: item[1].get("seq", 0), reverse=True
    )
    kept: dict[str, StateEntry] = {}
    total = 0
    for key, entry in newest_first:
        total += _entry_size(entry)
        if total > budget and kept:  # always keep the most recent write
            break
        kept[key] = entry
    if len(kept) == len(merged):
        return merged
    return {key: entry for key, entry in merged.items() if key in kept}


class AgentState(_BaseAgentState):
    """The agent's built-in state plus the namespace tools publish into."""

    tool_state: NotRequired[Annotated[dict[str, StateEntry], merge_tool_state]]
