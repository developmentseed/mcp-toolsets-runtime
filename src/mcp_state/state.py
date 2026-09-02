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

Keys are *qualified* — ``dataset-search/search_datasets/area_of_interest``
rather than ``area_of_interest`` (see
:func:`mcp_runtime.declarations.qualified`), so one toolset's write cannot
overwrite another's, and so the key a model reads says which call produced the
value.

Values are wrapped in a :class:`StateEntry` rather than stored bare, because a
listing has to say where each value came from and which write was most recent.
Readers take ``entry["value"]``.

An entry also records what the call that produced it was *given*
(``inputs``). A tool owns its outputs and this never second-guesses them; what
it records is the one thing the client knows for certain, which is where each
argument came from. Read one level — the call that produced the entry you are
looking at — and it says whether a value rests on something a tool produced or
on something the model wrote. Read further, since every argument it names is
either the model or another key, and it is a chain.

Last write wins, so a key holds one value and not a history. What an entry
does keep is the turn it was written in, and how many turns hold a value for
its key (``turns_written``) — because a reader handed the current value has no
other way to learn an earlier turn holds one too. Turns, not writes: a key
written twice inside one turn ends that turn holding the second value, and the
first is reachable by nothing, so counting it would name something that cannot
be fetched. Whether two turns actually *differ* is not claimed and is not
knowable from an entry; the reader finds out by fetching one, which a host that
retains turns makes possible — see :mod:`mcp_state.history`.
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
    #: Where each argument of the producing call came from: a ``tool_state``
    #: key, or :data:`MODEL_AUTHORED` for one the model wrote. Absent where
    #: that call took no arguments. Parameter names and keys only — never
    #: values, so this stays cheap however large the call was.
    inputs: NotRequired[dict[str, str]]
    #: The turn this value was written in, counted off the user's questions
    #: and stamped by :class:`~mcp_state.middleware.StateCaptureMiddleware`.
    #: What :func:`merge_tool_state` needs to tell a displaced value the
    #: conversation can still reach from one it cannot.
    turn: NotRequired[int]
    #: How many turns of the conversation hold a value for this key, this one
    #: included, counted by :func:`merge_tool_state`. Each is one value a
    #: turn-scoped read can fetch. Absent means one, so an entry written
    #: before this was recorded reads as the only one. Read it through
    #: :func:`turns_written`.
    turns_written: NotRequired[int]


#: Recorded in a :class:`StateEntry`'s ``inputs`` for an argument the model
#: wrote itself, as against one that named a stored value.
MODEL_AUTHORED = "model"


def turns_written(entry: StateEntry | None) -> int:
    """How many turns of the conversation hold a value for an entry's key.

    Named for what it counts. Turns are the unit a read can address — a
    turn-scoped read resolves to what a turn *ended* holding — so one turn
    that wrote this key is one value that can be fetched, and the count is
    exactly the number of them.

    Deliberately **not** a count of how many times the value changed. Deciding
    that needs the value each turn ended with, which is history, and an entry
    holds no history; an attempt at it from the previous *write* alone gets
    both directions wrong inside a turn. So this claims only what it can back
    up, and the model finds out whether two turns differ by reading them.

    ``1`` for a key written in one turn, and for one captured before the count
    was recorded — the honest default, since a listing that flagged every
    pre-existing entry would train a model to ignore the flag.
    """
    return entry.get("turns_written", 1) if entry else 1


def authored(entry: StateEntry | None) -> list[str]:
    """The parameters of an entry's producing call that the model wrote.

    Empty for a value whose call took only stored values, and for one captured
    before ``inputs`` was recorded. One level: this reads the call that
    produced *this* entry and follows nothing further, because at depth "the
    model wrote something upstream" is true of every value in a session — it
    wrote the query that found the dataset.
    """
    inputs = entry.get("inputs") if entry else None
    return sorted(
        name for name, origin in (inputs or {}).items() if origin == MODEL_AUTHORED
    )


def rests_on_state(entry: StateEntry | None) -> list[str]:
    """The ``tool_state`` keys an entry's producing call was given.

    The other half of ``inputs`` from :func:`authored`, and the half that says
    a value was built on something a tool found rather than on something
    invented. Same one level, for the same reason.
    """
    inputs = entry.get("inputs") if entry else None
    return sorted(
        origin for origin in (inputs or {}).values() if origin != MODEL_AUTHORED
    )


def merge_tool_state(
    current: dict[str, StateEntry] | None, update: dict[str, StateEntry] | None
) -> dict[str, StateEntry]:
    """Reducer for ``tool_state``: merge per key, later writes win.

    Without it langgraph would replace the whole dict on every write, so one
    tool's update would erase another's — and parallel tool calls in the same
    step would conflict instead of merging.

    Also stamps ``seq`` on entries that arrive without one, continuing from
    the highest already stored, and ``turns_written`` on one that displaces a
    value an earlier turn left under the same key. Both are counted here rather
    than in the middleware for the same reason: the reducer is the one place
    that sees old and new state together, so nobody else has to hold a counter.
    """
    merged = {**(current or {})}
    if not update:
        return merged
    next_seq = max((entry.get("seq", 0) for entry in merged.values()), default=0) + 1
    for key, entry in update.items():
        if entry.get("seq") is None:
            entry = {**entry, "seq": next_seq}
            next_seq += 1
        entry = _counted(merged.get(key), entry)
        merged[key] = entry
    return _within_budget(merged)


def _counted(previous: StateEntry | None, entry: StateEntry) -> StateEntry:
    """``entry`` carrying how many turns hold a value for its key.

    Counted here for the same reason ``seq`` is: the reducer is the one place
    that sees the outgoing entry and the incoming one together. Doing it at
    merge time also means it survives whatever the checkpointer prunes — the
    count still says an earlier turn held a value once that value is gone,
    which is exactly the case a reader most needs to be told about.

    A second write **inside the turn that wrote the entry it displaces** adds
    nothing. That turn ends holding one value however many times it was
    written, and the earlier ones are reachable by nothing, so counting them
    would name a version that cannot be fetched.

    Turns are compared rather than assumed to advance, so where no turn was
    stamped — a host driving capture outside a graph — every write counts.
    That deployment has no turn history to read either way, and going quiet
    there would lose the signal everywhere capture runs without a message
    list.
    """
    if previous is None or entry.get("turns_written") is not None:
        return entry
    held = turns_written(previous)
    return {
        **entry,
        "turns_written": held + 1 if _later_turn(previous, entry) else held,
    }


def _later_turn(previous: StateEntry, entry: StateEntry) -> bool:
    """Whether ``entry`` was written in a turn after the one it displaces.

    ``True`` when either side went unstamped: an unknown turn cannot be shown
    to be the same one, and treating it as such would silence the signal
    everywhere capture runs outside a graph.
    """
    before, after = previous.get("turn"), entry.get("turn")
    if before is None or after is None:
        return True
    return after > before


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
    # Dropping an entry drops what it knew about its key, `turns_written`
    # included — so a later write to an evicted key finds no previous entry and
    # its count restarts at 1, and a read of it will not say earlier turns
    # wrote it. Best-effort on purpose: holding the count past eviction means a
    # tombstone per key, which is a second thing to budget for the sake of a
    # hint. Only the hint is lost, not the values: the checkpointer still holds
    # those turns, so `turn=` reads them for anyone who thinks to ask.
    return {key: entry for key, entry in merged.items() if key in kept}


class AgentState(_BaseAgentState):
    """The agent's built-in state plus the namespace tools publish into."""

    tool_state: NotRequired[Annotated[dict[str, StateEntry], merge_tool_state]]
