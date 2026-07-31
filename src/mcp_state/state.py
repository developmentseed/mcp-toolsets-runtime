"""Agent graph state that MCP tools publish into and read back from.

Where a client keeps what tools exchange. :mod:`mcp_runtime.declarations` is
how a *server* may describe what it publishes and takes; this is the namespace
those values live in, and it fills up whether or not anything was declared.

A value lands here either because its tool declared it, or because it was too
large to leave in the transcript. Everything lands under one namespace, so
tools can never touch the agent's own channels (``messages`` etc.), and no
agent-side declaration is needed.

The stored values never enter the model's context: the tool message becomes
the ``message`` plus a ``[state updated: …]`` breadcrumb. From there a value
travels one of two ways — the model reads it on demand with ``inspect_state``,
or the client feeds it straight back into a later tool call without the model
ever seeing it (:mod:`mcp_state.injection`).

Two properties of the namespace make that second path work:

Keys are *qualified* — ``dataset-search/geometry`` rather than ``geometry``
(see :func:`mcp_runtime.declarations.qualified`), so one toolset's write cannot
overwrite another's.

Values are wrapped in a :class:`StateEntry` rather than stored bare, because
resolving by kind has to know each value's kind and which write was most
recent. Readers take ``entry["value"]``.
"""

from typing import Annotated, Any, NotRequired, TypedDict

from langchain.agents.middleware import AgentState as _BaseAgentState

TOOL_STATE_KEY = "tool_state"


class StateEntry(TypedDict):
    """One published value, with what a consumer needs to find it again."""

    value: Any
    kind: NotRequired[str | None]
    tool: NotRequired[str]
    #: Monotonic write order, assigned by :func:`merge_tool_state`. Kind
    #: resolution picks the highest — "the AOI we are working with" is
    #: reliably the most recently published one.
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
    return merged


class AgentState(_BaseAgentState):
    """The agent's built-in state plus the namespace tools publish into."""

    tool_state: NotRequired[Annotated[dict[str, StateEntry], merge_tool_state]]


def entries_of_kind(
    tool_state: dict[str, StateEntry] | None, kind: str
) -> list[tuple[str, StateEntry]]:
    """Every published entry of ``kind``, most recently written first."""
    return sorted(
        (
            (key, entry)
            for key, entry in (tool_state or {}).items()
            if entry.get("kind") == kind
        ),
        key=lambda item: item[1].get("seq", 0),
        reverse=True,
    )
