"""A thread's past turns, and what session state held at each of them.

The state route serves a key's *current* value, which is the only thing the
live stream ever describes. But a conversation is a sequence of turns, and a
client showing "what this turn ran on" needs the value as it stood then — a key
overwritten by a later turn resolves to the later value, and there is no way to
ask for the earlier one from the snapshot alone. Neither does the model, which
is the same gap seen from the other side: :class:`CheckpointHistory` is what
closes it for :func:`~mcp_state.inspect.make_inspect_state`.

Nothing new has to be stored for that. LangGraph writes an **immutable
checkpoint per super-step**, each carrying the whole graph state, ``tool_state``
included — so every past value is already retained and reachable. This module
turns that sequence of checkpoints into the unit a conversation is actually
made of.

**A turn is the checkpoints between two questions.** The graph has no notion of
one; what it has is a growing message list. So the turns are derived by
counting human messages: every checkpoint whose message list holds *n* of them
belongs to turn *n*, and the last such checkpoint is where that turn ended.

**Two ways in, one derivation.** A compiled graph offers
``aget_state_history``; a checkpointer offers ``alist`` — which matters because
tools are built before the graph they will run in, so an ``inspect_state`` that
needs history can only be handed the saver. :func:`turns_of` and
:func:`turns_from` differ in where the checkpoints come from and in nothing
else, so "what a turn is" is defined once. Neither source's *order* is relied
on: the base saver contract promises an iterator of checkpoints and says
nothing about their order, so which checkpoint ended a turn is decided from the
checkpoints themselves (:func:`_ends_later`).

**The walk is the whole conversation; what it holds need not be.** Both halves
of that cost money and neither is small. A saver materialises every row it is
asked for before yielding the first — LangGraph's PostgreSQL one does
``fetchall`` over the thread — so the read is paged (:data:`PAGE`), which is
what bounds the bytes the *store* holds. And a caller after one turn is handed
one turn (:func:`_derive`'s ``keep``), which is what bounds the objects *this*
holds. Together they make a read's cost a function of the page size rather than
of the conversation: on a thread carrying real geometries it is the same at
twenty turns as at six.

**Counting questions assumes they all survive.** Middleware that trims or
summarises a thread's messages removes some of them, and a checkpoint written
after that holds fewer questions than the turn it belongs to — so its turn
number goes backwards and collides with a turn already derived. Nothing here
ships such middleware, and a host that adds one loses turn numbering rather
than anything else: what a turn *held* is still in the checkpoints, and the
counts a :class:`~mcp_state.state.StateEntry` carries are stamped when the
write happens, from the same message list.

**Retention is the checkpointer's, not ours.** An in-process saver keeps
everything for the life of the process; a PostgreSQL one keeps whatever it has
not been pruned of. So "turn 2 is not here" has two meanings, and a
:class:`History` reports both counts so a caller can tell them apart: a turn
that never existed is a mistake, and a turn that has been evicted is a fact
about the deployment.
"""

from collections.abc import AsyncIterator, Sequence
from typing import Any, NamedTuple, cast

from langgraph.checkpoint.base import BaseCheckpointSaver

from mcp_state.history import Snapshot
from mcp_state.state import TOOL_STATE_KEY, StateEntry

#: LangChain's discriminator for a message from the user.
HUMAN = "human"


class Turn(NamedTuple):
    """One question, and the session state the thread held when it finished."""

    n: int
    question: str
    checkpoint_id: str | None
    state: dict[str, StateEntry]


class History(NamedTuple):
    """The turns still retained, and how many the thread has had.

    ``total`` is counted from the thread's *newest* checkpoint, whose messages
    survive however far back the rest of them go — so ``total > len(turns)`` is
    the signal that something was evicted rather than never written.
    """

    turns: list[Turn]
    total: int

    def find(self, n: int) -> Turn | None:
        return next((turn for turn in self.turns if turn.n == n), None)


class Checkpoint(NamedTuple):
    """The three things deriving a turn needs out of one checkpoint.

    Whatever it came from. A graph snapshot and a raw checkpoint tuple carry
    these under different names and a great deal else besides; reducing both to
    this is what lets one walk serve either.
    """

    messages: Sequence[Any]
    state: dict[str, StateEntry]
    checkpoint_id: str | None


def _humans(messages: Sequence[Any]) -> list[Any]:
    return [message for message in messages if getattr(message, "type", None) == HUMAN]


def _text(message: Any) -> str:
    """A question as text, however the client sent its content."""
    text = getattr(message, "text", None)
    return text if isinstance(text, str) else str(getattr(message, "content", ""))


class Placed(NamedTuple):
    """Where a checkpoint sits within its turn, without holding the checkpoint.

    Ranking one checkpoint against another needs two facts — how many messages
    it has and its id — and neither needs the messages themselves. Reducing to
    this before anything is retained is what keeps a walk's cost independent of
    the conversation's length: a turn nobody asked about is counted and placed,
    and carries nothing.
    """

    messages: int
    checkpoint_id: str | None
    state: dict[str, StateEntry]


async def _derive(
    checkpoints: AsyncIterator[Checkpoint],
    current: Sequence[Any] | None = None,
    *,
    keep: int | None = None,
) -> History:
    """Turns out of a walk of a thread's checkpoints.

    Walks the whole history, which is proportional to the length of the
    conversation rather than to what is being asked for. That is the honest
    cost of deriving turns from a structure that does not record them.

    What it *retains* need not be, and ``keep`` is how a caller says so: named
    a turn, every other one is placed and counted but hands back no state.
    A read of one turn is the common case and the expensive one, since session
    state holds precisely the values too big for a transcript — so keeping
    every turn's to answer about one multiplies the largest thing here by the
    length of the conversation. ``None`` keeps them all, which is what a client
    listing the whole thread wants.

    ``current`` is the thread's messages as they stand, for a caller that has
    them. Without it the questions come from the richest checkpoint the walk
    saw, which is the same list: a message list only grows, so whichever
    checkpoint holds the most questions holds all of them, and it survives
    however far back the older ones were pruned. Either way ``total`` counts
    questions rather than turns found, which is what makes it exceed them.
    """
    # A turn's *last* checkpoint is the one holding everything it published.
    # Decided per checkpoint rather than taken from the walk's order: the
    # savers this repo runs against happen to yield newest first, but the base
    # contract promises an iterator and nothing more, and trusting it would
    # make an ascending saver silently return each turn's *starting* state —
    # a well-formed wrong value, the exact failure this module exists to
    # remove.
    ending: dict[int, Placed] = {}
    seen: list[Any] = []
    async for checkpoint in checkpoints:
        humans = _humans(checkpoint.messages)
        if not (count := len(humans)):
            continue
        if count > len(seen):
            seen = humans
        placed = Placed(
            messages=len(checkpoint.messages),
            checkpoint_id=checkpoint.checkpoint_id,
            state=checkpoint.state if keep is None or count == keep else {},
        )
        held = ending.get(count)
        if held is None or _ends_later(placed, held):
            ending[count] = placed

    asked = _humans(current) if current is not None else seen

    def turn(n: int, placed: Placed) -> Turn:
        return Turn(
            n=n,
            # From the whole thread's questions rather than this checkpoint's,
            # so a turn whose own checkpoints were evicted could still be named.
            question=_text(asked[n - 1]) if n <= len(asked) else "",
            checkpoint_id=placed.checkpoint_id,
            state=placed.state,
        )

    return History(
        turns=[turn(n, placed) for n, placed in sorted(ending.items())],
        total=len(asked),
    )


def _ends_later(candidate: Placed, held: Placed) -> bool:
    """Whether ``candidate`` sits later in its turn than ``held``.

    Within one turn the message list only grows — every super-step appends and
    none removes — so more messages is later. Equal counts fall back to the
    checkpoint id, which LangGraph mints time-ordered (UUIDv6, so lexical
    order is write order): that covers a super-step that touched other
    channels and left the messages alone.
    """
    if candidate.messages != held.messages:
        return candidate.messages > held.messages
    return (candidate.checkpoint_id or "") > (held.checkpoint_id or "")


def _config(thread_id: str) -> Any:
    return cast(Any, {"configurable": {"thread_id": thread_id}})


async def turns_of(agent: Any, thread_id: str) -> History:
    """Every retained turn of a thread, oldest first, read through the graph."""
    config = _config(thread_id)

    async def walk() -> AsyncIterator[Checkpoint]:
        async for snapshot in agent.aget_state_history(config):
            values = getattr(snapshot, "values", None) or {}
            configurable = (getattr(snapshot, "config", None) or {}).get(
                "configurable", {}
            )
            yield Checkpoint(
                messages=values.get("messages") or [],
                state=dict(values.get(TOOL_STATE_KEY) or {}),
                checkpoint_id=configurable.get("checkpoint_id"),
            )

    current = await agent.aget_state(config)
    return await _derive(
        walk(), (getattr(current, "values", None) or {}).get("messages") or []
    )


#: Checkpoints fetched per query when walking a thread off the saver.
#:
#: ``alist`` materialises every row it is asked for before yielding the first —
#: LangGraph's PostgreSQL saver does ``fetchall`` — so this is what bounds the
#: bytes the *saver* holds, as streaming inside the page bounds the objects
#: this builds from them. Together they make a read's cost a function of the
#: page and not of the conversation.
#:
#: Chosen by measuring, against PostgreSQL on a thread carrying real
#: geometries. Peak is flat in turn count at every size and rises with the
#: size; queries are the other way about. Ten turns of a 4.3 MB state:
#:
#: ==== ======== ===========
#: page peak     queries
#: ==== ======== ===========
#: 2    115 MB   26
#: 8    166 MB   7
#: 16   198 MB   4
#: 32   266 MB   2
#: ==== ======== ===========
#:
#: Eight because the curve has flattened by there and the round trips have
#: not yet piled up. Every row of those figures is flat in the turn count: what
#: a read costs is set here and nowhere else.
PAGE = 8


class _NotDescending(Exception):
    """A saver whose walk cannot be paged, raised from inside the page."""


async def _paged(
    saver: BaseCheckpointSaver, thread_id: str, size: int
) -> AsyncIterator[Checkpoint]:
    """A thread's checkpoints, newest first, ``size`` fetched at a time.

    ``before`` is strictly-less-than on the checkpoint id and the walk is
    ordered by that id descending, so the last id of a page is exactly where
    the next one starts: nothing is seen twice and nothing is skipped. A short
    page is the end of the thread.

    Each page is yielded as it arrives rather than collected first, which is
    the difference between holding one deserialised checkpoint and holding
    ``size`` of them. Both matter and they are not the same size: a page bounds
    the *serialised* rows the saver fetched, and streaming bounds the Python
    objects built from them — and on a thread carrying geometries the objects
    cost an order of magnitude more than the bytes.

    Paging is the one thing in this module that needs the walk to be ordered,
    and :func:`_derive` goes out of its way not to be. The contract still does
    not promise it, so this checks: out of order, or a checkpoint with no id to
    page from, and it raises for :func:`turns_from` to fall back to the unpaged
    walk — slower, and correct. The saver's own iterator is closed either way.
    """
    config = _config(thread_id)
    before: Any = None
    while True:
        seen, last = 0, None
        entries = saver.alist(config, before=before, limit=size)
        try:
            async for entry in entries:
                checkpoint = _checkpoint(entry)
                identifier = checkpoint.checkpoint_id
                if not identifier or (last is not None and identifier > last):
                    raise _NotDescending
                seen, last = seen + 1, identifier
                yield checkpoint
        finally:
            # Only an async *generator* can be closed, and the contract types
            # an async iterator — so ask before closing rather than assume.
            # It matters when the guard below raises part-way through a page:
            # the saver has a cursor open, and nothing else will shut it.
            if (close := getattr(entries, "aclose", None)) is not None:
                await close()
        if seen < size or last is None:
            return
        before = cast(
            Any, {"configurable": {"thread_id": thread_id, "checkpoint_id": last}}
        )


async def turns_from(
    saver: BaseCheckpointSaver,
    thread_id: str,
    *,
    keep: int | None = None,
    page: int = PAGE,
) -> History:
    """The same, read straight off the checkpointer.

    For a caller that has the store but not the graph. ``aget_state_history``
    additionally applies pending writes and resolves the tasks a snapshot would
    run next, which raw checkpoints do not carry — neither bears on
    ``tool_state`` at a turn that has finished, which is all this reads.

    Read a page at a time, and retaining only ``keep``'s state, because both
    halves of the memory this costs are avoidable and neither is small: the
    saver holds every row it fetched, and the caller holds every turn it was
    handed. See :data:`PAGE` and :func:`_derive`.
    """

    async def whole() -> AsyncIterator[Checkpoint]:
        async for entry in saver.alist(_config(thread_id)):
            yield _checkpoint(entry)

    try:
        return await _derive(_paged(saver, thread_id, page), keep=keep)
    except _NotDescending:
        return await _derive(whole(), keep=keep)


def _checkpoint(entry: Any) -> Checkpoint:
    """One :class:`CheckpointTuple` reduced to what a turn is derived from."""
    values = entry.checkpoint.get("channel_values") or {}
    return Checkpoint(
        messages=values.get("messages") or [],
        state=dict(values.get(TOOL_STATE_KEY) or {}),
        checkpoint_id=(entry.config.get("configurable") or {}).get("checkpoint_id"),
    )


class CheckpointHistory:
    """:class:`~mcp_state.history.ThreadHistory` backed by a LangGraph saver.

    The adapter that lets a model read a state key as of an earlier turn. It
    takes the checkpointer rather than the agent deliberately: ``inspect_state``
    is constructed *before* the graph it will be bound into, and the saver is
    already in hand at that point, so nothing has to be filled in afterwards.

    A deployment with no checkpointer has no past to offer and builds none of
    this — see :func:`mcp_agent.main.with_session_state`.
    """

    def __init__(self, saver: BaseCheckpointSaver) -> None:
        self._saver = saver

    async def snapshot(self, thread_id: str, turn: int) -> Snapshot:
        history = await turns_from(self._saver, thread_id, keep=turn)
        found = history.find(turn)
        return Snapshot(
            state=found.state if found else None,
            retained=frozenset(each.n for each in history.turns),
            total=history.total,
        )
