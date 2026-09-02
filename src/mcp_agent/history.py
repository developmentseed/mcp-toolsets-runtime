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

from mcp_state.history import Snapshots
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


async def _derive(
    checkpoints: AsyncIterator[Checkpoint], current: Sequence[Any]
) -> History:
    """Turns out of a newest-first walk of a thread's checkpoints.

    Walks the whole history, which is proportional to the length of the
    conversation rather than to what is being asked for. That is the honest
    cost of deriving turns from a structure that does not record them; a thread
    long enough for it to matter has other problems first.

    ``current`` is the thread's messages as they stand, read separately from
    the walk. That separation is what carries retention: the questions survive
    in the newest checkpoint however far back the older ones have been pruned,
    so counting them there rather than in the walk is what makes ``total``
    bigger than the number of turns found.
    """
    # A turn's *last* checkpoint is the one holding everything it published.
    # Decided per checkpoint rather than taken from the walk's order: the
    # savers this repo runs against happen to yield newest first, but the base
    # contract promises an iterator and nothing more, and trusting it would
    # make an ascending saver silently return each turn's *starting* state —
    # a well-formed wrong value, the exact failure this module exists to
    # remove.
    ending: dict[int, Checkpoint] = {}
    async for checkpoint in checkpoints:
        if count := len(_humans(checkpoint.messages)):
            held = ending.get(count)
            if held is None or _ends_later(checkpoint, held):
                ending[count] = checkpoint

    asked = _humans(current)

    def turn(n: int, checkpoint: Checkpoint) -> Turn:
        return Turn(
            n=n,
            # From the current messages rather than the checkpoint's, so a turn
            # whose own checkpoints were evicted could still be named.
            question=_text(asked[n - 1]) if n <= len(asked) else "",
            checkpoint_id=checkpoint.checkpoint_id,
            state=checkpoint.state,
        )

    return History(
        turns=[turn(n, checkpoint) for n, checkpoint in sorted(ending.items())],
        total=len(asked),
    )


def _ends_later(candidate: Checkpoint, held: Checkpoint) -> bool:
    """Whether ``candidate`` sits later in its turn than ``held``.

    Within one turn the message list only grows — every super-step appends and
    none removes — so more messages is later. Equal lists fall back to the
    checkpoint id, which LangGraph mints time-ordered (UUIDv6, so lexical
    order is write order): that covers a super-step that touched other
    channels and left the messages alone.
    """
    if len(candidate.messages) != len(held.messages):
        return len(candidate.messages) > len(held.messages)
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


async def turns_from(saver: BaseCheckpointSaver, thread_id: str) -> History:
    """The same, read straight off the checkpointer.

    For a caller that has the store but not the graph. ``aget_state_history``
    additionally applies pending writes and resolves the tasks a snapshot would
    run next, which raw checkpoints do not carry — neither bears on
    ``tool_state`` at a turn that has finished, which is all this reads.
    """

    config = _config(thread_id)

    async def walk() -> AsyncIterator[Checkpoint]:
        async for entry in saver.alist(config):
            yield _checkpoint(entry)

    latest = await saver.aget_tuple(config)
    return await _derive(walk(), _checkpoint(latest).messages if latest else [])


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

    async def snapshots(self, thread_id: str) -> Snapshots:
        history = await turns_from(self._saver, thread_id)
        return Snapshots(
            turns={turn.n: turn.state for turn in history.turns},
            total=history.total,
        )
