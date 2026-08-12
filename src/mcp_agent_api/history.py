"""A thread's past turns, and what session state held at each of them.

The state route serves a key's *current* value, which is the only thing the
live stream ever describes. But a conversation is a sequence of turns, and a
client showing "what this turn ran on" needs the value as it stood then — a key
overwritten by a later turn resolves to the later value, and there is no way to
ask for the earlier one from the snapshot alone.

Nothing new has to be stored for that. LangGraph writes an **immutable
checkpoint per super-step**, each carrying the whole graph state, ``tool_state``
included — so every past value is already retained and reachable through
``aget_state_history``. This module turns that sequence of checkpoints into the
unit a conversation is actually made of.

**A turn is the checkpoints between two questions.** The graph has no notion of
one; what it has is a growing message list. So the turns are derived by
counting human messages: every checkpoint whose message list holds *n* of them
belongs to turn *n*, and the last such checkpoint is where that turn ended.

**Retention is the checkpointer's, not ours.** An in-process saver keeps
everything for the life of the process; a PostgreSQL one keeps whatever it has
not been pruned of. So "turn 2 is not here" has two meanings, and
:func:`turns_of` reports both counts so a caller can tell them apart: a turn
that never existed is a mistake, and a turn that has been evicted is a fact
about the deployment.
"""

from collections.abc import Sequence
from typing import Any, NamedTuple, cast

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

    ``total`` is counted from the thread's *current* messages, which survive
    however far back the checkpoints go — so ``total > len(turns)`` is the
    signal that something was evicted rather than never written.
    """

    turns: list[Turn]
    total: int

    def find(self, n: int) -> Turn | None:
        return next((turn for turn in self.turns if turn.n == n), None)


def _humans(messages: Sequence[Any]) -> list[Any]:
    return [message for message in messages if getattr(message, "type", None) == HUMAN]


def _text(message: Any) -> str:
    """A question as text, however the client sent its content."""
    text = getattr(message, "text", None)
    return text if isinstance(text, str) else str(getattr(message, "content", ""))


async def turns_of(agent: Any, thread_id: str) -> History:
    """Every retained turn of a thread, oldest first.

    Walks the whole checkpoint history, which is proportional to the length of
    the conversation rather than to what is being asked for. That is the honest
    cost of deriving turns from a structure that does not record them; a thread
    long enough for it to matter has other problems first.
    """
    config = cast(Any, {"configurable": {"thread_id": thread_id}})

    # A turn's *last* checkpoint is the one holding everything it published,
    # and LangGraph yields newest first — so the first checkpoint seen for a
    # given question count is that turn's last. `setdefault` keeps it.
    ending: dict[int, Any] = {}
    async for snapshot in agent.aget_state_history(config):
        values = getattr(snapshot, "values", None) or {}
        if count := len(_humans(values.get("messages") or [])):
            ending.setdefault(count, snapshot)

    current = await agent.aget_state(config)
    asked = _humans((getattr(current, "values", None) or {}).get("messages") or [])

    def turn(n: int, snapshot: Any) -> Turn:
        values = getattr(snapshot, "values", None) or {}
        configurable = (getattr(snapshot, "config", None) or {}).get("configurable", {})
        return Turn(
            n=n,
            # From the current messages rather than the checkpoint's, so a turn
            # whose own checkpoints were evicted could still be named.
            question=_text(asked[n - 1]) if n <= len(asked) else "",
            checkpoint_id=configurable.get("checkpoint_id"),
            state=dict(values.get(TOOL_STATE_KEY) or {}),
        )

    return History(
        turns=[turn(n, snapshot) for n, snapshot in sorted(ending.items())],
        total=len(asked),
    )
