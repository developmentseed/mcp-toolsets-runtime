"""What a host can tell this package about a thread's past.

Session state holds one value per key. When a later tool call republishes a
key, :func:`~mcp_state.state.merge_tool_state` overwrites it — so a model
asked to compare what it found first with what it has now reads the key,
receives a well-formed value, and compares the current value with itself.

Nothing has to be stored to fix that: a host running the agent under a
checkpointer already retains every past value. What is missing is a way to
*ask* for one, in terms this package can express.

**This is that seam, and it is deliberately narrow.** ``mcp_state`` knows a
dict of entries and the tools that read them; it has never known about
threads, checkpoints or retention. A :class:`ThreadHistory` is described in the
vocabulary it does have — a turn number and a ``{key: entry}`` mapping — so a
host supplies one from whatever it stores conversations in, and this package
still knows nothing about how.

``total`` alongside the state is what carries the difference between *a turn
the thread never had* and *a turn that has been pruned*. The first is a
miscount and the second is a fact about the deployment, and a model that
cannot tell them apart will treat "gone" as "no". So the host reports the
count and the distinction is drawn here, without either side naming a
checkpointer.
"""

from collections.abc import Mapping
from typing import NamedTuple, Protocol

from mcp_state.state import StateEntry

#: One turn's session state, by key, as that turn ended.
TurnState = Mapping[str, StateEntry]


class Snapshot(NamedTuple):
    """One turn's session state, and what the thread holds around it.

    ``state`` is ``None`` in two cases that mean opposite things, and
    :meth:`never_had` is what separates them: a turn the thread never had is a
    miscount, and a turn that has been pruned is a fact about the deployment.

    ``retained`` carries turn numbers and no state, so it costs nothing to
    return them all. It is what a pruned answer offers instead of a value:
    which turns are still worth asking about.
    """

    state: TurnState | None
    retained: frozenset[int]
    total: int

    def never_had(self, n: int) -> bool:
        """Whether the thread never had turn ``n``, as against no longer holds it.

        The whole reason ``total`` is carried alongside the state. A caller
        that only checked ``state is None`` would report a pruned value as one
        that never existed, which is the answer most likely to be believed and
        most likely to be wrong.
        """
        return n < 1 or n > self.total


class ThreadHistory(Protocol):
    """A host's answer to "what did this thread's state hold at turn *n*".

    One method rather than a read-at-turn plus a version count, because both
    answers come out of the same walk and a host deriving turns from a
    checkpoint log pays for the whole conversation either way. It takes the
    turn all the same: the walk is no cheaper for knowing it, but what has to
    be *kept* from the walk is one turn's state rather than every turn's, and
    on a thread carrying geometries those differ by an order of magnitude.
    """

    async def snapshot(self, thread_id: str, turn: int) -> Snapshot:
        """``turn``'s state, which turns survive, and how many there were."""
        ...
