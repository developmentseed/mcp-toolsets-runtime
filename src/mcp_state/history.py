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
vocabulary it does have — turn numbers to ``{key: entry}`` mappings — so a host
supplies one from whatever it stores conversations in, and this package still
knows nothing about how.

``total`` alongside ``turns`` is what carries the difference between *a turn
the thread never had* and *a turn that has been pruned*. The first is a
miscount and the second is a fact about the deployment, and a model that
cannot tell them apart will treat "gone" as "no". So the host reports both
counts and the distinction is drawn here, without either side naming a
checkpointer.
"""

from collections.abc import Mapping
from typing import NamedTuple, Protocol

from mcp_state.state import StateEntry

#: One turn's session state, by key, as that turn ended.
TurnState = Mapping[str, StateEntry]


class Snapshots(NamedTuple):
    """Retained turns of one thread, and how many it has had.

    ``turns`` is 1-based and sparse: a host whose store has been pruned
    supplies only what it still holds, and ``total`` says how many there were.
    ``total > len(turns)`` is therefore the signal that something was evicted
    rather than never written.
    """

    turns: Mapping[int, TurnState]
    total: int

    def at(self, n: int) -> TurnState | None:
        """Turn ``n``'s state, or ``None`` if it is not available."""
        return self.turns.get(n)

    def never_had(self, n: int) -> bool:
        """Whether the thread never had turn ``n``, as against no longer holds it.

        The whole reason both counts are carried. A caller that only checked
        ``at(n) is None`` would report a pruned value as one that never
        existed, which is the answer most likely to be believed and most likely
        to be wrong.
        """
        return n < 1 or n > self.total


class ThreadHistory(Protocol):
    """A host's answer to "what did this thread's state look like at each turn".

    One method rather than a read-at-turn plus a version count, because both
    answers come out of the same walk and a host deriving turns from a
    checkpoint log pays for the whole conversation either way.
    """

    async def snapshots(self, thread_id: str) -> Snapshots:
        """Every retained turn of ``thread_id``, and how many it has had."""
        ...
