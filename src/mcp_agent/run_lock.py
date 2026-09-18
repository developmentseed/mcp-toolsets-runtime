"""One run per thread.

Two runs on one thread at the same time do not interleave: each reads the
thread when it starts and writes it back when it ends, so the one that finishes
last replaces the other's turn, and both are told they succeeded (#96). The
routes refuse the second run instead, and this is what they ask.

**The lock only has to be as shared as the conversations are.** An in-memory
checkpointer keeps every thread inside one process, so a lock in that process
covers them all. A PostgreSQL checkpointer shares threads across replicas, so
the lock does too, and the database it needs is the one already there.
:meth:`mcp_agent.main.Checkpointing.run_lock` picks the one that matches.

A deployment with a checkpointer of its own and more than one replica should
pass a lock that spans them. Without one it gets :class:`InProcessRunLock`,
which refuses an overlap within a replica and none across them: today's
behaviour, never worse.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class RunLock(Protocol):
    """What the routes need: claim a thread, let it go, and say whether it is
    taken."""

    async def claim(self, thread_id: str) -> bool:
        """Take the thread for a run. ``False`` if a run already has it."""
        ...

    async def release(self, thread_id: str) -> None:
        """Let the thread go. Releasing one not held does nothing."""
        ...

    async def running(self, thread_id: str) -> bool:
        """Whether a run holds the thread now. Only looks."""
        ...

    async def released(self, thread_id: str) -> None:
        """Return once no run holds the thread, at once if none does."""
        ...


class InProcessRunLock:
    """Threads claimed by this process.

    A claim is a check and an insert with no ``await`` between them, so two
    requests in one event loop cannot both win.
    """

    def __init__(self) -> None:
        # One event per held thread, set when it is released, for the waiters.
        self._held: dict[str, asyncio.Event] = {}

    async def claim(self, thread_id: str) -> bool:
        if thread_id in self._held:
            return False
        self._held[thread_id] = asyncio.Event()
        return True

    async def release(self, thread_id: str) -> None:
        if (event := self._held.pop(thread_id, None)) is not None:
            event.set()

    async def running(self, thread_id: str) -> bool:
        return thread_id in self._held

    async def released(self, thread_id: str) -> None:
        if (event := self._held.get(thread_id)) is not None:
            await event.wait()


# A thread's advisory lock is keyed on ``hashtextextended(thread_id, 0)``, the
# 64-bit hash Postgres has built in, so every replica derives the same key.
_CLAIM = "SELECT pg_try_advisory_lock(hashtextextended(%s, 0))"
_UNLOCK = "SELECT pg_advisory_unlock(hashtextextended(%s, 0))"

#: Whether any session holds the thread's advisory lock. A 64-bit key appears
#: in ``pg_locks`` split in two: ``classid`` is the high 32 bits and ``objid``
#: the low 32, with ``objsubid = 1`` marking the one-key form.
_HELD = """
SELECT EXISTS (
    SELECT 1 FROM pg_locks
    WHERE locktype = 'advisory' AND granted AND objsubid = 1
      AND database = (SELECT oid FROM pg_database WHERE datname = current_database())
      AND classid = ((key.k >> 32) & 4294967295)::oid
      AND objid = (key.k & 4294967295)::oid
)
FROM (SELECT hashtextextended(%s, 0) AS k) AS key
"""


class PostgresRunLock:
    """Threads claimed across every replica sharing one PostgreSQL database.

    A session-level advisory lock per thread, on **one** connection per
    process. A connection can hold any number of them, so concurrent runs cost
    nothing extra, and when a process dies its connection goes and Postgres
    releases its locks with it.

    Advisory locks are re-entrant within a session: two runs in this process
    would both be granted the same one. So an :class:`InProcessRunLock` is asked
    first, and the advisory lock only settles it between processes.

    **It fails open.** If the database cannot be reached, a claim logs and
    proceeds on the in-process lock alone, which is what the service did before
    there was a lock at all. Refusing every run because a lock could not be
    taken would turn a database blip into an outage.

    Waiting on another replica's run is a re-check of ``pg_locks`` every
    :attr:`poll` seconds rather than a ``LISTEN``: it needs no second
    connection, and it notices a replica that died mid-run, which would never
    have sent a notification.
    """

    #: Seconds between checks while waiting on another replica's run.
    poll = 1.0

    def __init__(self, url: str) -> None:
        self._url = url
        self._local = InProcessRunLock()
        self._connection: Any = None
        self._connecting = asyncio.Lock()

    async def _scalar(self, sql: str, thread_id: str) -> Any:
        async with self._connecting:
            if self._connection is None or self._connection.closed:
                from psycopg import AsyncConnection

                self._connection = await AsyncConnection.connect(
                    self._url, autocommit=True
                )
        cursor = await self._connection.execute(sql, (thread_id,))
        row = await cursor.fetchone()
        return row[0] if row else None

    async def claim(self, thread_id: str) -> bool:
        if not await self._local.claim(thread_id):
            return False
        try:
            taken = await self._scalar(_CLAIM, thread_id)
        except Exception:
            logger.warning(
                "run lock: the database could not be reached, so thread %s is "
                "claimed in this process only",
                thread_id,
                exc_info=True,
            )
            return True
        if not taken:
            await self._local.release(thread_id)
        return bool(taken)

    async def release(self, thread_id: str) -> None:
        try:
            await self._scalar(_UNLOCK, thread_id)
        except Exception:
            # A connection that dropped took its locks with it, so there is
            # nothing left to release.
            logger.warning(
                "run lock: could not release thread %s", thread_id, exc_info=True
            )
        finally:
            await self._local.release(thread_id)

    async def running(self, thread_id: str) -> bool:
        if await self._local.running(thread_id):
            return True
        try:
            return bool(await self._scalar(_HELD, thread_id))
        except Exception:
            logger.warning("run lock: could not read pg_locks", exc_info=True)
            return False

    async def released(self, thread_id: str) -> None:
        # A run in this process wakes its waiters the moment it ends.
        await self._local.released(thread_id)
        while await self.running(thread_id):
            await asyncio.sleep(self.poll)

    async def aclose(self) -> None:
        """Close the connection, which releases every lock it holds."""
        if self._connection is not None:
            await self._connection.close()
            self._connection = None


class DeferredRunLock:
    """A lock decided on first use rather than on construction.

    For :meth:`mcp_agent.main.Checkpointing.run_lock`: a router is built before
    the lifespan that reads the checkpoint target, and reading it early would
    move a misconfiguration from startup to import.
    """

    def __init__(self, resolve: Callable[[], Awaitable[RunLock]]) -> None:
        self._resolve = resolve

    async def claim(self, thread_id: str) -> bool:
        return await (await self._resolve()).claim(thread_id)

    async def release(self, thread_id: str) -> None:
        await (await self._resolve()).release(thread_id)

    async def running(self, thread_id: str) -> bool:
        return await (await self._resolve()).running(thread_id)

    async def released(self, thread_id: str) -> None:
        await (await self._resolve()).released(thread_id)
