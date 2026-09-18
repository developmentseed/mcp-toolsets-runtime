"""One run per thread.

``POST /runs`` claims the thread before it starts and releases it when the
response ends. A second run on a claimed thread is refused (#96).

:class:`InProcessRunLock` covers one process. :class:`PostgresRunLock` covers
every process sharing a PostgreSQL database.
:meth:`mcp_agent.main.Checkpointing.run_lock` returns the one matching the
checkpointer.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class RunLock(Protocol):
    """Claims, releases and reports runs, by thread id."""

    async def claim(self, thread_id: str) -> bool:
        """Claim the thread. ``False`` if a run already holds it."""
        ...

    async def release(self, thread_id: str) -> None:
        """Release the thread. Does nothing if it is not held."""
        ...

    async def running(self, thread_id: str) -> bool:
        """Whether a run holds the thread."""
        ...

    async def released(self, thread_id: str) -> None:
        """Return when no run holds the thread."""
        ...


class InProcessRunLock:
    """Threads claimed in this process.

    ``claim`` checks and inserts with no ``await`` in between, so it is atomic
    within the event loop.
    """

    def __init__(self) -> None:
        # Each held thread's event is set when it is released.
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
        # Loops because another run can claim the thread before this wakes.
        while (event := self._held.get(thread_id)) is not None:
            await event.wait()


# The advisory lock key is hashtextextended(thread_id, 0).
_CLAIM = "SELECT pg_try_advisory_lock(hashtextextended(%s, 0))"
_UNLOCK = "SELECT pg_advisory_unlock(hashtextextended(%s, 0))"

#: Whether any session holds the key. ``pg_locks`` stores a 64-bit key as
#: ``classid`` (high 32 bits) and ``objid`` (low 32 bits), with ``objsubid = 1``.
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


@dataclass
class _Watch:
    """One poll of ``pg_locks`` for a thread, shared by its waiters."""

    task: asyncio.Task[None]
    waiters: int = 0


class PostgresRunLock:
    """Threads claimed across every process sharing one PostgreSQL database.

    Each claim is a session-level advisory lock, taken on one connection per
    instance. Postgres releases a connection's locks when it closes.

    Advisory locks are re-entrant within a session, so each claim is checked
    against an :class:`InProcessRunLock` first.

    If a query fails, ``claim`` returns the in-process result, ``release``
    releases in-process, and ``running`` returns the in-process result. Each
    failure is logged.

    ``released`` waits on the in-process lock, then polls ``pg_locks`` every
    :attr:`poll` seconds. Waiters on one thread share one poll.
    """

    #: Seconds between polls of ``pg_locks``.
    poll = 1.0

    def __init__(self, url: str) -> None:
        self._url = url
        self._local = InProcessRunLock()
        self._connection: Any = None
        self._connecting = asyncio.Lock()
        self._watches: dict[str, _Watch] = {}

    async def _query(self, sql: str, thread_id: str) -> Any:
        async with self._connecting:
            if self._connection is None or self._connection.closed:
                from psycopg import AsyncConnection

                self._connection = await AsyncConnection.connect(
                    self._url, autocommit=True
                )
        cursor = await self._connection.execute(sql, (thread_id,))
        row = await cursor.fetchone()
        return row[0] if row else None

    async def _scalar(self, sql: str, thread_id: str) -> Any:
        # Shielded so that a cancelled caller does not interrupt a query on the
        # shared connection.
        return await asyncio.shield(self._query(sql, thread_id))

    async def claim(self, thread_id: str) -> bool:
        if not await self._local.claim(thread_id):
            return False
        try:
            taken = await self._scalar(_CLAIM, thread_id)
        except Exception:
            logger.warning(
                "run lock: advisory lock failed; thread %s claimed in-process only",
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
            logger.warning(
                "run lock: advisory unlock failed for thread %s",
                thread_id,
                exc_info=True,
            )
        finally:
            await self._local.release(thread_id)

    async def running(self, thread_id: str) -> bool:
        if await self._local.running(thread_id):
            return True
        try:
            return bool(await self._scalar(_HELD, thread_id))
        except Exception:
            logger.warning("run lock: reading pg_locks failed", exc_info=True)
            return False

    async def _poll(self, thread_id: str) -> None:
        while await self.running(thread_id):
            await asyncio.sleep(self.poll)

    async def released(self, thread_id: str) -> None:
        await self._local.released(thread_id)
        watch = self._watches.get(thread_id)
        if watch is None:
            watch = _Watch(asyncio.create_task(self._poll(thread_id)))
            self._watches[thread_id] = watch
        watch.waiters += 1
        try:
            await asyncio.shield(watch.task)
        finally:
            watch.waiters -= 1
            if watch.waiters == 0:
                watch.task.cancel()
                self._watches.pop(thread_id, None)

    async def aclose(self) -> None:
        """Close the connection. Postgres releases its locks."""
        if self._connection is not None:
            await self._connection.close()
            self._connection = None


class DeferredRunLock:
    """Delegates to the lock ``resolve`` returns, resolved on each call.

    :meth:`mcp_agent.main.Checkpointing.run_lock` returns one, so the
    checkpoint target is read on first use rather than when the router is
    built.
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
