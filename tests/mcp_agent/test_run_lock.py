"""The run lock, on its own: in-process, deferred, and on PostgreSQL.

The Postgres tests need a real database, because what is under test is how
advisory locks behave between sessions. Set ``MCP_AGENT_TEST_POSTGRES`` to a
DSN to run them; they are skipped otherwise. Two lock instances stand in for
two replicas: each has its own connection, which is all that separates two
processes as far as Postgres is concerned.
"""

import asyncio
import os
import uuid

import pytest

from mcp_agent.main import Checkpointing
from mcp_agent.run_lock import (
    DeferredRunLock,
    InProcessRunLock,
    PostgresRunLock,
    RunLock,
)

POSTGRES = os.environ.get("MCP_AGENT_TEST_POSTGRES", "")
needs_postgres = pytest.mark.skipif(
    not POSTGRES, reason="set MCP_AGENT_TEST_POSTGRES to a DSN to run"
)


# --- in-process -------------------------------------------------------------


async def test_a_claimed_thread_cannot_be_claimed_again():
    lock = InProcessRunLock()

    assert await lock.claim("t1")
    assert not await lock.claim("t1")
    assert await lock.claim("t2")


async def test_a_released_thread_can_be_claimed_again():
    lock = InProcessRunLock()
    await lock.claim("t1")
    await lock.release("t1")

    assert not await lock.running("t1")
    assert await lock.claim("t1")


async def test_releasing_a_thread_nobody_holds_does_nothing():
    await InProcessRunLock().release("never")


async def test_a_waiter_is_woken_by_the_release():
    lock = InProcessRunLock()
    await lock.claim("t1")
    waiting = asyncio.create_task(lock.released("t1"))
    await asyncio.sleep(0)
    assert not waiting.done()

    await lock.release("t1")
    await asyncio.wait_for(waiting, 1)


async def test_waiting_on_a_free_thread_returns_at_once():
    await asyncio.wait_for(InProcessRunLock().released("t1"), 1)


# --- following the checkpointer ---------------------------------------------


async def test_the_in_memory_checkpointer_gets_an_in_process_lock():
    """Its threads live in this process, so nothing else can run them."""
    checkpointing = Checkpointing("memory")

    assert isinstance(await checkpointing._run_lock(), InProcessRunLock)


async def test_a_postgres_checkpointer_gets_a_postgres_lock():
    """Decided without connecting: the lock opens its connection on first use."""
    checkpointing = Checkpointing("postgresql://db/agent")

    assert isinstance(await checkpointing._run_lock(), PostgresRunLock)


async def test_the_lock_is_decided_on_first_use_not_on_asking():
    """A router is built before the lifespan reads the target, so asking for the
    lock must not read it: a bad value should fail at startup, as it does now."""
    lock: RunLock = Checkpointing("not a target").run_lock()

    assert isinstance(lock, DeferredRunLock)
    with pytest.raises(ValueError):
        await lock.claim("t1")


# --- postgres ---------------------------------------------------------------


async def test_an_unreachable_database_fails_open():
    """A database blip must not refuse every run. The claim falls back to this
    process, which is what the service did before there was a lock."""
    lock = PostgresRunLock("postgresql://nobody@127.0.0.1:1/none?connect_timeout=1")

    assert await lock.claim("t1")
    assert not await lock.claim("t1")
    await lock.release("t1")
    assert not await lock.running("t1")


def _thread() -> str:
    # Unique per test, so a lock left by a failed run cannot leak into another.
    return f"test-{uuid.uuid4().hex}"


@needs_postgres
async def test_two_replicas_exclude_each_other():
    one, two = PostgresRunLock(POSTGRES), PostgresRunLock(POSTGRES)
    thread = _thread()
    try:
        assert await one.claim(thread)
        assert not await two.claim(thread)
        assert await two.running(thread)

        await one.release(thread)
        assert not await two.running(thread)
        assert await two.claim(thread)
    finally:
        await one.aclose()
        await two.aclose()


@needs_postgres
async def test_one_replica_cannot_claim_its_own_thread_twice():
    """Advisory locks are re-entrant in a session: without the in-process check
    two runs in one process would both be granted it."""
    lock = PostgresRunLock(POSTGRES)
    thread = _thread()
    try:
        assert await lock.claim(thread)
        assert not await lock.claim(thread)
    finally:
        await lock.aclose()


@needs_postgres
async def test_a_waiter_on_one_replica_hears_a_release_on_another():
    one, two = PostgresRunLock(POSTGRES), PostgresRunLock(POSTGRES)
    two.poll = 0.05
    thread = _thread()
    try:
        await one.claim(thread)
        waiting = asyncio.create_task(two.released(thread))
        await asyncio.sleep(0.2)
        assert not waiting.done()

        await one.release(thread)
        await asyncio.wait_for(waiting, 2)
    finally:
        await one.aclose()
        await two.aclose()


@needs_postgres
async def test_a_replica_that_goes_away_takes_its_locks_with_it():
    """A pod that dies closes its connection, and Postgres lets its locks go:
    nothing waits forever behind a dead process."""
    one, two = PostgresRunLock(POSTGRES), PostgresRunLock(POSTGRES)
    two.poll = 0.05
    thread = _thread()
    try:
        await one.claim(thread)
        waiting = asyncio.create_task(two.released(thread))

        await one.aclose()
        await asyncio.wait_for(waiting, 2)
        assert await two.claim(thread)
    finally:
        await two.aclose()
