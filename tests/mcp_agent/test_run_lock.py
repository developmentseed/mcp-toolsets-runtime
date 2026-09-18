"""Run locks: in-process, deferred, and PostgreSQL.

The PostgreSQL tests run when ``MCP_AGENT_TEST_POSTGRES`` holds a DSN and are
skipped otherwise. Each lock instance has its own connection, so two instances
act as two processes.
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


async def test_a_waiter_returns_when_the_thread_is_released():
    lock = InProcessRunLock()
    await lock.claim("t1")
    waiting = asyncio.create_task(lock.released("t1"))
    await asyncio.sleep(0)
    assert not waiting.done()

    await lock.release("t1")
    await asyncio.wait_for(waiting, 1)


async def test_a_waiter_keeps_waiting_if_the_thread_is_claimed_again():
    lock = InProcessRunLock()
    await lock.claim("t1")
    waiting = asyncio.create_task(lock.released("t1"))
    await asyncio.sleep(0)

    await lock.release("t1")
    await lock.claim("t1")
    await asyncio.sleep(0.01)
    assert not waiting.done()

    await lock.release("t1")
    await asyncio.wait_for(waiting, 1)


async def test_waiting_on_a_free_thread_returns_at_once():
    await asyncio.wait_for(InProcessRunLock().released("t1"), 1)


# --- Checkpointing.run_lock -------------------------------------------------


async def test_the_in_memory_target_gets_an_in_process_lock():
    checkpointing = Checkpointing("memory")

    assert isinstance(await checkpointing._run_lock(), InProcessRunLock)


async def test_a_postgres_target_gets_a_postgres_lock_without_connecting():
    checkpointing = Checkpointing("postgresql://db/agent")

    assert isinstance(await checkpointing._run_lock(), PostgresRunLock)


async def test_the_target_is_read_on_first_use():
    lock: RunLock = Checkpointing("not a target").run_lock()

    assert isinstance(lock, DeferredRunLock)
    with pytest.raises(ValueError):
        await lock.claim("t1")


# --- PostgreSQL -------------------------------------------------------------


async def test_an_unreachable_database_falls_back_to_in_process():
    lock = PostgresRunLock("postgresql://nobody@127.0.0.1:1/none?connect_timeout=1")

    assert await lock.claim("t1")
    assert not await lock.claim("t1")
    await lock.release("t1")
    assert not await lock.running("t1")


def _thread() -> str:
    return f"test-{uuid.uuid4().hex}"


@needs_postgres
async def test_two_instances_exclude_each_other():
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
async def test_one_instance_cannot_claim_a_thread_twice():
    lock = PostgresRunLock(POSTGRES)
    thread = _thread()
    try:
        assert await lock.claim(thread)
        assert not await lock.claim(thread)
    finally:
        await lock.aclose()


@needs_postgres
async def test_a_waiter_returns_when_another_instance_releases():
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
async def test_waiters_on_one_thread_share_one_poll():
    one, two = PostgresRunLock(POSTGRES), PostgresRunLock(POSTGRES)
    two.poll = 0.05
    thread = _thread()
    try:
        await one.claim(thread)
        waiters = [asyncio.create_task(two.released(thread)) for _ in range(3)]
        await asyncio.sleep(0.1)
        assert len(two._watches) == 1
        assert two._watches[thread].waiters == 3

        await one.release(thread)
        await asyncio.wait_for(asyncio.gather(*waiters), 2)
        assert two._watches == {}
    finally:
        await one.aclose()
        await two.aclose()


@needs_postgres
async def test_a_cancelled_waiter_is_removed_and_the_lock_still_works():
    one, two = PostgresRunLock(POSTGRES), PostgresRunLock(POSTGRES)
    two.poll = 0.01
    thread, other = _thread(), _thread()
    try:
        await one.claim(thread)
        waiting = asyncio.create_task(two.released(thread))
        await asyncio.sleep(0.1)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting

        assert two._watches == {}
        assert await two.claim(other)
        assert await two.running(thread)
    finally:
        await one.aclose()
        await two.aclose()


@needs_postgres
async def test_closing_an_instance_releases_its_locks():
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
