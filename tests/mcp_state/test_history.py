"""The seam a host fills in, and the three answers a turn-scoped read can give.

A fake ``ThreadHistory`` is the right thing here and a shortcut nowhere else:
the protocol *is* the boundary, so a hand-written one is what a host actually
supplies. The checkpoint-backed implementation is exercised against real turns
in ``tests/mcp_agent/test_across_turns.py``; what these pin down is what
``mcp_state`` does with whatever it is handed — including the case no
in-process saver can produce, which is a turn that has been pruned.
"""

import json

from mcp_state.history import Snapshots
from mcp_state.inspect import read_state_key, read_state_key_at_turn
from mcp_state.state import TOOL_STATE_KEY, merge_tool_state, rewritten, versions

KEY = "gazet/get_aoi/bbox"
THREAD = "t"


class FakeHistory:
    """What a host with its own store would write."""

    def __init__(self, snapshots: Snapshots) -> None:
        self._snapshots = snapshots

    async def snapshots(self, thread_id: str) -> Snapshots:
        assert thread_id == THREAD
        return self._snapshots


def _entry(value: object) -> dict:
    return {"value": value, "tool": "get_aoi"}


COPENHAGEN = [12.4, 55.5, 12.7, 55.8]
AARHUS = [9.9, 56.1, 10.3, 56.2]


async def test_a_retained_turn_reads_back_as_that_turn_left_it() -> None:
    history = FakeHistory(
        Snapshots({1: {KEY: _entry(COPENHAGEN)}, 2: {KEY: _entry(AARHUS)}}, total=2)
    )

    assert (
        json.loads(await read_state_key_at_turn(KEY, 1, THREAD, history)) == COPENHAGEN
    )


async def test_narrowing_works_the_same_at_a_turn_as_it_does_now() -> None:
    """The turn only chooses which state; everything downstream is unchanged."""
    state = {KEY: _entry({"bbox": COPENHAGEN, "name": "København"})}
    history = FakeHistory(Snapshots({1: state}, total=1))

    assert (
        json.loads(await read_state_key_at_turn(KEY, 1, THREAD, history, path="bbox"))
        == COPENHAGEN
    )
    assert "København" in await read_state_key_at_turn(
        KEY, 1, THREAD, history, pattern="Kø"
    )


async def test_a_pruned_turn_is_not_reported_as_one_that_never_existed() -> None:
    """The distinction the whole seam carries ``total`` for.

    "It is gone" and "it never was" lead a model to opposite answers, and the
    wrong one of the two is the one it can state confidently.
    """
    history = FakeHistory(Snapshots({3: {KEY: _entry(AARHUS)}}, total=3))

    answer = json.loads(await read_state_key_at_turn(KEY, 1, THREAD, history))
    assert answer["turn_no_longer_retained"] == 1
    assert answer["retained_turns"] == [3]
    assert "existed" in answer["reason"]

    missed = json.loads(await read_state_key_at_turn(KEY, 9, THREAD, history))
    assert missed["no_such_turn"] == 9
    assert missed["turns_so_far"] == 3


async def test_turn_zero_is_a_miscount_not_a_read() -> None:
    """Turns are the user's questions counted from 1, so 0 names nothing."""
    history = FakeHistory(Snapshots({1: {KEY: _entry(COPENHAGEN)}}, total=1))

    assert "no_such_turn" in await read_state_key_at_turn(KEY, 0, THREAD, history)


async def test_a_thread_id_without_a_history_answers_like_a_host_without_one() -> None:
    """Both mean the same thing to a model: nothing here is keeping the past."""
    history = FakeHistory(Snapshots({1: {KEY: _entry(COPENHAGEN)}}, total=1))

    assert "no_turn_history" in await read_state_key_at_turn(KEY, 1, None, history)
    assert "no_turn_history" in await read_state_key_at_turn(KEY, 1, THREAD, None)


async def test_an_unknown_key_at_a_turn_lists_what_that_turn_held() -> None:
    """Not what state holds now — the answer has to be about the turn asked for."""
    history = FakeHistory(
        Snapshots({1: {KEY: _entry(COPENHAGEN)}, 2: {"gazet/other/x": _entry(1)}}, 2)
    )

    answer = json.loads(await read_state_key_at_turn(KEY, 2, THREAD, history))
    assert answer["unknown_or_empty_key"] == KEY
    assert list(answer["available_keys"]) == ["gazet/other/x"]


def test_the_reducer_counts_values_a_key_has_held_not_writes_to_it() -> None:
    """A republished identical value loses nothing, so it is not a rewrite.

    The count exists to tell a model when the value in front of it differs
    from the one an earlier turn used. Counting writes instead would raise the
    flag on every idempotent tool, and a flag that is always up is ignored.
    """
    state = merge_tool_state(None, {KEY: _entry(COPENHAGEN)})
    assert versions(state[KEY]) == 1
    assert not rewritten(state[KEY])

    state = merge_tool_state(state, {KEY: _entry(COPENHAGEN)})
    assert not rewritten(state[KEY])

    state = merge_tool_state(state, {KEY: _entry(AARHUS)})
    assert versions(state[KEY]) == 2
    assert rewritten(state[KEY])

    # Ordering is still the reducer's other job, and unaffected.
    assert state[KEY]["seq"] == 3


def test_an_entry_written_before_the_count_existed_reads_as_one_version() -> None:
    """A checkpoint restored from an earlier release must not claim a rewrite."""
    assert versions({"value": COPENHAGEN}) == 1
    assert versions(None) == 1


def test_a_key_that_lost_a_value_still_says_so_after_the_value_is_gone() -> None:
    """Counting at merge time is what survives the checkpointer's pruning.

    The entry the model is looking at carries the count, so "this was replaced"
    holds even where the replaced value can no longer be fetched — which is
    exactly when saying it matters most.
    """
    state = merge_tool_state(None, {KEY: _entry(COPENHAGEN)})
    state = merge_tool_state(state, {KEY: _entry(AARHUS)})

    note = read_state_key(KEY, {TOOL_STATE_KEY: state})
    assert "held 2 different values" in note
