"""The seam a host fills in, and the three answers a turn-scoped read can give.

A fake ``ThreadHistory`` is the right thing here and a shortcut nowhere else:
the protocol *is* the boundary, so a hand-written one is what a host actually
supplies. The checkpoint-backed implementation is exercised against real turns
in ``tests/mcp_agent/test_across_turns.py``; what these pin down is what
``mcp_state`` does with whatever it is handed — including the case no
in-process saver can produce, which is a turn that has been pruned.
"""

import json

from mcp_state.handles import available
from mcp_state.history import Snapshots
from mcp_state.inspect import read_state_key, read_state_key_at_turn
from mcp_state.state import TOOL_STATE_KEY, merge_tool_state, turns_written

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

    read = await read_state_key_at_turn(KEY, 1, THREAD, history)
    value, _, label = read.partition("\n")
    assert json.loads(value) == COPENHAGEN
    assert label == "[gazet/get_aoi/bbox as it stood at the end of turn 1.]"


async def test_narrowing_works_the_same_at_a_turn_as_it_does_now() -> None:
    """The turn only chooses which state; everything downstream is unchanged."""
    state = {KEY: _entry({"bbox": COPENHAGEN, "name": "København"})}
    history = FakeHistory(Snapshots({1: state}, total=1))

    narrowed = await read_state_key_at_turn(KEY, 1, THREAD, history, path="bbox")
    assert json.loads(narrowed.partition("\n")[0]) == COPENHAGEN
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


def test_the_count_is_turns_that_wrote_the_key_not_values_it_held() -> None:
    """It claims only what an entry can back up, and no more.

    Whether two turns hold *different* values needs the value each turn ended
    with, which is history and which an entry does not have. Deciding it from
    the previous write alone gets both directions wrong inside a turn — the
    reason this counts turns and lets the reader fetch one to find out.
    """

    def at(turn: int, value: object) -> dict:
        return {**_entry(value), "turn": turn}

    state = merge_tool_state(None, {KEY: at(1, COPENHAGEN)})
    assert turns_written(state[KEY]) == 1

    state = merge_tool_state(state, {KEY: at(2, AARHUS)})
    assert turns_written(state[KEY]) == 2

    # Same value, but a third turn holds it — a third fetchable version.
    state = merge_tool_state(state, {KEY: at(3, AARHUS)})
    assert turns_written(state[KEY]) == 3

    # Ordering is still the reducer's other job, and unaffected.
    assert state[KEY]["seq"] == 3


def test_a_second_write_inside_one_turn_does_not_add_a_version() -> None:
    """The count is in turns because a read is, and it must not promise more.

    A turn-scoped read resolves to what a turn *ended* holding, so a tool
    called twice before the same answer leaves its first value beyond reach of
    any turn. Counting it would name a version the model cannot fetch, which
    is worse than not mentioning it: the point of the number is that acting on
    it works.
    """

    def at(turn: int, value: object) -> dict:
        return {**_entry(value), "turn": turn}

    state = merge_tool_state(None, {KEY: at(1, COPENHAGEN)})
    state = merge_tool_state(state, {KEY: at(1, AARHUS)})
    assert turns_written(state[KEY]) == 1  # one turn, so one fetchable value

    state = merge_tool_state(state, {KEY: at(2, [10.4])})
    assert turns_written(state[KEY]) == 2  # turn 1 ended on AARHUS; turn 2 here


def test_an_unstamped_write_falls_back_to_counting_writes() -> None:
    """A host driving capture outside a graph has no turn to stamp.

    An unknown turn cannot be shown to be the same one, so the signal stays
    on. That deployment has no turn history to read either way; going quiet
    would lose the signal everywhere capture runs without a message list.
    """
    state = merge_tool_state(None, {KEY: _entry(COPENHAGEN)})
    state = merge_tool_state(state, {KEY: _entry(AARHUS)})

    assert turns_written(state[KEY]) == 2


def test_the_refusal_listing_marks_a_key_more_than_one_turn_wrote() -> None:
    """The listing a refused model chooses from offers handles, not reads.

    A handle resolves to the present, so picking the right key is no help if
    the wrong version of it comes back — and this listing is where the choice
    is made.
    """
    state = merge_tool_state(None, {KEY: {**_entry(COPENHAGEN), "turn": 1}})
    state = merge_tool_state(state, {KEY: {**_entry(AARHUS), "turn": 2}})

    line = available(state)[0]
    assert line.startswith(f"@state:{KEY} — ")
    assert "written in 2 turns" in line
    assert "from get_aoi" in line


def test_a_key_one_turn_wrote_keeps_its_listing_line_short() -> None:
    state = merge_tool_state(None, {KEY: {**_entry(COPENHAGEN), "turn": 1}})

    assert "turns" not in available(state)[0]


def test_an_entry_written_before_the_count_existed_reads_as_one_turn() -> None:
    """A checkpoint restored from an earlier release must not claim a rewrite."""
    assert turns_written({"value": COPENHAGEN}) == 1
    assert turns_written(None) == 1


def test_a_key_that_lost_a_value_still_says_so_after_the_value_is_gone() -> None:
    """Counting at merge time is what survives the checkpointer's pruning.

    The entry the model is looking at carries the count, so "this was replaced"
    holds even where the replaced value can no longer be fetched — which is
    exactly when saying it matters most.
    """
    state = merge_tool_state(None, {KEY: _entry(COPENHAGEN)})
    state = merge_tool_state(state, {KEY: _entry(AARHUS)})

    note = read_state_key(KEY, {TOOL_STATE_KEY: state})
    assert "2 turns of this conversation wrote this key" in note


async def test_a_turn_scoped_read_names_its_turn_and_claims_no_latest() -> None:
    """A historical read must not repeat the present-tense note.

    "The above is the latest" is the whole point of that note on a read of
    now, and false the moment the value came out of an earlier turn. Its count
    would mislead too: an entry records how many turns had written the key by
    then, so turn 2 reports 2 however many have written it since.
    """
    at_two = {**_entry(AARHUS), "turns_written": 2}
    history = FakeHistory(
        Snapshots({1: {KEY: _entry(COPENHAGEN)}, 2: {KEY: at_two}}, total=2)
    )

    read = await read_state_key_at_turn(KEY, 2, THREAD, history)
    assert "as it stood at the end of turn 2" in read
    assert "the above is the latest" not in read
    assert "turns of this conversation wrote this key" not in read

    # The same entry read as the present keeps the note it has always had.
    now = read_state_key(KEY, {TOOL_STATE_KEY: {KEY: at_two}})
    assert "the above is the latest" in now
    assert "as it stood at the end of" not in now
