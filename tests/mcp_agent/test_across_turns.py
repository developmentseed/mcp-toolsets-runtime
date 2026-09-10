"""Reading a state key as it stood at an earlier turn.

Session state holds one value per key, so a tool called twice leaves the model
looking at the second value with nothing to say the first ever existed. These
run *real* turns against a real checkpointer rather than assembling snapshots,
because the whole claim is that the earlier value is already retained — a fake
history would prove the plumbing and not the premise.

The publisher writes a different bounding box each call for the same reason the
sibling suite's does: with one value, "as of turn 1" and "as of now" are the
same bytes and the test would pass whether or not any of this worked.
"""

from typing import Any, cast

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver

from mcp_agent.history import CheckpointHistory, _config, turns_from, turns_of
from mcp_agent.main import with_session_state
from mcp_runtime.declarations import PRODUCES_META_KEY
from tests.mcp_agent.test_streaming import StreamingScriptedModel, _tool_call

KEY = "gazet/get_aoi/bbox"
THREAD = "compare-across-turns"

#: One box per place, so which turn a read landed on is visible in the digits.
BOXES = {
    "Copenhagen": [12.4, 55.5, 12.7, 55.8],
    "Aarhus": [9.9, 56.1, 10.3, 56.2],
    "Odense": [10.2, 55.3, 10.5, 55.5],
}


def _gazetteer() -> StructuredTool:
    """Publishes a different bbox under the same key on every call."""

    async def call(place: str) -> tuple[str, dict[str, Any]]:
        return f"found {place}", {
            "structured_content": {"message": f"found {place}", "bbox": BOXES[place]}
        }

    return StructuredTool(
        name="get_aoi",
        description="look a place up",
        args_schema={
            "type": "object",
            "properties": {"place": {"type": "string"}},
            "required": ["place"],
        },
        coroutine=call,
        response_format="content_and_artifact",
        metadata={"_meta": {PRODUCES_META_KEY: [{"stateKey": KEY, "field": "bbox"}]}},
    )


def _agent(script: list[Any], saver: InMemorySaver | None = None) -> Any:
    return with_session_state(
        StreamingScriptedModel(script=script),
        [_gazetteer()],
        saver if saver is not None else InMemorySaver(),
    )


async def _ask(agent: Any, question: str) -> dict[str, Any]:
    return await agent.ainvoke(
        {"messages": [HumanMessage(question)]},
        {"configurable": {"thread_id": THREAD}},
    )


def _read(result: dict[str, Any]) -> str:
    """What the last ``inspect_state`` call returned to the model."""
    reads = [
        message
        for message in result["messages"]
        if isinstance(message, ToolMessage) and message.name == "inspect_state"
    ]
    return str(reads[-1].content)


async def test_the_model_can_compare_a_value_with_its_own_earlier_version() -> None:
    """The issue this exists for: turn 3 asks what turn 1 found, and is told.

    Without a turn to read at, the second lookup has overwritten the first and
    the model compares Aarhus with Aarhus — a well-formed answer to a question
    nobody asked.
    """
    agent = _agent(
        [
            _tool_call("get_aoi", "c1", {"place": "Copenhagen"}),
            AIMessage(content="Copenhagen is here."),
            AIMessage(content="Yes, it is a city."),
            _tool_call("get_aoi", "c2", {"place": "Aarhus"}),
            _tool_call("inspect_state", "c3", {"key": KEY, "turn": 1}),
            AIMessage(content="Aarhus is west and north of it."),
        ]
    )

    await _ask(agent, "where is Copenhagen")
    await _ask(agent, "is that a city")
    third = await _ask(agent, "and Aarhus — how does it compare with the first")

    # The value as it stood at the end of turn 1, not the one in state now.
    assert "12.4" in _read(third)
    assert "9.9" not in _read(third)
    # State itself is untouched: the read is a view of the past, not a rewind.
    assert third["tool_state"][KEY]["value"] == BOXES["Aarhus"]


async def test_a_plain_read_says_an_earlier_turn_wrote_the_key_too() -> None:
    """The part that makes the other part get used.

    A model will not ask for turn 1 unless something tells it turn 1 differs,
    and the only surface it sees is the read itself.
    """
    agent = _agent(
        [
            _tool_call("get_aoi", "c1", {"place": "Copenhagen"}),
            AIMessage(content="Copenhagen is here."),
            _tool_call("get_aoi", "c2", {"place": "Aarhus"}),
            _tool_call("inspect_state", "c3", {"key": KEY}),
            AIMessage(content="Aarhus is here."),
        ]
    )
    await _ask(agent, "where is Copenhagen")
    second = await _ask(agent, "and Aarhus")

    read = _read(second)
    assert "9.9" in read  # still the current value, still returned whole
    assert "2 turns of this conversation wrote this key" in read
    assert "turn=" in read


async def test_the_key_listing_marks_a_key_more_than_one_turn_wrote() -> None:
    agent = _agent(
        [
            _tool_call("get_aoi", "c1", {"place": "Copenhagen"}),
            AIMessage(content="ok"),
            _tool_call("get_aoi", "c2", {"place": "Aarhus"}),
            _tool_call("inspect_state", "c3", {"key": "*"}),
            AIMessage(content="ok"),
        ]
    )
    await _ask(agent, "where is Copenhagen")
    second = await _ask(agent, "and Aarhus")

    assert "written in 2 turns" in _read(second)


async def test_two_calls_in_one_turn_count_as_the_one_value_a_turn_can_reach() -> None:
    """The count and the read have to be in the same unit, which is turns.

    A tool called twice before the same answer leaves its first value beyond
    reach: state holds one value per key, and a turn-scoped read resolves to
    what the turn *ended* holding. So the count says two, not three, and both
    of the two are fetchable — a model acting on the number never asks for a
    turn that is not there.
    """
    saver = InMemorySaver()
    agent = _agent(
        [
            # Turn 1 publishes twice. Only its second value survives the turn.
            _tool_call("get_aoi", "c1", {"place": "Copenhagen"}),
            _tool_call("get_aoi", "c2", {"place": "Aarhus"}),
            AIMessage(content="two places."),
            _tool_call("get_aoi", "c3", {"place": "Odense"}),
            _tool_call("inspect_state", "c4", {"key": KEY}),
            AIMessage(content="three places."),
        ],
        saver,
    )
    await _ask(agent, "Copenhagen and Aarhus")
    second = await _ask(agent, "and Odense")

    assert second["tool_state"][KEY]["turns_written"] == 2
    assert "2 turns of this conversation wrote this key" in _read(second)

    # Every version the count names resolves, and to a distinct value. One
    # call per turn, which is the shape the protocol takes: a host answering
    # about one turn is never asked to hand back the rest.
    history = CheckpointHistory(saver)
    retained = sorted((await history.snapshot(THREAD, 1)).retained)
    reachable = [
        (await history.snapshot(THREAD, n)).state[KEY]["value"] for n in retained
    ]
    assert reachable == [BOXES["Aarhus"], BOXES["Odense"]]
    # Copenhagen is gone, and nothing offered it.
    assert BOXES["Copenhagen"] not in reachable


async def test_a_turn_that_repeats_then_moves_is_still_announced() -> None:
    """The case a value comparison gets wrong, and the reason there is none.

    Turn 2 republishes turn 1's value and then changes it. Comparing the
    incoming value with the write it displaces sees Aarhus against Copenhagen
    *within* turn 2 and calls it a same-turn write, so it counts nothing — and
    the model is shown Aarhus with no note, never learning that turn 1 holds
    Copenhagen and is there to be read. That is the silent miss this whole
    feature exists to remove, so nothing here compares values: two turns wrote
    the key, and two is what it says.
    """
    agent = _agent(
        [
            _tool_call("get_aoi", "c1", {"place": "Copenhagen"}),
            AIMessage(content="Copenhagen."),
            _tool_call("get_aoi", "c2", {"place": "Copenhagen"}),
            _tool_call("get_aoi", "c3", {"place": "Aarhus"}),
            _tool_call("inspect_state", "c4", {"key": KEY}),
            AIMessage(content="Aarhus."),
        ]
    )
    await _ask(agent, "where is Copenhagen")
    second = await _ask(agent, "confirm, then try Aarhus")

    assert second["tool_state"][KEY]["turns_written"] == 2
    assert "2 turns of this conversation wrote this key" in _read(second)


async def test_the_breadcrumb_says_what_the_write_displaced() -> None:
    """The surface a model moving a value between tools actually reads.

    A handle resolves to the present and says nothing about earlier turns, so
    a model passing ``@state:<key>`` never triggers the read that would have
    warned it. The breadcrumb is the one surface every capture reaches, and it
    lands in the turn the overwrite happens.
    """
    agent = _agent(
        [
            _tool_call("get_aoi", "c1", {"place": "Copenhagen"}),
            AIMessage(content="one"),
            _tool_call("get_aoi", "c2", {"place": "Aarhus"}),
            AIMessage(content="two"),
        ]
    )
    await _ask(agent, "where is Copenhagen")
    second = await _ask(agent, "and Aarhus")

    notes = [
        str(message.content)
        for message in second["messages"]
        if isinstance(message, ToolMessage) and message.name == "get_aoi"
    ]
    # Turn 1 published for the first time, so there was nothing to displace.
    assert "replaces" not in notes[0]
    assert f"This replaces what {KEY} held at turn 1" in notes[-1]
    assert "inspect_state(key, turn=1)" in notes[-1]


async def test_a_write_inside_the_same_turn_displaces_nothing_reachable() -> None:
    """So the breadcrumb does not offer a turn that answers with this value.

    A turn ends holding one value per key however many times it wrote them.
    Naming the turn here would send the model to fetch what it already has.
    """
    agent = _agent(
        [
            _tool_call("get_aoi", "c1", {"place": "Copenhagen"}),
            _tool_call("get_aoi", "c2", {"place": "Aarhus"}),
            AIMessage(content="both"),
        ]
    )
    first = await _ask(agent, "Copenhagen and then Aarhus")

    notes = [
        str(message.content)
        for message in first["messages"]
        if isinstance(message, ToolMessage) and message.name == "get_aoi"
    ]
    assert not any("replaces" in note for note in notes)


async def test_a_key_written_in_one_turn_says_nothing_extra() -> None:
    """The note has to stay worth reading, so it is not on every value."""
    agent = _agent(
        [
            _tool_call("get_aoi", "c1", {"place": "Copenhagen"}),
            _tool_call("inspect_state", "c2", {"key": KEY}),
            AIMessage(content="ok"),
        ]
    )
    first = await _ask(agent, "where is Copenhagen")

    assert _read(first) == "[12.4, 55.5, 12.7, 55.8]"


async def test_a_turn_the_thread_never_had_says_how_many_it_has() -> None:
    """A miscount is the model's to fix, so it is told what to count to."""
    agent = _agent(
        [
            _tool_call("get_aoi", "c1", {"place": "Copenhagen"}),
            _tool_call("inspect_state", "c2", {"key": KEY, "turn": 4}),
            AIMessage(content="ok"),
        ]
    )
    first = await _ask(agent, "where is Copenhagen")

    read = _read(first)
    assert "no_such_turn" in read
    assert '"turns_so_far": 1' in read


async def test_a_deployment_with_no_checkpointer_says_so() -> None:
    """No conversation is being kept, so there is no past to read.

    Not an error and not an empty answer: a model told "nothing there" would
    conclude the earlier value never existed.
    """
    agent = with_session_state(
        StreamingScriptedModel(
            script=[
                _tool_call("get_aoi", "c1", {"place": "Copenhagen"}),
                _tool_call("inspect_state", "c2", {"key": KEY, "turn": 1}),
                AIMessage(content="ok"),
            ]
        ),
        [_gazetteer()],
    )
    result = await agent.ainvoke({"messages": [HumanMessage("where is Copenhagen")]})

    assert "no_turn_history" in _read(result)


async def test_turns_derive_the_same_whatever_order_the_saver_yields() -> None:
    """The saver contract promises an iterator, not an order.

    Every langgraph-shipped saver happens to yield newest first, and a
    derivation that leaned on that would keep each turn's *first* checkpoint
    when handed an ascending one — returning the state a turn started with as
    though it were what the turn produced. So the walk is fed forwards,
    backwards and shuffled, and must say the same thing each time.
    """
    import random

    from mcp_agent.history import _checkpoint, _derive

    saver = InMemorySaver()
    agent = _agent(
        [
            _tool_call("get_aoi", "c1", {"place": "Copenhagen"}),
            AIMessage(content="one"),
            _tool_call("get_aoi", "c2", {"place": "Aarhus"}),
            AIMessage(content="two"),
            _tool_call("get_aoi", "c3", {"place": "Odense"}),
            AIMessage(content="three"),
        ],
        saver,
    )
    await _ask(agent, "where is Copenhagen")
    await _ask(agent, "and Aarhus")
    await _ask(agent, "and Odense")

    config: Any = {"configurable": {"thread_id": THREAD}}
    entries = [_checkpoint(entry) async for entry in saver.alist(config)]
    current = max(entries, key=lambda checkpoint: len(checkpoint.messages)).messages

    async def feed(sequence: list[Any]) -> Any:
        for checkpoint in sequence:
            yield checkpoint

    baseline = await _derive(feed(entries), current)
    # The baseline itself is right, not merely stable: each turn ends holding
    # the box its own question produced, and the earliest is not the latest.
    assert [turn.n for turn in baseline.turns] == [1, 2, 3]
    assert baseline.find(1).state[KEY]["value"] == BOXES["Copenhagen"]
    assert baseline.find(3).state[KEY]["value"] == BOXES["Odense"]

    reordered = list(reversed(entries))
    assert await _derive(feed(reordered), current) == baseline
    shuffled = entries[:]
    random.Random(101).shuffle(shuffled)
    assert await _derive(feed(shuffled), current) == baseline


async def test_the_checkpointer_and_the_graph_derive_the_same_turns() -> None:
    """The equivalence the adapter rests on, checked rather than assumed.

    ``inspect_state`` is built before the graph it runs in, so its history can
    only come from the saver. That is only sound if the saver's own walk yields
    what the graph's does.
    """
    saver = InMemorySaver()
    agent = _agent(
        [
            _tool_call("get_aoi", "c1", {"place": "Copenhagen"}),
            AIMessage(content="one"),
            _tool_call("get_aoi", "c2", {"place": "Aarhus"}),
            AIMessage(content="two"),
        ],
        saver,
    )
    await _ask(agent, "where is Copenhagen")
    await _ask(agent, "and Aarhus")

    assert await turns_from(saver, THREAD) == await turns_of(agent, THREAD)
    first = await CheckpointHistory(saver).snapshot(THREAD, 1)
    assert first.state[KEY]["value"] == BOXES["Copenhagen"]
    assert first.retained == frozenset({1, 2}) and first.total == 2


async def _three_turns(saver: InMemorySaver) -> Any:
    """One thread of three turns, each publishing a different box."""
    agent = _agent(
        [
            _tool_call("get_aoi", "c1", {"place": "Copenhagen"}),
            AIMessage(content="one"),
            _tool_call("get_aoi", "c2", {"place": "Aarhus"}),
            AIMessage(content="two"),
            _tool_call("get_aoi", "c3", {"place": "Odense"}),
            AIMessage(content="three"),
        ],
        saver,
    )
    for question in ("where is Copenhagen", "and Aarhus", "and Odense"):
        await _ask(agent, question)
    return agent


async def test_a_page_at_a_time_derives_what_one_query_does() -> None:
    """The walk is bounded by paging, and bounded by nothing else.

    `alist` materialises every row it is asked for before yielding one, so the
    page size is the only thing standing between a turn-scoped read and the
    whole conversation being resident. It has to buy that without changing a
    single answer, at any size — one per page is the shape that would expose an
    off-by-one in the `before` cursor as a skipped or repeated checkpoint.
    """
    saver = InMemorySaver()
    await _three_turns(saver)

    whole = await turns_from(saver, THREAD, page=10_000)
    for size in (1, 2, 3, 7):
        assert await turns_from(saver, THREAD, page=size) == whole, size
    assert [turn.n for turn in whole.turns] == [1, 2, 3]
    assert whole.find(1).state[KEY]["value"] == BOXES["Copenhagen"]

    # And the cursor advances by a whole page. A `before` taken from the wrong
    # end of one still answers correctly — a checkpoint seen twice changes
    # nothing — so only the query count says whether paging is paging or
    # crawling a row at a time.
    counted = Counting(saver)
    checkpoints = len([entry async for entry in saver.alist(_config(THREAD))])
    for size in (2, 3):
        counted.queries = 0
        assert await turns_from(cast(Any, counted), THREAD, page=size) == whole
        assert counted.queries == checkpoints // size + 1, size


class Counting:
    """A saver that passes everything through and says how often it was asked."""

    def __init__(self, saver: InMemorySaver) -> None:
        self._saver = saver
        self.queries = 0

    async def alist(self, config: Any, **kwargs: Any) -> Any:
        self.queries += 1
        async for entry in self._saver.alist(config, **kwargs):
            yield entry


async def test_a_saver_that_will_not_page_is_read_whole_instead() -> None:
    """Paging needs an order the contract does not promise, so it checks.

    `before` is strictly-less-than on the checkpoint id, so a page that came
    back ascending would advance the cursor from the wrong end — repeating rows
    or skipping them, depending on where the page fell. The page is therefore
    rejected whole, before any of it is used, and the walk is redone unpaged:
    that costs the memory paging was there to save and returns the right
    answer, which is the correct way round.
    """

    class Ascending:
        """A saver that yields oldest first, which the contract permits."""

        def __init__(self, saver: InMemorySaver) -> None:
            self._saver = saver
            self.pages = 0

        async def alist(self, config: Any, **kwargs: Any) -> Any:
            self.pages += 1
            entries = [entry async for entry in self._saver.alist(config)]
            for entry in reversed(entries):
                yield entry

    saver = InMemorySaver()
    await _three_turns(saver)
    backwards = Ascending(saver)

    assert await turns_from(cast(Any, backwards), THREAD) == await turns_from(
        saver, THREAD
    )
    # One rejected page, then the whole walk. Not a loop, and not a truncation.
    assert backwards.pages == 2


async def test_only_the_turn_asked_for_carries_its_state() -> None:
    """The other turns are placed and counted, and hand back nothing.

    Session state holds the values too big for a transcript, so keeping every
    turn's to answer about one multiplies the largest thing here by the length
    of the conversation.
    """
    saver = InMemorySaver()
    await _three_turns(saver)

    narrowed = await turns_from(saver, THREAD, keep=2)
    assert [turn.n for turn in narrowed.turns] == [1, 2, 3]
    assert narrowed.total == 3
    assert narrowed.find(2).state[KEY]["value"] == BOXES["Aarhus"]
    assert narrowed.find(1).state == {} and narrowed.find(3).state == {}
    # Naming no turn keeps them all, which is what the thread listing wants.
    assert (await turns_from(saver, THREAD)).find(1).state[KEY]["value"] == BOXES[
        "Copenhagen"
    ]
