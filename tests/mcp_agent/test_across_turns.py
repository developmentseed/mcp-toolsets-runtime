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

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver

from mcp_agent.history import CheckpointHistory, turns_from, turns_of
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


async def test_a_plain_read_says_the_key_has_been_rewritten() -> None:
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
    assert "held 2 different values" in read
    assert "turn=" in read


async def test_the_key_listing_marks_a_rewritten_key() -> None:
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

    assert "rewritten, 2 versions" in _read(second)


async def test_a_key_written_once_says_nothing_about_versions() -> None:
    """The word has to stay worth reading, so it is not on every value."""
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
    assert (await CheckpointHistory(saver).snapshots(THREAD)).turns[1][KEY][
        "value"
    ] == BOXES["Copenhagen"]
