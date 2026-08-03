"""The bundled agent's session-state wiring, and the switch that turns it off.

``mcp_state`` itself is covered in ``tests/mcp_state``; what is tested here is
that the agent actually installs it — the failure mode being silent, since an
agent with no capture works perfectly well and merely costs a fortune.
"""

from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver

from mcp_agent.main import (
    Checkpointing,
    StateSettings,
    receipt_lines,
    run_turn,
    with_session_state,
)
from mcp_runtime.declarations import CONSUMES_META_KEY, PRODUCES_META_KEY
from mcp_runtime.kinds import GEOJSON_AREA_OF_INTEREST
from mcp_state.state import TOOL_STATE_KEY

AOI = {"type": "FeatureCollection", "features": [{"id": "polygon"}]}

PUBLISHES_AOI = {
    PRODUCES_META_KEY: [
        {
            "stateKey": "dataset-search/geometry",
            "field": "geometry",
            "kind": GEOJSON_AREA_OF_INTEREST,
        }
    ]
}

NEEDS_AOI = {
    CONSUMES_META_KEY: [
        {
            "parameter": "aoi",
            "kind": GEOJSON_AREA_OF_INTEREST,
            "required": True,
            "modelGeneratable": False,
        }
    ]
}


def mcp_tool(name: str, meta: dict[str, Any] | None = None) -> StructuredTool:
    """A stand-in for a tool the adapter loaded from an MCP server."""

    async def call(**arguments: Any) -> Any:
        return "ok", None

    return StructuredTool(
        name=name,
        description=name,
        args_schema={"type": "object", "properties": {}},
        coroutine=call,
        metadata={"_meta": meta} if meta else None,
    )


class _RecordingAgent:
    """Stands in for the compiled graph: records what it was invoked with."""

    def __init__(self, result: dict[str, Any], already: int = 0) -> None:
        self.result = result
        self.already = already  # messages the thread already held
        self.seen: list[dict[str, Any]] = []
        self.configs: list[dict[str, Any]] = []

    async def aget_state(self, config: dict[str, Any]) -> Any:
        return SimpleNamespace(values={"messages": ["old"] * self.already})

    async def ainvoke(
        self, state: dict[str, Any], config: dict[str, Any]
    ) -> dict[str, Any]:
        self.seen.append(state)
        self.configs.append(config)
        return self.result


class _ScriptedModel(GenericFakeChatModel):
    """A model that replays a fixed script, ignoring what it is bound to."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self


def _publishing_tool() -> StructuredTool:
    """A local tool shaped like an MCP one: content plus a structured artifact."""

    async def call() -> tuple[str, dict[str, Any]]:
        return "ok", {"structured_content": {"message": "found", "geometry": AOI}}

    return StructuredTool(
        name="search",
        description="search",
        args_schema={"type": "object", "properties": {}},
        coroutine=call,
        response_format="content_and_artifact",
        metadata={"_meta": PUBLISHES_AOI},
    )


def _agent_over(script: list[Any]) -> Any:
    """A real agent over a real in-process checkpointer, driven by ``script``."""
    agent, _ = with_session_state(
        _ScriptedModel(messages=iter(script)), [_publishing_tool()], InMemorySaver()
    )
    return agent


def _tool_call(index: int) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {"name": "search", "args": {}, "id": f"c{index}", "type": "tool_call"}
        ],
    )


def _record_create_agent(monkeypatch) -> dict[str, Any]:
    """Capture the arguments the agent is built with, without building one."""
    recorded: dict[str, Any] = {}

    def fake_create_agent(model, tools, **kwargs):
        recorded["tools"] = [tool.name for tool in tools]
        recorded.update(kwargs)
        return "agent"

    monkeypatch.setattr("mcp_agent.main.create_agent", fake_create_agent)
    return recorded


def test_state_is_on_unless_switched_off(monkeypatch):
    monkeypatch.delenv("MCP_AGENT_STATE", raising=False)
    assert StateSettings(_env_file=None).mcp_agent_state is True
    monkeypatch.setenv("MCP_AGENT_STATE", "0")
    assert StateSettings(_env_file=None).mcp_agent_state is False


def test_state_switch_reads_dotenv(monkeypatch, tmp_path):
    # The deployment sets it the same way it sets every other agent option.
    monkeypatch.delenv("MCP_AGENT_STATE", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("MCP_AGENT_STATE=false\n")
    assert StateSettings(_env_file=env_file).mcp_agent_state is False


def test_all_three_pieces_are_installed(monkeypatch):
    # Any one of them missing leaves the others doing nothing, so the test is
    # on the set rather than on any single argument. The `tool_state` channel
    # comes with the middleware, so it is not passed separately.
    recorded = _record_create_agent(monkeypatch)
    with_session_state("model", [mcp_tool("search", PUBLISHES_AOI)])
    assert "inspect_state" in recorded["tools"]
    assert "state_schema" not in recorded
    (middleware,) = recorded["middleware"]
    assert type(middleware).__name__ == "StateCaptureMiddleware"
    assert TOOL_STATE_KEY in middleware.state_schema.__annotations__


def test_an_uncallable_tool_is_withheld_and_reported(monkeypatch):
    # clip needs an AOI nothing publishes and may not be invented: every call
    # would raise before reaching the server, so it is not offered at all.
    recorded = _record_create_agent(monkeypatch)
    _, withheld = with_session_state("model", [mcp_tool("clip", NEEDS_AOI)])
    assert "clip" not in recorded["tools"]
    assert [(item.tool, item.parameter) for item in withheld] == [("clip", "aoi")]


def test_a_tool_stays_when_its_producer_is_connected(monkeypatch):
    recorded = _record_create_agent(monkeypatch)
    _, withheld = with_session_state(
        "model", [mcp_tool("clip", NEEDS_AOI), mcp_tool("search", PUBLISHES_AOI)]
    )
    assert withheld == []
    assert {"clip", "search"} <= set(recorded["tools"])


def test_a_host_can_layer_its_own_prompt_tools_and_middleware(monkeypatch):
    """The seams a host with its own agent would otherwise fork main.py for."""
    recorded = _record_create_agent(monkeypatch)
    local = mcp_tool("load_skill")
    with_session_state(
        "model",
        [mcp_tool("search", PUBLISHES_AOI)],
        system_prompt="be terse",
        extra_tools=[local],
        middleware=["tracing"],
    )
    assert recorded["system_prompt"] == "be terse"
    assert "load_skill" in recorded["tools"]
    # Capture stays first: a host's middleware layers over it, not instead.
    assert type(recorded["middleware"][0]).__name__ == "StateCaptureMiddleware"
    assert recorded["middleware"][1] == "tracing"


def test_an_extra_tool_is_not_treated_as_an_mcp_tool(monkeypatch):
    """A host's own tool is added as given: not bound to state, not withheld."""
    recorded = _record_create_agent(monkeypatch)
    _, withheld = with_session_state(
        "model", [], extra_tools=[mcp_tool("clip", NEEDS_AOI)]
    )
    assert withheld == []
    assert "clip" in recorded["tools"]


def _thread_of(*contents: str) -> dict[str, Any]:
    return {"messages": [AIMessage(content=text) for text in contents]}


async def test_run_turn_sends_only_the_new_message():
    """The thread holds the transcript, so resending it would duplicate it."""
    agent = _RecordingAgent(_thread_of("old", "you", "reply"), already=1)
    await run_turn(agent, "hello", "thread-1")
    assert [m.content for m in agent.seen[-1]["messages"]] == ["hello"]
    assert agent.configs[-1]["configurable"]["thread_id"] == "thread-1"


async def test_run_turn_returns_only_this_turns_replies():
    """New messages exclude the human turn that triggered them."""
    agent = _RecordingAgent(_thread_of("old", "you", "reply"), already=1)
    turn = await run_turn(agent, "hello", "thread-1")
    assert [m.content for m in turn.history] == ["old", "you", "reply"]
    assert [m.content for m in turn.new_messages] == ["reply"]
    assert turn.answer == "reply"


async def test_run_turn_merges_caller_config():
    """A host attaches per-turn callbacks; thread_id still wins in configurable."""
    agent = _RecordingAgent(_thread_of("old", "you", "reply"), already=1)
    await run_turn(
        agent,
        "hello",
        "thread-1",
        {"callbacks": ["handler"], "configurable": {"thread_id": "ignored"}},
    )
    assert agent.configs[-1]["callbacks"] == ["handler"]
    assert agent.configs[-1]["configurable"]["thread_id"] == "thread-1"


async def test_a_turn_that_produced_no_reply_has_an_empty_answer():
    agent = _RecordingAgent(_thread_of("old", "you"), already=1)
    turn = await run_turn(agent, "hello", "thread-1")
    assert turn.new_messages == []
    assert turn.answer == ""
    assert turn.citations == []


async def test_state_and_transcript_both_survive_between_turns():
    """The point of checkpointing: turn 2 sees turn 1 without being told.

    Over a real agent and a real checkpointer. The previous arrangement made
    the caller hand state back between turns, and getting that wrong was
    silent — nothing captured, nothing injected, no error.
    """
    agent = _agent_over(
        [_tool_call(1), AIMessage(content="found"), AIMessage(content="still here")]
    )

    first = await run_turn(agent, "find it", "thread-1")
    assert list(first.sidecar or {}) == ["dataset-search/geometry"]

    turn = await run_turn(agent, "and again", "thread-1")
    assert list(turn.sidecar or {}) == ["dataset-search/geometry"], "state must persist"
    assert len(turn.history) > len(turn.new_messages), "the transcript must persist too"
    assert "find it" in [str(message.content) for message in turn.history]


async def test_threads_do_not_share_state():
    """One process serves many conversations; they must not see each other's."""
    agent = _agent_over(
        [_tool_call(1), AIMessage(content="found"), AIMessage(content="nothing yet")]
    )

    first = await run_turn(agent, "find it", "thread-1")
    assert list(first.sidecar or {}) == ["dataset-search/geometry"]

    second = await run_turn(agent, "what do you have?", "thread-2")
    assert not (second.sidecar or {}), "a fresh thread starts with empty state"
    assert "find it" not in [str(message.content) for message in second.history]


async def test_checkpointing_defaults_to_memory(monkeypatch):
    monkeypatch.delenv("MCP_AGENT_CHECKPOINT", raising=False)
    async with Checkpointing() as checkpointing:
        assert isinstance(await checkpointing.saver(), InMemorySaver)


async def test_checkpointing_builds_one_saver_and_reuses_it():
    # A Postgres saver owns a connection pool, and the web host asks once per
    # session and again on every model change — that must not mean two pools.
    async with Checkpointing("memory") as checkpointing:
        assert await checkpointing.saver() is await checkpointing.saver()


async def test_separate_checkpointing_objects_do_not_share():
    """No hidden process-wide instance: what you hold is what you write to."""
    async with Checkpointing("memory") as one, Checkpointing("memory") as two:
        assert await one.saver() is not await two.saver()


async def test_an_unrecognised_checkpoint_target_is_refused():
    """Rather than being handed to a driver as a DSN and failing obscurely."""
    async with Checkpointing("mysql://db/agent") as checkpointing:
        with pytest.raises(ValueError, match="is not a checkpointer"):
            await checkpointing.saver()


async def test_constructing_checkpointing_reads_nothing(monkeypatch):
    """So a misconfigured environment fails where it is opened, not on import."""
    monkeypatch.setenv("MCP_AGENT_CHECKPOINT", "nonsense")
    checkpointing = Checkpointing()  # must not raise
    with pytest.raises(ValueError, match="is not a checkpointer"):
        await checkpointing.open()


def test_a_bad_target_is_caught_without_opening_anything(monkeypatch):
    """Chainlit swallows startup-hook errors, so the entry points check first.

    Validation is therefore I/O-free and synchronous: it has to run before the
    app starts, which is before there is an event loop to open a pool in.
    """
    monkeypatch.setenv("MCP_AGENT_CHECKPOINT", "nonsense")
    with pytest.raises(ValueError, match="is not a checkpointer"):
        Checkpointing().validate()


def test_validate_accepts_what_can_actually_be_built(monkeypatch):
    monkeypatch.delenv("MCP_AGENT_CHECKPOINT", raising=False)
    assert Checkpointing().validate() == "memory"
    assert Checkpointing(" postgres://db/agent ").validate() == "postgres://db/agent"


async def test_closing_releases_the_saver():
    checkpointing = Checkpointing("memory")
    first = await checkpointing.saver()
    await checkpointing.aclose()
    assert await checkpointing.saver() is not first


def _received(receipts: dict) -> ToolMessage:
    return ToolMessage(
        content="ok",
        name="clip_raster",
        tool_call_id="2",
        artifact={"structured_content": {}, "injected_state": receipts},
    )


FILLED = {
    "aoi": {
        "key": "dataset-search/geometry",
        "via": "declaration",
        "kind": "geojson.AreaOfInterest",
        "tool": "search_datasets",
    }
}


def test_the_cli_names_the_parameter_the_printed_call_omits():
    """`→ clip_raster {'dataset_id': 'chirps'}` shows no aoi; this is why."""
    assert receipt_lines({"dataset_id": "chirps"}, _received(FILLED)) == [
        "aoi ← dataset-search/geometry, published by search_datasets"
    ]


def test_the_cli_does_not_repeat_a_handle_already_in_the_call():
    args = {"geometry": "@state:dataset-search/geometry"}
    handle = {"geometry": {**FILLED["aoi"], "via": "handle"}}
    assert receipt_lines(args, _received(handle)) == []


def test_the_cli_prints_nothing_for_a_tool_that_took_nothing_from_state():
    assert receipt_lines({"city": "Reading"}, _received({})) == []
    assert receipt_lines({"city": "Reading"}, None) == []
