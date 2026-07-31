"""The bundled agent's session-state wiring, and the switch that turns it off.

``mcp_state`` itself is covered in ``tests/mcp_state``; what is tested here is
that the agent actually installs it — the failure mode being silent, since an
agent with no capture works perfectly well and merely costs a fortune.
"""

from typing import Any

from langchain_core.tools import StructuredTool

from mcp_agent.main import StateSettings, run_turn, with_session_state
from mcp_runtime.declarations import CONSUMES_META_KEY, PRODUCES_META_KEY
from mcp_runtime.kinds import GEOJSON_AREA_OF_INTEREST
from mcp_state.state import TOOL_STATE_KEY, StateEntry

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
    """Stands in for the compiled graph: records the state it was invoked with."""

    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        self.seen: list[dict[str, Any]] = []

    async def ainvoke(self, state: dict[str, Any]) -> dict[str, Any]:
        self.seen.append(state)
        return self.result


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


def test_all_four_pieces_are_installed(monkeypatch):
    # Any one of them missing leaves the others doing nothing, so the test is
    # on the set rather than on any single argument.
    recorded = _record_create_agent(monkeypatch)
    with_session_state("model", [mcp_tool("search", PUBLISHES_AOI)])
    assert "inspect_state" in recorded["tools"]
    assert TOOL_STATE_KEY in recorded["state_schema"].__annotations__
    assert [type(m).__name__ for m in recorded["middleware"]] == [
        "StateCaptureMiddleware"
    ]


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


async def test_state_survives_between_turns():
    """The bug this guards: the agent is invoked per turn, so state that isn't
    handed back in is gone, and injection finds nothing with no error."""
    entry = StateEntry(value={"type": "FeatureCollection"}, kind=None, tool="search")
    agent = _RecordingAgent({"messages": ["reply"], TOOL_STATE_KEY: {"a/b": entry}})

    _, _, state = await run_turn(agent, [], "first")
    assert state == {"a/b": entry}

    await run_turn(agent, [], "second", state)
    assert agent.seen[-1][TOOL_STATE_KEY] == {"a/b": entry}


async def test_no_state_key_is_sent_to_an_agent_that_has_none():
    """With state off the graph has no such channel; sending one risks an error
    on a schema that never declared it."""
    agent = _RecordingAgent({"messages": ["reply"]})
    _, _, state = await run_turn(agent, [], "hello")
    assert state is None
    assert TOOL_STATE_KEY not in agent.seen[-1]
