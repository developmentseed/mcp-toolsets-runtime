"""A toolset tool that asks the user a question, through the agent and back.

The round trip runs against a real MCP server over streamable HTTP, because
every step of it belongs to someone else: the bridge returns the question the
SDK's way, fastmcp negotiates the protocol version that allows it, and
``langchain.mcp`` pauses the run. A stub would agree with whatever this code
assumed about all three.
"""

import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest
import uvicorn
from ag_ui.core import ResumeEntry
from fastapi import HTTPException
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from mcp.server.mcpserver import MCPServer

from mcp_agent.interrupts import INPUT_REQUIRED, PendingInterrupt, answer, question
from mcp_agent.main import list_tools, with_credential_support, with_session_state
from mcp_agent.streaming import ToolFinished, TurnFinished, stream_turn
from mcp_agent_api.events import agui_interrupt
from mcp_agent_api.routes import resume_responses
from mcp_runtime.bridge import to_mcp_tool
from mcp_runtime.questions import AskUser, answers
from mcp_runtime.tool_result import ToolResult
from tests.mcp_agent.test_streaming import StreamingScriptedModel

YEAR_SCHEMA = {
    "type": "object",
    "properties": {"year": {"type": "integer"}},
    "required": ["year"],
}


@tool
async def pick_year() -> ToolResult:
    """Pick a year, asking the user which."""
    if (replies := answers()) is None:
        raise AskUser(year=("Which year?", YEAR_SCHEMA))
    given = replies["year"]
    if given.action != "accept" or given.content is None:
        return ToolResult(message=f"no year: {given.action}")
    return ToolResult(message=f"picked {given.content['year']}")


def _asking_server() -> MCPServer:
    return MCPServer("asker", tools=[to_mcp_tool(pick_year)])


@pytest.fixture(scope="module")
def asking_url() -> Iterator[str]:
    app = _asking_server().streamable_http_app(stateless_http=True)
    # Port 0, read back once bound: probing for a free port and releasing it
    # races anything else on the machine for it.
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}/mcp"
    server.should_exit = True
    thread.join()


async def _agent(url: str) -> Any:
    connections = {"asker": {"url": url, "transport": "streamable_http"}}
    (client,) = with_credential_support(connections, None).values()
    model = StreamingScriptedModel(
        script=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "pick_year", "args": {}, "id": "c1", "type": "tool_call"}
                ],
            ),
            AIMessage(content="done"),
        ]
    )
    return with_session_state(model, await list_tools(client), InMemorySaver())


async def _turn(agent: Any, text: str | None, **kwargs: Any) -> list:
    return [event async for event in stream_turn(agent, text, "t1", **kwargs)]


def _result(events: list) -> Any:
    assert isinstance(events[-1], TurnFinished)
    return events[-1].result


async def test_the_turn_stops_on_the_tools_question(asking_url):
    result = _result(await _turn(await _agent(asking_url), "pick"))

    [asked] = result.interrupts
    assert asked.value == {
        "reason": INPUT_REQUIRED,
        "message": "Which year?",
        "metadata": {"tool": "pick_year"},
        "responseSchema": YEAR_SCHEMA,
    }


async def test_the_answer_reaches_the_tool_that_asked(asking_url):
    agent = await _agent(asking_url)
    [asked] = _result(await _turn(agent, "pick")).interrupts

    events = await _turn(
        agent,
        None,
        resume={asked.id: {"status": "resolved", "payload": {"year": 2024}}},
    )

    [finished] = [event for event in events if isinstance(event, ToolFinished)]
    assert "picked 2024" in str(finished.content)
    assert _result(events).answer == "done"


async def test_a_skipped_question_declines_the_tools_request(asking_url):
    agent = await _agent(asking_url)
    [asked] = _result(await _turn(agent, "pick")).interrupts

    events = await _turn(agent, None, resume={asked.id: {"status": "cancelled"}})

    [finished] = [event for event in events if isinstance(event, ToolFinished)]
    assert "no year: decline" in str(finished.content)


# --- the translation, both ways ---------------------------------------------


def _elicitation(*requests: dict[str, Any]) -> dict[str, Any]:
    return {"type": "mcp_elicitation", "tool_name": "t", "requests": list(requests)}


FORM = {
    "key": "year",
    "message": "Which year?",
    "mode": "form",
    "requested_schema": YEAR_SCHEMA,
}
URL = {
    "key": "login",
    "message": "Sign in at",
    "mode": "url",
    "url": "https://example.org/in",
}


def test_several_requests_are_one_question_keyed_by_request():
    asked = question(_elicitation(FORM, URL))

    assert asked["message"] == "Which year?\nSign in at https://example.org/in"
    assert asked["responseSchema"] == {
        "type": "object",
        "properties": {"year": YEAR_SCHEMA},
        "required": ["year"],
    }
    assert answer(
        _elicitation(FORM, URL),
        {"status": "resolved", "payload": {"year": {"year": 2024}}},
    ) == {
        "responses": {
            "year": {"action": "accept", "content": {"year": 2024}},
            "login": {"action": "accept"},
        }
    }


def test_url_requests_alone_have_no_schema_to_fill():
    for asked in (question(_elicitation(URL)), question(_elicitation(URL, URL))):
        assert "responseSchema" not in asked


def test_a_request_the_answer_leaves_out_is_declined():
    """Accepting it with no content is what the server would reject."""
    resumed = answer(
        _elicitation(FORM, {**FORM, "key": "month"}),
        {"status": "resolved", "payload": {"year": {"year": 2024}}},
    )
    assert resumed["responses"]["month"] == {"action": "decline"}


def test_the_api_refuses_an_answer_that_does_not_fit():
    asked = PendingInterrupt(id="i1", value=question(_elicitation(FORM)))
    wrong = ResumeEntry(interrupt_id="i1", status="resolved", payload={"year": "soon"})
    with pytest.raises(HTTPException) as refused:
        resume_responses([wrong], [asked])
    assert refused.value.status_code == 422


def test_anything_else_passes_through_both_ways():
    gate = {"reason": INPUT_REQUIRED, "message": "Which?", "responseSchema": {}}
    response = {"status": "resolved", "payload": {"choice": "a"}}
    assert question(gate) is gate
    assert answer(gate, response) is response


def test_the_question_goes_out_as_an_agui_interrupt():
    sent = agui_interrupt(PendingInterrupt(id="i1", value=question(_elicitation(FORM))))

    assert sent.message == "Which year?"
    assert sent.response_schema == YEAR_SCHEMA
    assert sent.metadata == {"tool": "t"}
