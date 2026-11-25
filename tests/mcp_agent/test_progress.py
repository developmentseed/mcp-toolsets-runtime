"""A toolset tool that reports progress, through the agent to the AG-UI stream.

Over a real MCP server on streamable HTTP, like the elicitation test: the
notification is sent by the SDK, received by fastmcp, and routed back to its
tool call by a context variable, and only a real round trip shows that the
handler runs where that variable is set.
"""

import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest
import uvicorn
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from mcp.server.mcpserver import MCPServer

from mcp_agent.main import list_tools, with_credential_support, with_session_state
from mcp_agent.streaming import ToolFinished, ToolProgress, ToolStarted, stream_turn
from mcp_agent_api.events import TOOL_PROGRESS, agui_events
from mcp_runtime.bridge import to_mcp_tool
from mcp_runtime.progress import report
from mcp_runtime.tool_result import ToolResult
from tests.mcp_agent.test_streaming import StreamingScriptedModel


@tool
async def wait_for_it() -> ToolResult:
    """Wait, saying so."""
    await report("Queued")
    await report("Running, 30 s")
    return ToolResult(message="ready")


@pytest.fixture(scope="module")
def url() -> Iterator[str]:
    app = MCPServer("waiter", tools=[to_mcp_tool(wait_for_it)]).streamable_http_app(
        stateless_http=True
    )
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


async def _turn(url: str) -> list[Any]:
    connections = {"waiter": {"url": url, "transport": "streamable_http"}}
    (client,) = with_credential_support(connections, None).values()
    model = StreamingScriptedModel(
        script=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "wait_for_it", "args": {}, "id": "c1", "type": "tool_call"}
                ],
            ),
            AIMessage(content="done"),
        ]
    )
    agent = with_session_state(model, await list_tools(client), InMemorySaver())
    return [event async for event in stream_turn(agent, "go", "t1")]


async def test_progress_arrives_between_the_call_and_its_result(url):
    events = await _turn(url)

    kinds = [
        type(e)
        for e in events
        if isinstance(e, ToolStarted | ToolProgress | ToolFinished)
    ]
    assert kinds == [ToolStarted, ToolProgress, ToolProgress, ToolFinished]
    assert [e for e in events if isinstance(e, ToolProgress)] == [
        ToolProgress("c1", "Queued"),
        ToolProgress("c1", "Running, 30 s"),
    ]


async def test_each_report_replaces_the_calls_one_progress_activity(url):
    async def turn() -> Any:
        for event in await _turn(url):
            yield event

    activities = [
        e
        async for e in agui_events(turn(), thread_id="t1", run_id="r1")
        if getattr(e, "activity_type", None) == TOOL_PROGRESS
    ]

    assert {a.message_id for a in activities} == {"progress_r1_c1"}
    assert [a.content["display"] for a in activities] == ["Queued", "Running, 30 s"]


async def test_report_does_nothing_off_mcp():
    assert (await wait_for_it.ainvoke({}))["message"] == "ready"
