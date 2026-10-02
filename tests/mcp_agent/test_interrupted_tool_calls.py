"""A run cancelled mid-tool must not break the thread.

Recreates what a deployment saw: the model asks for a tool, the run is
cancelled while the tool runs (the browser closed the stream), and the
checkpoint keeps a tool call with no result. A model that checks the history
the way Mistral does then rejects every later turn — unless the repair closes
the call, which ``with_session_state`` always wires in.
"""

import asyncio
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool
from langgraph.checkpoint.memory import InMemorySaver

from mcp_agent.interrupted_tool_calls import (
    INTERRUPTED_TOOL_CALL,
    close_dangling_tool_calls,
)
from mcp_agent.main import with_session_state

MISTRAL_3230 = "Not the same number of function calls and responses"


class StrictModel(BaseChatModel):
    """Asks for ``search`` on the first turn, answers after; rejects like Mistral."""

    @property
    def _llm_type(self) -> str:
        return "strict"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "StrictModel":
        return self

    def _generate(self, messages: Any, *args: Any, **kwargs: Any) -> ChatResult:
        calls = [
            c["id"] for m in messages if isinstance(m, AIMessage) for c in m.tool_calls
        ]
        results = [m.tool_call_id for m in messages if isinstance(m, ToolMessage)]
        if set(calls) != set(results):
            raise ValueError(MISTRAL_3230)
        if not calls:
            reply = AIMessage(
                "", tool_calls=[{"name": "search", "args": {}, "id": "SH2lrEzCG"}]
            )
        else:
            reply = AIMessage("answered")
        return ChatResult(generations=[ChatGeneration(message=reply)])


def _blocking_search(started: asyncio.Event) -> BaseTool:
    @tool
    async def search() -> str:
        """Search."""
        started.set()
        await asyncio.Event().wait()
        return "hits"

    return search


async def _cancel_mid_tool(agent: Any, started: asyncio.Event) -> RunnableConfig:
    """Run one turn and cancel it while its tool runs, as a disconnect does."""
    config: RunnableConfig = {"configurable": {"thread_id": "t"}}
    run = asyncio.create_task(
        agent.ainvoke({"messages": [HumanMessage("and EWDS?")]}, config)
    )
    await started.wait()
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run
    return config


async def test_a_cancelled_run_breaks_a_thread_with_no_repair():
    started = asyncio.Event()
    agent = create_agent(
        StrictModel(), [_blocking_search(started)], checkpointer=InMemorySaver()
    )
    config = await _cancel_mid_tool(agent, started)
    with pytest.raises(ValueError, match=MISTRAL_3230):
        await agent.ainvoke({"messages": [HumanMessage("try again")]}, config)


async def test_with_session_state_heals_the_thread_on_the_next_turn():
    started = asyncio.Event()
    agent = with_session_state(
        StrictModel(),
        [_blocking_search(started)],
        checkpointer=InMemorySaver(),
        interrupt_gate=False,
    )
    config = await _cancel_mid_tool(agent, started)
    result = await agent.ainvoke({"messages": [HumanMessage("try again")]}, config)
    messages = result["messages"]
    assert messages[-1].content == "answered"
    # The placeholder sits straight after the call it answers, before the retry.
    assert isinstance(messages[2], ToolMessage)
    assert messages[2].tool_call_id == "SH2lrEzCG"
    assert messages[2].content == INTERRUPTED_TOOL_CALL
    assert messages[3].content == "try again"
    # Written back, so the checkpoint itself is healed, not just this request.
    state = await agent.aget_state(config)
    assert state.values["messages"][2].content == INTERRUPTED_TOOL_CALL


def test_nothing_changes_when_every_call_is_answered():
    messages = [
        AIMessage("", tool_calls=[{"name": "search", "args": {}, "id": "a"}]),
        ToolMessage("hits", tool_call_id="a"),
    ]
    assert close_dangling_tool_calls(messages) is messages


def test_only_the_unanswered_call_of_a_parallel_pair_is_closed():
    messages = [
        AIMessage(
            "",
            tool_calls=[
                {"name": "search", "args": {}, "id": "a"},
                {"name": "search", "args": {}, "id": "b"},
            ],
        ),
        ToolMessage("hits", tool_call_id="a"),
    ]
    closed = close_dangling_tool_calls(messages)
    assert [m.tool_call_id for m in closed if isinstance(m, ToolMessage)] == [
        "b",
        "a",
    ]
