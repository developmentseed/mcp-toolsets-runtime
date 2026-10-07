"""Heal a thread whose last run died between a tool call and its result.

A run cancelled while a tool is running — the client closing the stream, a pod
rolled mid-turn — has already checkpointed the model's ``AIMessage`` asking for
the tool, but never the tool's result. Providers reject that history outright
(Mistral: "Not the same number of function calls and responses"), so every
later turn on the thread fails, and nothing the user sends can fix it.

:data:`repair_interrupted_tool_calls` closes each such call with an error
``ToolMessage`` at the start of the next turn and writes that back to the
checkpoint, so the thread heals once and the model can simply run the tool
again. A thread paused on an ``interrupt()`` is never touched: a message on
one is refused before the graph runs, and a resume re-enters the paused node
without passing through ``before_agent`` (see :mod:`mcp_agent.interrupts`).
"""

from collections.abc import Sequence
from typing import Any

from langchain.agents.middleware import AgentState, before_agent
from langchain_core.messages import AIMessage, BaseMessage, RemoveMessage, ToolMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime

INTERRUPTED_TOOL_CALL = (
    "This tool call was interrupted before it returned, so it has no result."
)


def close_dangling_tool_calls(
    messages: Sequence[BaseMessage],
) -> Sequence[BaseMessage]:
    """``messages`` with an error ``ToolMessage`` after each unanswered call.

    Each placeholder goes straight after the call it answers, since providers
    check the order as well as the count. Returns ``messages`` itself when
    nothing is missing.
    """
    answered = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
    if not any(
        call["id"] not in answered
        for m in messages
        if isinstance(m, AIMessage)
        for call in m.tool_calls
    ):
        return messages
    closed: list[BaseMessage] = []
    for message in messages:
        closed.append(message)
        if isinstance(message, AIMessage):
            closed += [
                ToolMessage(
                    INTERRUPTED_TOOL_CALL,
                    tool_call_id=call["id"],
                    name=call["name"],
                    status="error",
                )
                for call in message.tool_calls
                if call["id"] not in answered
            ]
    return closed


@before_agent
def repair_interrupted_tool_calls(
    state: AgentState, runtime: Runtime
) -> dict[str, Any] | None:
    """Close any tool call an earlier, cancelled run left unanswered."""
    messages = state["messages"]
    closed = close_dangling_tool_calls(messages)
    if closed is messages:
        return None
    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *closed]}
