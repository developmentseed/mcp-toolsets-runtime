"""A tool's MCP progress notifications, onto the run's stream.

A server tool reports with :func:`mcp_runtime.progress.report`; the MCP client
hands each notification to :func:`forward_progress`, which writes it to
LangGraph's ``custom`` stream as :data:`PROGRESS`, naming the tool call it
belongs to. :func:`~mcp_agent.streaming.stream_turn` turns that into a
:class:`~mcp_agent.streaming.ToolProgress`.

The notification carries no tool call id, so :data:`track_tool_calls` keeps the
running call's id in a context variable. That works because the client runs
the handler in the calling task's context, where the variable is set.
"""

from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from typing import Any

from langchain.agents.middleware import wrap_tool_call
from langchain.agents.middleware.types import ToolCallRequest
from langgraph.config import get_stream_writer

#: The key a progress write carries on the ``custom`` stream.
PROGRESS = "mcp_agent.progress"

_calling: ContextVar[str | None] = ContextVar("mcp_agent_tool_call", default=None)


@wrap_tool_call
async def track_tool_calls(
    request: ToolCallRequest, handler: Callable[[ToolCallRequest], Awaitable[Any]]
) -> Any:
    """Run each tool call with its id where :func:`forward_progress` finds it."""
    token = _calling.set(request.tool_call.get("id"))
    try:
        return await handler(request)
    finally:
        _calling.reset(token)


async def forward_progress(
    progress: float, total: float | None, message: str | None
) -> None:
    """The MCP client's progress handler: one write on the ``custom`` stream."""
    if (call := _calling.get()) is None or not message:
        return
    try:
        write = get_stream_writer()
    except RuntimeError:
        return  # called outside a graph run: nobody is listening
    write({PROGRESS: {"tool_call_id": call, "message": message}})


def with_progress(session: Any) -> None:
    """Have every tool call on ``session`` report through :func:`forward_progress`.

    ``langchain.mcp`` calls a tool on the session itself, not through the
    fastmcp client, so the client's ``progress_handler`` never sees the call.
    The session is the per-call one ``_on_its_own_session`` opens, so the patch
    ends with it.
    """
    call = session.call_tool

    async def calling(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("progress_callback", forward_progress)
        return await call(*args, **kwargs)

    session.call_tool = calling
