"""Progress from a running tool, as MCP progress notifications.

A tool that waits on something slow says what it is waiting on::

    from mcp_runtime.progress import report

    await report("Queued, 30 s so far")

Served over MCP, each call is a ``notifications/progress`` on the open tool
call, which a client shows while the call is still running. Called any other
way (a test, a direct ``ainvoke``) there is no request to report on, and it
does nothing.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

#: The open request's MCP context, and how many reports it has sent. The
#: protocol wants each notification's ``progress`` to increase, so the count
#: is the progress: a tool says what is happening, not how far along it is.
_open: ContextVar[tuple[Any, list[int]] | None] = ContextVar(
    "mcp_runtime_progress", default=None
)


@contextmanager
def reporting(ctx: Any) -> Iterator[None]:
    """Send this call's reports on ``ctx`` (the bridge)."""
    token = _open.set((ctx, [0]) if ctx is not None else None)
    try:
        yield
    finally:
        _open.reset(token)


async def report(message: str) -> None:
    """Tell the client what the running tool is doing. A no-op off MCP."""
    if (opened := _open.get()) is None:
        return
    ctx, count = opened
    count[0] += 1
    await ctx.report_progress(count[0], None, message)
