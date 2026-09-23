"""Read per-user credentials from the calling MCP request's HTTP headers.

Tools that act on a user's behalf (e.g. downloading from a source that needs
the user's API key) must not bake secrets into the deployment. Instead the
MCP client sends them as HTTP headers on every call — a client connection
takes a ``headers`` dict, ``mcp-cli`` takes ``--header`` — and the tool reads
them at call time with :func:`credential_from_header`.

The credential rides the transport, so it never appears in the model context,
tool schemas, chat history or traces. Works with the stateless streamable
HTTP runtime (every tool call is its own request, carrying its own headers)
and from both sync and async tools.

The headers reach the tool through a context variable that
:func:`credential_middleware` sets for the duration of each inbound request.
The v1 SDK published one of these itself, which this module read; v2 hands
its per-request context to handlers as an argument instead, and a tool body
several frames down has no way to reach an argument. So the contextvar is
ours now, filled from ``ctx.request`` at the top of the chain — and a tool
that reads a credential keeps the signature it always had.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from mcp.server.context import CallNext, HandlerResult, ServerRequestContext
from starlette.requests import Request

#: Headers of the MCP request being served, or ``None`` outside one.
_headers: ContextVar[dict[str, str] | None] = ContextVar(
    "mcp_runtime_credentials", default=None
)


class MissingCredentialError(Exception):
    """A required credential header was absent from the calling request."""

    def __init__(self, header: str) -> None:
        super().__init__(
            f"missing credential: send the {header!r} HTTP header with your "
            f"MCP requests (a client connection takes a 'headers' dict; "
            f"mcp-cli takes --header)"
        )
        self.header = header


async def credential_middleware(
    ctx: ServerRequestContext[Any, Any], call_next: CallNext
) -> HandlerResult:
    """Publish the inbound request's headers for the handlers beneath it.

    Registered on every server :func:`mcp_runtime.server.build_server` builds.
    Runs for each inbound message, so a stateless deployment — where each tool
    call is its own HTTP request — reads that call's own headers and never a
    neighbouring user's.
    """
    request = ctx.request
    headers = dict(request.headers) if isinstance(request, Request) else None
    token = _headers.set(headers)
    try:
        return await call_next(ctx)
    finally:
        _headers.reset(token)


def credential_from_header(header: str) -> str:
    """Return the named header from the MCP request that invoked this tool.

    Raises :class:`MissingCredentialError` if the header is absent or the
    tool was not invoked over HTTP (e.g. called directly in tests).
    """
    headers = _headers.get()
    value = headers.get(header.lower()) if headers else None
    if not value:
        raise MissingCredentialError(header)
    return value


@contextmanager
def header_context(headers: dict[str, str]) -> Iterator[None]:
    """Run as if inside an MCP request carrying ``headers`` (test support).

    Lets toolset tests exercise credential-reading tools without a server:

        with header_context({"x-demo-token": "secret"}):
            my_tool.invoke({...})
    """
    token = _headers.set({name.lower(): value for name, value in headers.items()})
    try:
        yield
    finally:
        _headers.reset(token)
