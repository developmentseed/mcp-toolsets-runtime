"""How much of a client's context a toolset spends, with a warning past a budget.

The runtime agent moves a result's data fields into session state, so its model
reads the ``message`` alone. A plain MCP client has no such step: its model
receives the whole result, once as the SDK's indented JSON text and again as
``structuredContent``. Clients cap what they take. Claude Code cuts a result off
at 25,000 tokens, Gemini CLI at 40,000 characters, and a client on the Python
SDK fails the call outright past 1 MiB. ChatGPT publishes no figure and cuts
silently. About 20,000 characters, text and structured together, fits them all.

Two budgets, both warnings in the log rather than errors, since a result's size
depends on its input and a toolset can't always prevent a large one:

- ``MCP_RESULT_BUDGET`` (default 20,000): characters of one tool result. A
  result over it is logged with the tool's name and size.
- ``MCP_DEFINITIONS_BUDGET`` (default 20,000): characters of the toolset's
  ``tools/list``, which a client sends with every request. Over it is logged
  once, at startup.

A toolset's tests check their worst case with :func:`assert_result_fits`, which
also warns when a ``message`` looks like a stub.
"""

import logging
import os
import warnings
from typing import Any

from mcp.server.mcpserver.tools import Tool as MCPTool
from mcp.types import Tool as WireTool
from pydantic_core import to_json

__all__ = [
    "DEFINITIONS_BUDGET",
    "RESULT_BUDGET",
    "StubMessageWarning",
    "assert_result_fits",
    "check_definitions",
    "check_result",
    "definitions_size",
    "result_size",
]

logger = logging.getLogger(__name__)

RESULT_BUDGET = int(os.environ.get("MCP_RESULT_BUDGET", 20_000))
DEFINITIONS_BUDGET = int(os.environ.get("MCP_DEFINITIONS_BUDGET", 20_000))

#: A ``message`` shorter than this beside data fields is probably a stub such as
#: "Found 6 records.", with the answer only in the data.
# ponytail: a length is a guess at "says nothing"; a short message can answer
# fully. It only warns, in tests, so a false alarm costs a glance.
STUB_LENGTH = 80


class StubMessageWarning(UserWarning):
    """A result's ``message`` looks too short to answer on its own."""


def result_size(result: Any) -> int:
    """Characters a plain client receives for ``result``.

    The text content as the SDK writes it (JSON, indented by two) plus
    ``structuredContent``.
    """
    text = to_json(result, fallback=str, indent=2).decode()
    structured = to_json(result, fallback=str).decode()
    return len(text) + len(structured)


def check_result(tool: str, result: Any) -> None:
    """Log a warning when ``tool``'s ``result`` is over ``MCP_RESULT_BUDGET``."""
    size = result_size(result)
    if size > RESULT_BUDGET:
        logger.warning(
            "tool %r returned %d characters, over MCP_RESULT_BUDGET (%d); "
            "a plain MCP client may cut it off or fail the call",
            tool,
            size,
            RESULT_BUDGET,
        )


def definitions_size(tools: list[MCPTool]) -> int:
    """Characters of ``tools/list`` for ``tools``, built as the SDK builds it."""
    return sum(
        len(
            WireTool(
                name=tool.name,
                title=tool.title,
                description=tool.description,
                input_schema=tool.parameters,
                output_schema=tool.output_schema,
                annotations=tool.annotations,
                _meta=tool.meta,
            ).model_dump_json(by_alias=True, exclude_none=True)
        )
        for tool in tools
    )


def check_definitions(toolset: str, tools: list[MCPTool]) -> None:
    """Log a warning when the toolset's ``tools/list`` is over its budget."""
    size = definitions_size(tools)
    if size > DEFINITIONS_BUDGET:
        logger.warning(
            "toolset %r lists %d characters of tool definitions, over "
            "MCP_DEFINITIONS_BUDGET (%d); a client sends them with every request",
            toolset,
            size,
            DEFINITIONS_BUDGET,
        )


def assert_result_fits(result: dict[str, Any], budget: int | None = None) -> None:
    """For a toolset's tests: fail when ``result`` is over budget.

    ``budget`` defaults to ``MCP_RESULT_BUDGET``. Also warns
    (:class:`StubMessageWarning`) when a successful result carries data fields
    beside a ``message`` shorter than :data:`STUB_LENGTH`: over plain MCP the
    ``message`` should answer on its own.
    """
    limit = RESULT_BUDGET if budget is None else budget
    size = result_size(result)
    if size > limit:
        # Raised rather than asserted, so it holds under python -O as well.
        raise AssertionError(f"result is {size} characters, over the budget of {limit}")
    message = result.get("message")
    data = sorted(set(result) - {"message"})
    if isinstance(message, str) and data and len(message) < STUB_LENGTH:
        warnings.warn(
            f"message {message!r} may not answer without its data fields "
            f"({', '.join(data)})",
            StubMessageWarning,
            stacklevel=2,
        )
