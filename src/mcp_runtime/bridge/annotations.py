"""Where a LangChain tool's return annotation is read.

Read by :func:`~mcp_runtime.bridge.to_mcp_tool` for the output schema, and
by :mod:`mcp_runtime.declarations` for what a tool publishes.
"""

from typing import Any, get_type_hints

from langchain_core.tools import BaseTool


def return_annotation(tool: BaseTool) -> Any:
    """The return annotation of the function behind a LangChain tool, if any."""
    fn = getattr(tool, "coroutine", None) or getattr(tool, "func", None)
    if fn is None:
        return None
    return get_type_hints(fn).get("return")
