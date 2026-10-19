"""A LangChain tool as an MCP tool, with an output schema from its return.

The SDK builds a tool from a *function*, reading name, description and schema
off its signature. A LangChain tool carries those as data instead, so this
package assembles :class:`mcp.server.mcpserver.tools.Tool` directly: arguments
from ``tool_call_schema``, the body from ``ainvoke``.

The output model comes from the return annotation — a TypedDict/BaseModel, or
a union of them, with a required str ``message`` on at least one arm (the
:mod:`mcp_runtime.tool_result` contract) — so results travel as
``structuredContent`` and not JSON text alone. Any other annotation raises at
conversion, aborting ``build_server``: the SDK would wrap such values in
``{"result": ...}`` or the schema would guarantee nothing.

The server validates every returned dict against the model; undeclared keys
are dropped, so the annotation is the complete list of keys a client can see.

Arguments are made to match. The argument model forbids extras, so an
undeclared *input* is an error naming the parameter rather than an argument
silently dropped, and ``additionalProperties: false`` publishes that rule.

A tool asks the user with :func:`~mcp_runtime.questions.ask`; the call
returns the SDK's ``InputRequiredResult`` for it, and the retry's answer is
published to the tool through :mod:`mcp_runtime.questions`.

This was ``langchain-mcp-adapters``' ``to_fastmcp`` wrapped in our own
output-schema derivation. That package is retired and its successor
(``langchain.mcp``) converts in the other direction only, so the conversion
is ours now. It is the only code that touches the SDK's internals. The half
we already owned, the output schema and its gate, is
:func:`mcp_runtime.tool_result.output_schema`.
"""

from typing import Any

from langchain_core.tools import BaseTool
from mcp.server.mcpserver import Context
from mcp.server.mcpserver.tools import Tool as MCPTool

# Internal to the SDK, and imported only in this package: a minor release can
# change it, and test_the_sdk_internals_we_build_on_keep_their_shape says so.
from mcp.server.mcpserver.utilities.func_metadata import FuncMetadata
from pydantic import BaseModel

from mcp_runtime.bridge.annotations import return_annotation
from mcp_runtime.bridge.arguments import arguments, injected
from mcp_runtime.questions import Asked, replies
from mcp_runtime.tool_result import output_schema

__all__ = ["to_mcp_tool"]


def to_mcp_tool(tool: BaseTool) -> MCPTool:
    """Convert a LangChain tool to an MCP tool, deriving its output schema.

    Raises ``RuntimeError`` (naming the tool) when the return annotation does
    not follow the ToolResult contract — see the module docstring.
    """
    if not isinstance(tool.args_schema, type) or not issubclass(
        tool.args_schema, BaseModel
    ):
        raise RuntimeError(
            f"tool {tool.name!r} has a dict args_schema; MCP tools are built "
            "from a pydantic model"
        )
    if names := injected(tool):
        raise RuntimeError(
            f"tool {tool.name!r} takes injected arguments {names}, which "
            "nothing on the MCP side can fill"
        )
    model, schema = output_schema(tool.name, return_annotation(tool))
    arg_model, parameters = arguments(tool)

    async def call(ctx: Context | None = None, **arguments: Any) -> Any:
        with replies(ctx.input_responses if ctx else None):
            try:
                return await tool.ainvoke(arguments)
            except Asked as asked:
                return asked.result()

    return MCPTool(
        fn=call,
        name=tool.name,
        description=tool.description,
        parameters=parameters,
        fn_metadata=FuncMetadata(
            arg_model=arg_model,
            output_model=model,
            output_schema=schema,
            wrap_output=False,
        ),
        is_async=True,
        context_kwarg="ctx",
    )
