"""A LangChain tool as an MCP tool, with an output schema from its return.

The SDK builds a tool from a *function*, reading name, description and schema
off its signature. A LangChain tool carries those as data instead, so this
module assembles :class:`mcp.server.mcpserver.tools.Tool` directly: arguments
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

This was ``langchain-mcp-adapters``' ``to_fastmcp`` wrapped in our own
output-schema derivation. That package is retired and its successor
(``langchain.mcp``) converts in the other direction only, so the conversion
is ours now — some 30 lines, and the half we already owned is unchanged.
"""

from types import UnionType
from typing import Any, Union, cast, get_args, get_origin, get_type_hints, is_typeddict

from langchain_core.tools import BaseTool, InjectedToolArg
from mcp.server.mcpserver.tools import Tool as MCPTool
from mcp.server.mcpserver.utilities.func_metadata import ArgModelBase, FuncMetadata
from pydantic import BaseModel, RootModel, create_model


def _return_annotation(tool: BaseTool) -> Any:
    """The return annotation of the function behind a LangChain tool, if any."""
    fn = getattr(tool, "coroutine", None) or getattr(tool, "func", None)
    if fn is None:
        return None
    return get_type_hints(fn).get("return")


def _arms(annotation: Any) -> tuple[Any, ...]:
    """The annotation's union members, or the annotation itself."""
    if get_origin(annotation) in (Union, UnionType):
        return get_args(annotation)
    return (annotation,)


def _structured_dict(arm: Any) -> bool:
    """Whether one annotation arm maps 1:1 onto a structuredContent object."""
    return is_typeddict(arm) or (isinstance(arm, type) and issubclass(arm, BaseModel))


def _resolve(schema: dict[str, Any], defs: dict[str, Any]) -> dict[str, Any]:
    """A schema node with its top-level ``$ref`` resolved against ``$defs``."""
    reference = schema.get("$ref", "")
    if reference.startswith("#/$defs/"):
        return defs.get(reference.removeprefix("#/$defs/"), {})
    return schema


def _shape_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """The advertised schema, object-rooted as MCP clients expect.

    ``RootModel`` schemas root at a ``$ref`` (single model) or an ``anyOf``
    (union): inline the former, stamp ``"type": "object"`` on the latter.
    """
    defs = dict(schema.get("$defs", {}))
    if schema.get("$ref", "").startswith("#/$defs/"):
        inlined = dict(_resolve(schema, defs))
        defs.pop(schema["$ref"].removeprefix("#/$defs/"))
        if defs:
            inlined["$defs"] = defs
        return inlined
    if "anyOf" in schema:
        return {"type": "object", "anyOf": schema["anyOf"], "$defs": defs}
    return schema


def _offers_message(schema: dict[str, Any]) -> bool:
    """Whether some arm of the schema requires a str ``message`` property."""
    defs: dict[str, Any] = schema.get("$defs", {})
    arms = [_resolve(arm, defs) for arm in schema.get("anyOf", [schema])]
    return any(
        arm.get("properties", {}).get("message", {}).get("type") == "string"
        and "message" in arm.get("required", [])
        for arm in arms
    )


def _injected(tool: BaseTool) -> list[str]:
    """Argument names the LangChain tool expects to be filled in for it.

    ``Annotated[str, InjectedToolArg]`` reaches pydantic as field metadata,
    which is where this reads it — the marker is the whole declaration, so a
    caller is meant to supply the value and a model never sees the parameter.
    Nothing on the MCP side fills one, so a tool that takes one cannot be
    served and :func:`to_mcp_tool` refuses it.
    """

    def marked(metadata: list[Any]) -> bool:
        return any(
            isinstance(entry, InjectedToolArg)
            or (isinstance(entry, type) and issubclass(entry, InjectedToolArg))
            for entry in metadata
        )

    schema = cast(type[BaseModel], tool.args_schema)
    return [name for name, info in schema.model_fields.items() if marked(info.metadata)]


def _arguments(tool: BaseTool) -> tuple[type[ArgModelBase], dict[str, Any]]:
    """The argument model to validate against, and the schema to publish.

    Both forbid what the tool does not declare. Pydantic's default is to drop
    an undeclared argument: the call succeeds, the tool runs without it, the
    result looks ordinary, and the one thing that would explain it — that an
    argument went missing — is the one thing nobody is told. A validation
    error naming the parameter is something a model reads and corrects.

    Both halves are needed. Forbidding extras on the model is the enforcement;
    ``additionalProperties: false`` on the published schema is what tells a
    client the rule exists. Enforcing without publishing would refuse calls
    the advertised schema allowed, which is its own kind of surprise.
    """
    call_schema = cast(type[BaseModel], tool.tool_call_schema)
    fields = {
        name: (info.annotation, info) for name, info in call_schema.model_fields.items()
    }
    model = create_model(
        f"{tool.name}Arguments",
        **fields,  # type: ignore[call-overload]
        __base__=ArgModelBase,
    )
    model.model_config["extra"] = "forbid"
    model.model_rebuild(force=True)
    schema = call_schema.model_json_schema()
    schema["additionalProperties"] = False
    return model, schema


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
    if injected := _injected(tool):
        raise RuntimeError(
            f"tool {tool.name!r} takes injected arguments {injected}, which "
            "nothing on the MCP side can fill"
        )
    annotation = _return_annotation(tool)
    if annotation is None or not all(
        _structured_dict(arm) for arm in _arms(annotation)
    ):
        raise RuntimeError(
            f"tool {tool.name!r} needs a ToolResult return annotation "
            f"(a TypedDict/BaseModel or a union of them; got {annotation!r}) "
            "— see mcp_runtime.tool_result"
        )
    model: type[BaseModel] = RootModel[annotation]  # type: ignore[valid-type]
    schema = _shape_schema(model.model_json_schema())
    if not _offers_message(schema):
        raise RuntimeError(
            f"tool {tool.name!r} does not follow the ToolResult contract: no arm "
            "of its return annotation has a required str 'message' property"
        )
    arg_model, parameters = _arguments(tool)

    async def call(**arguments: Any) -> Any:
        return await tool.ainvoke(arguments)

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
    )
