"""A LangChain tool's arguments as the MCP server validates and publishes them."""

from typing import Any, cast

from langchain_core.tools import BaseTool, InjectedToolArg

# Internal to the SDK, and imported only in this package: a minor release can
# change it, and test_the_sdk_internals_we_build_on_keep_their_shape says so.
from mcp.server.mcpserver.utilities.func_metadata import ArgModelBase
from pydantic import BaseModel, create_model


def injected(tool: BaseTool) -> list[str]:
    """Argument names the LangChain tool expects to be filled in for it.

    ``Annotated[str, InjectedToolArg]`` reaches pydantic as field metadata,
    which is where this reads it — the marker is the whole declaration, so a
    caller is meant to supply the value and a model never sees the parameter.
    Nothing on the MCP side fills one, so a tool that takes one cannot be
    served and :func:`~mcp_runtime.bridge.to_mcp_tool` refuses it.
    """

    def marked(metadata: list[Any]) -> bool:
        return any(
            isinstance(entry, InjectedToolArg)
            or (isinstance(entry, type) and issubclass(entry, InjectedToolArg))
            for entry in metadata
        )

    schema = cast(type[BaseModel], tool.args_schema)
    return [name for name, info in schema.model_fields.items() if marked(info.metadata)]


def arguments(tool: BaseTool) -> tuple[type[ArgModelBase], dict[str, Any]]:
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
