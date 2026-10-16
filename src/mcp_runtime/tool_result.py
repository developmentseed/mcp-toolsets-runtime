"""The base contract every tool return follows.

A tool returns one dict per call, in one of two shapes:

- :class:`ToolResult` — success: a ``message`` (the text the model reads)
  plus any data keys the tool declares, captured into session state.
- :class:`ToolError` — failure: a machine-readable ``error`` kind and a
  ``detail`` saying what happened or what to do next.

Subclass :class:`ToolResult` per tool, one ``NotRequired`` field per data
key, and annotate the ``@tool`` function with the union::

    class SearchDatasetsResult(ToolResult):
        datasets: NotRequired[list[dict[str, Any]]]

    @tool
    async def search_datasets(query: str) -> SearchDatasetsResult | ToolError:
        ...

``mcp_runtime.bridge`` derives the tool's MCP ``outputSchema`` from
the annotation and refuses to serve a tool that breaks the contract — a
deploy-time gate, not a convention.
"""

from types import UnionType
from typing import Any, TypedDict, Union, get_args, get_origin, is_typeddict

from pydantic import BaseModel, RootModel
from typing_extensions import TypeIs


class ToolResult(TypedDict):
    """A successful tool return: a ``message`` plus declared data keys."""

    message: str


class ToolError(TypedDict):
    """A structured tool error: what kind, and what to do about it."""

    error: str
    detail: str


def is_error(result: dict[str, Any] | ToolError) -> TypeIs[ToolError]:
    """Whether a helper's raw dict is an error return (truthy ``error`` key)."""
    return bool(result.get("error"))


def arms(annotation: Any) -> tuple[Any, ...]:
    """The annotation's union members, or the annotation itself."""
    if get_origin(annotation) in (Union, UnionType):
        return get_args(annotation)
    return (annotation,)


def output_schema(name: str, annotation: Any) -> tuple[type[BaseModel], dict[str, Any]]:
    """The output model a tool's returns are validated against, and its schema.

    Raises ``RuntimeError`` naming the tool when ``annotation`` (the tool's
    return annotation) breaks the contract.
    """
    if annotation is None or not all(_structured_dict(arm) for arm in arms(annotation)):
        raise RuntimeError(
            f"tool {name!r} needs a ToolResult return annotation "
            f"(a TypedDict/BaseModel or a union of them; got {annotation!r}) "
            "— see mcp_runtime.tool_result"
        )
    model: type[BaseModel] = RootModel[annotation]
    schema = _shape_schema(model.model_json_schema())
    if not _offers_message(schema):
        raise RuntimeError(
            f"tool {name!r} does not follow the ToolResult contract: no arm "
            "of its return annotation has a required str 'message' property"
        )
    return model, schema


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
    members = [_resolve(arm, defs) for arm in schema.get("anyOf", [schema])]
    return any(
        arm.get("properties", {}).get("message", {}).get("type") == "string"
        and "message" in arm.get("required", [])
        for arm in members
    )
