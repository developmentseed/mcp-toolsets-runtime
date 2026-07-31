"""What a tool exchanges with session state, read off its own signature.

One marker, :class:`Kind`, used on both sides of a tool. On a ``ToolResult``
data key it says what the tool publishes; on a parameter it says what the tool
takes::

    class SearchDatasetsResult(ToolResult):
        geometry: NotRequired[
            Annotated[FeatureCollection, Kind(GEOJSON_AREA_OF_INTEREST)]
        ]


    @tool
    async def clip_raster(
        dataset_id: str,
        aoi: Annotated[FeatureCollection, Kind(GEOJSON_AREA_OF_INTEREST)],
    ) -> ClipResult | ToolError: ...

A kind names *what a value is*. It says nothing about where a value comes
from — that is the client's decision, made against everything it connected to
(:mod:`mcp_state.injection`). A server cannot know whether some other toolset
publishes the kind its tool consumes, so it does not try to.

The declaration is an **accelerator, not a requirement**. A client can move
values between tools that declare nothing, by recognising values by shape and
letting the model pass them by handle (:mod:`mcp_state.handles`). Declaring
buys three things that path cannot: the parameter leaves the model's schema
entirely, resolution costs no model turn, and a tag on something that is not a
parameter is caught at ``build_server`` rather than at connect.

Both sides are read from the same annotations the MCP schemas are derived
from, so a declaration cannot drift from the signature. Nothing here runs at
call time; a server only advertises.
"""

from dataclasses import dataclass
from typing import Annotated, Any, cast, get_args, get_origin, get_type_hints

from langchain_core.tools import BaseTool
from mcp.server.fastmcp.tools import Tool as FastMCPTool
from pydantic import BaseModel

from mcp_runtime.fastmcp_output import _arms, _return_annotation

# The ``_meta`` keys a client reads declarations from. Reverse-DNS per the MCP
# ``_meta`` convention, so they cannot collide with ``ui`` (MCP Apps) or
# another extension's keys.
PRODUCES_META_KEY = "io.developmentseed.toolsets/produces"
CONSUMES_META_KEY = "io.developmentseed.toolsets/consumes"

# Separator between the owning toolset and the field name in a state key.
NAMESPACE_SEP = "/"


@dataclass(frozen=True)
class Kind:
    """The semantic type of a value a tool publishes or takes.

    Args:
        kind: The type, from :mod:`mcp_runtime.kinds` (e.g.
            ``"geojson.AreaOfInterest"``). Two toolsets interoperate by
            agreeing on this string and nothing else.
        model_generatable: Whether a model may be asked for the value when no
            connected tool publishes the kind. ``True`` (the default) keeps the
            parameter in the model's schema, so the tool still works where a
            plain MCP client's would. Set ``False`` for a value a model can
            only fake — a 2000-vertex geometry, an item collection — and the
            tool is withheld instead. Meaningless on an output, where nothing
            is being asked of a model.
    """

    kind: str
    model_generatable: bool = True


def qualified(toolset: str, field: str) -> str:
    """The namespaced state key a toolset's data key is published under.

    Data keys are ``ToolResult`` field names, so two toolsets can easily both
    choose ``geometry``. Session state is one namespace merged last-write-wins,
    so qualifying every write by its owning toolset is what stops one
    overwriting the other. Storage only — consumers resolve by kind.
    """
    return f"{toolset}{NAMESPACE_SEP}{field}"


def _annotation_marker(annotation: Any, marker: type) -> Any | None:
    """The first ``marker`` instance in an ``Annotated[...]``, if any.

    Looks through ``NotRequired``/``Required`` wrappers so a ``ToolResult``
    field can be both optional and tagged.
    """
    if get_origin(annotation) is Annotated:
        for meta in get_args(annotation)[1:]:
            if isinstance(meta, marker):
                return meta
        return None
    args = get_args(annotation)
    return _annotation_marker(args[0], marker) if args else None


def consumed_kinds(tool: BaseTool) -> dict[str, Kind]:
    """The ``Kind`` tags on a tool's parameters, by parameter name."""
    fn = getattr(tool, "coroutine", None) or getattr(tool, "func", None)
    if fn is None:
        return {}
    hints = get_type_hints(fn, include_extras=True)
    return {
        name: marker
        for name, annotation in hints.items()
        if name != "return" and (marker := _annotation_marker(annotation, Kind))
    }


def output_kinds(tool: BaseTool) -> dict[str, str | None]:
    """A tool's ``ToolResult`` data keys mapped to their declared kind.

    Reads the same return annotation :mod:`mcp_runtime.fastmcp_output` derives
    the output schema from, so the two cannot disagree. ``message`` is the
    model-facing text, never state, and is excluded. A key with no
    :class:`Kind` tag maps to ``None``: it is still captured, and a client may
    still recognise its kind from the value's own shape.
    """
    annotation = _return_annotation(tool)
    if annotation is None:
        return {}
    keys: dict[str, str | None] = {}
    for arm in _arms(annotation):
        for field, field_annotation in get_type_hints(arm, include_extras=True).items():
            if field in ("message", "error", "detail"):
                continue
            tag = _annotation_marker(field_annotation, Kind)
            keys[field] = tag.kind if tag else keys.get(field)
    return keys


def _optional_in_schema(tool: BaseTool) -> set[str]:
    """The tool's parameters that its own input schema does not require."""
    model = cast(type[BaseModel], tool.get_input_schema())
    schema = model.model_json_schema()
    return set(schema.get("properties", {})) - set(schema.get("required", []))


def _declaration(name: str, marker: Kind, *, required: bool) -> dict[str, Any]:
    """One parameter's wire-form declaration.

    ``required`` is read from the tool's own input schema rather than declared
    separately: a parameter with a Python default is optional there, which is
    exactly the condition under which a client may leave it out of a call.
    """
    return {
        "parameter": name,
        "kind": marker.kind,
        "required": required,
        "modelGeneratable": marker.model_generatable,
    }


def _parameter_declarations(tool: BaseTool) -> list[dict[str, Any]]:
    """Every consumed-kind declaration for one tool, ordered by parameter."""
    markers = consumed_kinds(tool)
    if not markers:
        return []
    optional = _optional_in_schema(tool)
    return [
        _declaration(parameter, marker, required=parameter not in optional)
        for parameter, marker in sorted(markers.items())
    ]


def state_declarations(tools: list[BaseTool]) -> dict[str, Any]:
    """What this toolset publishes and consumes, for its ``/health`` route.

    The same declarations :func:`with_state_meta` stamps into ``_meta``,
    restated where a plain HTTP client can read them without speaking MCP —
    the route ``credential_headers`` already travels, and what lets the index
    show a deployment's data flow.
    """
    produces = sorted(
        {
            kind
            for tool in tools
            for kind in output_kinds(tool).values()
            if kind is not None
        }
    )
    consumes = [
        {"tool": tool.name, **declaration}
        for tool in tools
        for declaration in _parameter_declarations(tool)
    ]
    return {"produces": produces, "consumes": consumes}


def with_state_meta(
    toolset: str, tools: list[BaseTool], fastmcp_tools: list[FastMCPTool]
) -> list[FastMCPTool]:
    """Return ``fastmcp_tools`` with published/consumed kinds stamped into ``_meta``.

    Pure: inputs are left untouched; each tool that declares something is
    replaced by a copy carrying it.

    Raises if a ``Kind`` tags something that is not one of its tool's
    parameters, naming the offender so a typo fails ``build_server`` rather
    than going unnoticed until a client connects — the gate
    :mod:`mcp_runtime.fastmcp_output` applies to returns.
    """
    consumes_by_tool: dict[str, list[dict[str, Any]]] = {}
    publishes_by_tool: dict[str, dict[str, str | None]] = {}

    for tool in tools:
        arg_names = set(tool.args)
        for parameter in consumed_kinds(tool):
            if parameter not in arg_names:
                raise RuntimeError(
                    f"tool {tool.name!r} tags {parameter!r} with Kind, but it is "
                    f"not one of its parameters ({', '.join(sorted(arg_names))})"
                )
        if declarations := _parameter_declarations(tool):
            consumes_by_tool[tool.name] = declarations
        if kinds := output_kinds(tool):
            publishes_by_tool[tool.name] = kinds

    def stamped(fastmcp_tool: FastMCPTool) -> FastMCPTool:
        meta = dict(fastmcp_tool.meta or {})
        if declarations := consumes_by_tool.get(fastmcp_tool.name):
            meta[CONSUMES_META_KEY] = declarations
        if kinds := publishes_by_tool.get(fastmcp_tool.name):
            meta[PRODUCES_META_KEY] = [
                {"stateKey": qualified(toolset, field), "field": field, "kind": kind}
                for field, kind in sorted(kinds.items())
            ]
        if not meta:
            return fastmcp_tool
        return fastmcp_tool.model_copy(update={"meta": meta})

    return [stamped(fastmcp_tool) for fastmcp_tool in fastmcp_tools]
