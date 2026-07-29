"""Parameters the *caller* supplies from agent state, not the model.

Some tool inputs are bulk data the model has no business producing: a clip
geometry, a dataset manifest, a raster footprint. Asking a model for them
burns tokens on a value it can only copy imperfectly. But the value usually
already exists — an earlier tool in the same session returned it.

A toolset marks such a parameter with :class:`Injected`::

    @tool
    async def clip_raster(
        dataset_id: str,
        aoi: Annotated[FeatureCollection, Injected(kind=GEOJSON_AREA_OF_INTEREST)],
    ) -> ClipResult | ToolError: ...

The parameter stays in the tool's ``inputSchema`` — it is a real input, a
plain MCP client can still supply it, and its schema is the contract. What
:class:`Injected` adds is a *declaration*, stamped into the tool's ``_meta``
(the channel :mod:`mcp_runtime.views` already uses for ``ui://`` URIs), that
says: a client holding session state may fill this, and a client that does
should hide it from the model.

This is the input-side mirror of :mod:`mcp_runtime.tool_result`. A
``ToolResult`` subclass declares the data keys a tool *publishes* into session
state; :class:`Injected` declares what a tool *consumes* from it. Both are
read off the same annotations the MCP schemas are derived from, so the
declaration cannot drift from the signature.

Resolution is by **kind** (:mod:`mcp_runtime.kinds`), so a consuming toolset
never names the producing one. Nothing here executes at call time; a server
only advertises. Filling the value is the client's job
(:mod:`mcp_state.injection`).
"""

from dataclasses import dataclass
from typing import Annotated, Any, cast, get_args, get_origin, get_type_hints

from langchain_core.tools import BaseTool
from mcp.server.fastmcp.tools import Tool as FastMCPTool
from pydantic import BaseModel

from mcp_runtime.fastmcp_output import _arms, _return_annotation

# The ``_meta`` key a client reads injected-parameter declarations from.
# Reverse-DNS per the MCP ``_meta`` convention, so it cannot collide with
# ``ui`` (MCP Apps) or another extension's keys.
INJECTED_META_KEY = "io.developmentseed.toolsets/injected"

# The ``_meta`` key a client reads *produced* state-key declarations from —
# the same information ``StateCaptureMiddleware`` needs to namespace writes.
PRODUCES_META_KEY = "io.developmentseed.toolsets/produces"

# Separator between the owning toolset and the field name in a state key.
NAMESPACE_SEP = "/"


@dataclass(frozen=True)
class Injected:
    """Mark a tool parameter as caller-supplied rather than model-supplied.

    Args:
        kind: The semantic type of the value, from :mod:`mcp_runtime.kinds`
            (e.g. ``"geojson.AreaOfInterest"``). A client matches the parameter
            against state entries published under the same kind.
        key: An exact state key to read instead, fully qualified
            (``"dataset-search/geometry"``). Couples the two toolsets by name,
            so use it only when one specific producer is required.
        required: Whether the tool cannot run without it. A client that finds
            no matching state entry for a required parameter should say so
            rather than call with it missing.
        model_fallback: Whether the model may supply the value when *nothing
            connected publishes this kind at all* — a mistyped kind, or a
            producer that isn't deployed. ``True`` leaves the parameter in the
            model's schema; ``False`` (the default) withholds the tool. Set it
            for values small enough to be worth asking a model for, such as a
            bounding box.
    """

    kind: str | None = None
    key: str | None = None
    required: bool = True
    model_fallback: bool = False

    def __post_init__(self) -> None:
        if not self.kind and not self.key:
            raise ValueError("Injected needs a kind or an explicit key")


@dataclass(frozen=True)
class Kind:
    """Tag a ``ToolResult`` data key with the semantic type it publishes.

    The output-side counterpart of :class:`Injected`'s ``kind``::

        class SearchDatasetsResult(ToolResult):
            geometry: NotRequired[
                Annotated[FeatureCollection, Kind(GEOJSON_AREA_OF_INTEREST)]
            ]

    An untagged data key is still captured into state; it simply cannot
    satisfy a kind-resolved injected parameter.
    """

    kind: str


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


def injected_params(tool: BaseTool) -> dict[str, Injected]:
    """The ``Injected`` declarations on a tool's parameters, by parameter name."""
    fn = getattr(tool, "coroutine", None) or getattr(tool, "func", None)
    if fn is None:
        return {}
    hints = get_type_hints(fn, include_extras=True)
    return {
        name: marker
        for name, annotation in hints.items()
        if name != "return" and (marker := _annotation_marker(annotation, Injected))
    }


def produced_keys(tool: BaseTool) -> dict[str, str | None]:
    """A tool's ``ToolResult`` data keys mapped to their declared kind.

    Reads the same return annotation :mod:`mcp_runtime.fastmcp_output` derives
    the output schema from, so the two cannot disagree. ``message`` is the
    model-facing text, never state, and is excluded. A key with no
    :class:`Kind` tag maps to ``None``.
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


def _declaration(name: str, marker: Injected) -> dict[str, Any]:
    """One parameter's wire-form declaration."""
    declaration: dict[str, Any] = {"parameter": name, "required": marker.required}
    if marker.kind:
        declaration["kind"] = marker.kind
    if marker.key:
        declaration["stateKey"] = marker.key
    if marker.model_fallback:
        declaration["modelFallback"] = True
    return declaration


def _optional_in_schema(tool: BaseTool) -> set[str]:
    """The tool's parameters that its own input schema does not require."""
    model = cast(type[BaseModel], tool.get_input_schema())
    schema = model.model_json_schema()
    return set(schema.get("properties", {})) - set(schema.get("required", []))


def state_declarations(tools: list[BaseTool]) -> dict[str, Any]:
    """What this toolset publishes and consumes, for its ``/health`` route.

    The same declarations :func:`with_injected_meta` stamps into ``_meta``,
    restated where a plain HTTP client can read them without speaking MCP —
    the route ``credential_headers`` already travels, and what lets the index
    show a deployment's data flow.
    """
    produces = sorted(
        {
            kind
            for tool in tools
            for kind in produced_keys(tool).values()
            if kind is not None
        }
    )
    injects = [
        {"tool": tool.name, **_declaration(parameter, marker)}
        for tool in tools
        for parameter, marker in sorted(injected_params(tool).items())
    ]
    return {"produces": produces, "injects": injects}


def with_injected_meta(
    toolset: str, tools: list[BaseTool], fastmcp_tools: list[FastMCPTool]
) -> list[FastMCPTool]:
    """Return ``fastmcp_tools`` with injected/produced declarations stamped.

    Pure: inputs are left untouched; each tool that declares something is
    replaced by a copy carrying it under ``_meta``.

    Validates two things first, raising and naming the offender so a broken
    declaration fails ``build_server`` rather than at call time — the gate
    :mod:`mcp_runtime.fastmcp_output` applies to returns:

    - every ``Injected`` parameter is really a parameter of its tool;
    - an ``Injected(required=False)`` parameter is optional in the tool's own
      schema, so the tool can run when a client omits it.
    """
    by_name = {tool.name: tool for tool in tools}
    injected: dict[str, dict[str, Injected]] = {}
    produced: dict[str, dict[str, str | None]] = {}

    for name, tool in by_name.items():
        markers = injected_params(tool)
        arg_names = set(tool.args)
        optional = _optional_in_schema(tool) if markers else set()
        for parameter, marker in markers.items():
            if parameter not in arg_names:
                raise RuntimeError(
                    f"tool {name!r} marks {parameter!r} as Injected, but it is "
                    f"not one of its parameters ({', '.join(sorted(arg_names))})"
                )
            # A client omits an unfilled optional parameter entirely, so the
            # tool's own schema has to accept its absence — otherwise the call
            # is rejected server-side before the tool can pick a default.
            if not marker.required and parameter not in optional:
                raise RuntimeError(
                    f"tool {name!r} marks {parameter!r} as Injected(required=False) "
                    "but its own schema requires it — give the parameter a Python "
                    "default so the tool can run without it"
                )
        if markers:
            injected[name] = markers
        keys = produced_keys(tool)
        if keys:
            produced[name] = keys

    def stamped(fastmcp_tool: FastMCPTool) -> FastMCPTool:
        meta = dict(fastmcp_tool.meta or {})
        if markers := injected.get(fastmcp_tool.name):
            meta[INJECTED_META_KEY] = [
                _declaration(parameter, marker)
                for parameter, marker in sorted(markers.items())
            ]
        if keys := produced.get(fastmcp_tool.name):
            meta[PRODUCES_META_KEY] = [
                {"stateKey": qualified(toolset, field), "field": field, "kind": kind}
                for field, kind in sorted(keys.items())
            ]
        if not meta:
            return fastmcp_tool
        return fastmcp_tool.model_copy(update={"meta": meta})

    return [stamped(fastmcp_tool) for fastmcp_tool in fastmcp_tools]
