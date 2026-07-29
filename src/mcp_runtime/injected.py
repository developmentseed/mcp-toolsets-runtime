"""Parameters the *caller* supplies from agent state, not the model.

Some tool inputs are bulk data the model has no business producing: a clip
geometry, a dataset manifest, a raster footprint. Asking a model for them
burns tokens on a value it can only copy imperfectly. But the value usually
already exists — an earlier tool in the same session returned it.

A toolset marks such a parameter with :class:`Injected`::

    @tool
    async def clip_raster(
        dataset_id: str,
        aoi: Annotated[FeatureCollection, Injected(kind=GEOJSON)],
    ) -> ClipResult | ToolError: ...

The parameter stays in the tool's ``inputSchema`` — it is a real input, a
plain MCP client can still supply it, and its schema is the contract. What
:class:`Injected` adds is a *declaration*, stamped into the tool's ``_meta``
(the channel :mod:`mcp_runtime.views` already uses for ``ui://`` URIs), that
says: a client holding session state may fill this, and a client that does
should hide it from the model.

This is the input-side mirror of :mod:`mcp_runtime.tool_result`. A
``ToolResult`` subclass declares the keys a tool *publishes* into session
state; :class:`Injected` declares the keys a tool *consumes* from it. Both
are read off the same annotations the MCP schemas are derived from, so the
declaration cannot drift from the signature.

Resolution is by **kind**, not by name — see :func:`qualified` for why.
Nothing here executes at call time; a server only advertises. Filling the
value is the client's job (``mcp_agent.injection``).
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
        kind: The semantic type of the value, e.g. ``"geojson.FeatureCollection"``.
            This is what a client resolves on: it matches the parameter against
            state entries published under the same kind, so a consuming toolset
            never has to name the producing one. Strongly recommended.
        key: An exact state key to read instead, fully qualified
            (``"dataset-search/geometry"``). Use only when a specific producer
            is genuinely required — it couples the two toolsets by name.
        required: Whether the tool cannot run without it. A client that finds
            no matching state entry for a required parameter should say so
            rather than call with it missing.
        model_fallback: What to do when *nothing connected publishes this at
            all* — a mistyped kind, or a producer that isn't deployed. The
            parameter is in the advertised ``inputSchema`` either way, so
            ``True`` leaves it visible and lets the model supply it, which is
            what a client that implements none of this already does. Defaults
            to ``False``, because a parameter is usually marked injected
            precisely because model-generating it is the thing to avoid: a
            large geometry is better as a missing tool than as a silent
            token bill. Set it for values small enough to be worth asking for,
            like a bounding box.
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
            geometry: NotRequired[Annotated[FeatureCollection, Kind(GEOJSON)]]

    An untagged data key is still captured into state; it simply cannot
    satisfy a kind-resolved injected parameter.
    """

    kind: str


def qualified(toolset: str, field: str) -> str:
    """The namespaced state key a toolset's data key is published under.

    Data keys are ``ToolResult`` field names, so two toolsets independently
    choosing ``geometry`` is not unlikely — and session state is a single
    flat namespace merged last-write-wins, where that collision is silent
    corruption rather than an error. Qualifying every write by its owning
    toolset makes collisions impossible by construction.

    Qualification is a *storage* concern only. Consumers resolve by kind
    (:class:`Injected`), so namespacing costs them no coupling: a tool asks
    for "a GeoJSON FeatureCollection", not for "dataset-search's geometry".
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
    found = {
        name: marker
        for name, annotation in hints.items()
        if name != "return" and (marker := _annotation_marker(annotation, Injected))
    }
    return found


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


def _declaration(toolset: str, name: str, marker: Injected) -> dict[str, Any]:
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


def state_declarations(toolset: str, tools: list[BaseTool]) -> dict[str, Any]:
    """What this toolset publishes and consumes, for its ``/health`` route.

    The same declarations :func:`with_injected_meta` stamps into ``_meta``,
    restated where a plain HTTP client can read them. A server cannot tell on
    its own whether an injected kind is satisfiable — the producer is usually
    another toolset — so advertising both halves lets the index aggregate
    across a deployment and report a broken wire without connecting over MCP.
    Exactly how ``credential_headers`` already travels.
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
        {"tool": tool.name, **_declaration(toolset, parameter, marker)}
        for tool in tools
        for parameter, marker in sorted(injected_params(tool).items())
    ]
    return {"produces": produces, "injects": injects}


def with_injected_meta(
    toolset: str, tools: list[BaseTool], fastmcp_tools: list[FastMCPTool]
) -> list[FastMCPTool]:
    """Return ``fastmcp_tools`` with injected/produced declarations stamped.

    Pure: inputs are left untouched; each tool that declares something is
    replaced by a copy carrying it under ``_meta``. Validates first that every
    ``Injected`` parameter actually exists on the tool, and that a toolset
    which consumes a kind it also produces is not silently self-referential —
    raising (naming the offender) so a broken declaration fails
    ``build_server`` rather than at call time, the same gate
    :mod:`mcp_runtime.fastmcp_output` applies to returns.
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
                _declaration(toolset, parameter, marker)
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
