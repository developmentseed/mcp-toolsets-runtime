"""What a tool exchanges with session state, read off its own signature.

Two things a tool says about itself, both derived from annotations it already
carries for other reasons.

**What it publishes.** Every data key of a tool's ``ToolResult`` — every field
but ``message`` — is a value the tool produces, and is captured into session
state under a key naming where it came from. Being a data key rather than
prose is the whole declaration; nothing further is written::

    class SearchDatasetsResult(ToolResult):
        datasets: NotRequired[list[str]]
        area_of_interest: NotRequired[FeatureCollection]

Those field names travel. A model choosing a stored value to pass onward reads
the key it is stored under, so **a data key is a public name**: it is what the
next tool's model has to recognise. ``area_of_interest`` and ``footprint`` are
the same JSON, and only the name says which is which.

**What a model may not write.** :class:`NotAuthored` on a parameter says the
caller must supply a value that already exists. It names no type, so no second
toolset has to agree with anything, and it says nothing about session state —
a tool carrying it behaves normally against a client that ignores it::

    @tool
    async def clip_raster(
        dataset_id: str,
        aoi: Annotated[FeatureCollection, NotAuthored()],
    ) -> ClipResult | ToolError: ...

Nothing here runs at call time; a server only advertises.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Any, get_args, get_origin, get_type_hints

from langchain_core.tools import BaseTool
from mcp.server.fastmcp.tools import Tool as FastMCPTool

from mcp_runtime.fastmcp_output import _arms, _return_annotation

# The ``_meta`` keys a client reads declarations from. Reverse-DNS per the MCP
# ``_meta`` convention, so they cannot collide with ``ui`` (MCP Apps) or
# another extension's keys.
PRODUCES_META_KEY = "io.developmentseed.toolsets/produces"
NOT_AUTHORED_META_KEY = "io.developmentseed.toolsets/notAuthored"

# Separator between the parts of a state key.
NAMESPACE_SEP = "/"


@dataclass(frozen=True)
class NotAuthored:
    """A parameter whose value a model must not write.

    A claim about the parameter, not about where a value comes from: *the
    caller must supply one that already exists*. It says nothing about session
    state, and a tool carrying it works unchanged against a client that has
    never heard of any of this — which is the point, because the author of a
    tool should not have to know how a client moves values around in order to
    describe their own parameter.

    What a client does with it scales with how much it implements. One that
    ignores ``_meta`` leaves the parameter alone and the model fills it. One
    that reads only the description finds the sentence
    :data:`NOT_AUTHORED_NOTE` appended, and mostly obeys it. This package's
    client (:mod:`mcp_state.injection`) narrows the parameter's schema so that
    a literal is not accepted at all — the only value it will take is a
    ``@state:<key>`` reference to something already published.

    Use it for a value a model can only fake: a 2000-vertex geometry, an item
    collection, a bounding box that has to be *the* one under discussion rather
    than a plausible-looking set of four numbers.

    **It binds a parameter, not a concept.** The constraint reaches exactly the
    parameter it annotates, so a tool that accepts the same value a second way
    — an opaque ``dict[str, Any]`` request body with a field of its own for it —
    is unconstrained by that route, and a model asked for the same thing by two
    surfaces will answer both. Reading a stored value is a supported move
    (``inspect_state``), which is all it takes to obtain one to write. A tool
    with such a parameter has to reconcile the two itself; leaving one to
    silently win discards the other while the client's receipts still report it
    as used.
    """


#: Appended to a ``NotAuthored`` parameter's description in the served schema,
#: so a client that reads nothing but the schema still passes the constraint on
#: to its model.
NOT_AUTHORED_NOTE = (
    "This value must already exist — it comes from an earlier tool result, not "
    "from you. Do not write one."
)


def qualified(toolset: str, tool: str, field: str) -> str:
    """The state key a tool's data key is published under.

    Three parts, ``<toolset>/<tool>/<field>``. Session state is one namespace
    merged last-write-wins, so qualifying is what stops two toolsets that both
    return ``geometry`` overwriting each other. The **tool** is in there because
    the key is read: a model choosing between stored values is told
    ``dataset-search/search_datasets/area_of_interest`` rather than
    ``dataset-search/area_of_interest``, and the difference is whether it knows
    which call produced the value it is about to reuse.
    """
    return NAMESPACE_SEP.join((toolset, tool, field))


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


def output_fields(tool: BaseTool) -> list[str]:
    """A tool's ``ToolResult`` data keys, sorted.

    Reads the same return annotation :mod:`mcp_runtime.fastmcp_output` derives
    the output schema from, so the two cannot disagree. ``message`` is the
    model-facing text, never state, and is excluded along with the error arm's
    own fields.
    """
    annotation = _return_annotation(tool)
    if annotation is None:
        return []
    fields = {
        field
        for arm in _arms(annotation)
        for field in get_type_hints(arm, include_extras=True)
        if field not in ("message", "error", "detail")
    }
    return sorted(fields)


def not_authored(tool: BaseTool) -> list[str]:
    """The tool's parameters tagged :class:`NotAuthored`, in order."""
    fn = getattr(tool, "coroutine", None) or getattr(tool, "func", None)
    if fn is None:
        return []
    hints = get_type_hints(fn, include_extras=True)
    return [
        name
        for name, annotation in hints.items()
        if name != "return" and _annotation_marker(annotation, NotAuthored)
    ]


def _noted(parameters: dict[str, Any], names: Sequence[str]) -> dict[str, Any]:
    """``parameters`` with :data:`NOT_AUTHORED_NOTE` on each named property.

    The note goes in the JSON Schema rather than only in ``_meta`` so that the
    constraint survives a client that reads no extensions at all: the
    description is the one field every MCP client already passes to its model.
    """
    properties = parameters.get("properties")
    if not isinstance(properties, dict):
        return parameters
    updated = dict(properties)
    for name in names:
        schema = updated.get(name)
        if not isinstance(schema, dict):
            continue
        existing = str(schema.get("description") or "").strip()
        updated[name] = {
            **schema,
            "description": f"{existing} {NOT_AUTHORED_NOTE}".strip(),
        }
    if updated == properties:
        return parameters
    return {**parameters, "properties": updated}


def state_declarations(toolset: str, tools: list[BaseTool]) -> dict[str, Any]:
    """What this toolset publishes and will not let a model write.

    The same declarations :func:`with_state_meta` stamps into ``_meta``,
    restated where a plain HTTP client can read them without speaking MCP —
    the route ``credential_headers`` already travels, and what lets the index
    show a deployment's data flow. ``not_authored`` is the part of that flow a
    deployment cannot satisfy on its own: a parameter listed there needs some
    other toolset to have published a value first.
    """
    produces = [
        {
            "tool": tool.name,
            "field": field,
            "state_key": qualified(toolset, tool.name, field),
        }
        for tool in tools
        for field in output_fields(tool)
    ]
    withheld = [
        {"tool": tool.name, "parameter": parameter}
        for tool in tools
        for parameter in not_authored(tool)
    ]
    return {"produces": produces, "not_authored": withheld}


def with_state_meta(
    toolset: str, tools: list[BaseTool], fastmcp_tools: list[FastMCPTool]
) -> list[FastMCPTool]:
    """Return ``fastmcp_tools`` with what each tool declares stamped on.

    Published data keys go into ``_meta``; :class:`NotAuthored` parameters go
    into ``_meta`` *and* into the served input schema, as a sentence on the
    parameter's description. The second is what makes the constraint mean
    something to a client that reads no extensions.

    Pure: inputs are left untouched; each tool that declares something is
    replaced by a copy carrying it.

    Raises if :class:`NotAuthored` tags something that is not one of its tool's
    parameters, naming the offender so a typo fails ``build_server`` rather
    than going unnoticed until a client connects — the gate
    :mod:`mcp_runtime.fastmcp_output` applies to returns.
    """
    publishes_by_tool: dict[str, list[str]] = {}
    not_authored_by_tool: dict[str, list[str]] = {}

    for tool in tools:
        arg_names = set(tool.args)
        for parameter in not_authored(tool):
            if parameter not in arg_names:
                raise RuntimeError(
                    f"tool {tool.name!r} tags {parameter!r} NotAuthored, but it "
                    f"is not one of its parameters ({', '.join(sorted(arg_names))})"
                )
        if fields := output_fields(tool):
            publishes_by_tool[tool.name] = fields
        if names := not_authored(tool):
            not_authored_by_tool[tool.name] = names

    def stamped(fastmcp_tool: FastMCPTool) -> FastMCPTool:
        meta = dict(fastmcp_tool.meta or {})
        if fields := publishes_by_tool.get(fastmcp_tool.name):
            meta[PRODUCES_META_KEY] = [
                {
                    "stateKey": qualified(toolset, fastmcp_tool.name, field),
                    "field": field,
                }
                for field in fields
            ]
        names = not_authored_by_tool.get(fastmcp_tool.name)
        if names:
            meta[NOT_AUTHORED_META_KEY] = names
        if not meta:
            return fastmcp_tool
        update: dict[str, Any] = {"meta": meta}
        if names:
            update["parameters"] = _noted(fastmcp_tool.parameters, names)
        return fastmcp_tool.model_copy(update=update)

    return [stamped(fastmcp_tool) for fastmcp_tool in fastmcp_tools]
