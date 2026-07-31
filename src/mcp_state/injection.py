"""Fill tool parameters from session state instead of from the model.

Two paths, applied by one function, in order of how little the model has to do.

**Declared.** A server that tags a parameter with
:class:`mcp_runtime.declarations.Kind` gets the strong form: the parameter is
removed from the schema the model sees, and filled at call time from
``tool_state`` by matching the kind. The model neither generates the value nor
knows the parameter exists — zero tokens, and nothing to hallucinate.

**Undeclared.** Every other structured parameter gains a ``@state:<key>``
handle branch (:mod:`mcp_state.handles`), so the model can point at a stored
value by name rather than reproducing it. About ten tokens, and it needs
nothing from the server — the path an unmodified third-party tool takes.

Both fall out of one mechanism. LangGraph reads ``InjectedState``
annotations off the tool's *coroutine* as well as its schema
(``langgraph.prebuilt.tool_node._get_all_injected_args``), so a wrapper
coroutine carrying one gets the whole ``tool_state`` dict handed to it at call
time, while ``args_schema`` — the server's raw JSON Schema, minus the declared
parameters and plus the handle branches — is what reaches the model. Keeping
it a plain dict rather than round-tripping through pydantic preserves the
server's schema exactly, which matters now that MCP input schemas may use the
whole of JSON Schema 2020-12.

**A resolved value is validated against the parameter's own schema before
use.** A kind is a nominal type: two servers agreeing on the string
``geojson.AreaOfInterest`` does not make one's payload fit the other's schema.
A value that does not validate is treated as absent, so a mismatch degrades to
the model being asked (or a clear error) rather than the consuming server
receiving something it will reject with no one watching.

Resolution is by **kind**, so the tool that published a value may live in a
different toolset on a different MCP server, and neither end names the other.
The agent is the bus.
"""

from collections.abc import Callable, Container, Mapping
from typing import Annotated, Any

import jsonschema
from langchain_core.tools import BaseTool, StructuredTool, ToolException
from langgraph.prebuilt import InjectedState

from mcp_runtime.declarations import CONSUMES_META_KEY
from mcp_state.handles import dereference, offer_handles
from mcp_state.middleware import publishers
from mcp_state.state import TOOL_STATE_KEY, StateEntry, entries_of_kind

# The wrapper's parameter carrying the injected state. Not in ``args_schema``,
# so the model never sees it; LangGraph fills it from the annotation below.
STATE_PARAM = "injected_state"


def wants(declaration: dict[str, Any]) -> str:
    """The kind a declaration resolves against."""
    return str(declaration.get("kind") or "")


def satisfiable(declaration: dict[str, Any], published: Container[str]) -> bool:
    """Whether anything connected publishes the kind this declaration asks for.

    ``published`` is anything that answers ``kind in …`` — the set of kinds
    from :func:`mcp_state.middleware.published_kinds`, or the richer mapping
    from :func:`mcp_state.middleware.publishers`.
    """
    kind = declaration.get("kind")
    return bool(kind) and kind in published


def model_generatable(declaration: dict[str, Any]) -> bool:
    """Whether the model may be asked for this value when nothing publishes it."""
    return bool(declaration.get("modelGeneratable", True))


def declarations_for(tool: BaseTool) -> list[dict[str, Any]]:
    """A tool's consumed-kind declarations, from its server ``_meta``.

    ``langchain_mcp_adapters`` preserves the MCP tool's ``_meta`` onto the
    converted LangChain tool's ``metadata``, which is what makes a
    server-side declaration reachable here at all.
    """
    meta = (getattr(tool, "metadata", None) or {}).get("_meta") or {}
    found = meta.get(CONSUMES_META_KEY)
    return [item for item in found if isinstance(item, dict)] if found else []


def _property_schema(args_schema: Any, parameter: str) -> dict[str, Any] | None:
    """The sub-schema for one parameter, with the parent's ``$defs`` carried.

    ``$defs`` ride along because a property is very often a ``$ref`` into
    them, and validating the extracted fragment alone would fail to resolve.
    """
    if not isinstance(args_schema, dict):
        return None
    schema = (args_schema.get("properties") or {}).get(parameter)
    if not isinstance(schema, dict):
        return None
    if defs := args_schema.get("$defs"):
        return {**schema, "$defs": defs}
    return schema


def _validates(value: Any, schema: dict[str, Any] | None) -> bool:
    """Whether ``value`` satisfies ``schema`` (vacuously true with no schema)."""
    if schema is None:
        return True
    try:
        jsonschema.validate(value, schema)
    except jsonschema.ValidationError:
        return False
    except jsonschema.SchemaError:
        # A schema we cannot evaluate is not evidence the value is wrong.
        return True
    return True


def _prune(args_schema: Any, parameters: set[str]) -> Any:
    """The server's schema with ``parameters`` removed, otherwise untouched.

    Pure, and deliberately shallow: only ``properties`` and ``required`` name
    the parameters, so everything else — ``$defs``, ``allOf``, annotations —
    survives byte-for-byte. Removing nothing returns the original object, so a
    caller can tell by identity that the schema is untouched.
    """
    if not parameters or not isinstance(args_schema, dict):
        return args_schema
    pruned = dict(args_schema)
    if isinstance(properties := pruned.get("properties"), dict):
        pruned["properties"] = {
            name: schema
            for name, schema in properties.items()
            if name not in parameters
        }
    if isinstance(required := pruned.get("required"), list):
        remaining = [name for name in required if name not in parameters]
        if remaining:
            pruned["required"] = remaining
        else:
            pruned.pop("required", None)
    return pruned


def resolve(
    declaration: dict[str, Any],
    tool_state: dict[str, StateEntry] | None,
    schema: dict[str, Any] | None,
) -> tuple[bool, Any]:
    """Find the state entry satisfying one declaration.

    Returns ``(found, value)``. The most recently published entry of the
    declared kind that also validates against ``schema`` is used, so a stale or
    foreign-dialect value is passed over rather than injected.
    """
    kind = declaration.get("kind")
    if not kind:
        return False, None
    for _key, entry in entries_of_kind(tool_state, kind):
        value = entry.get("value")
        if _validates(value, schema):
            return True, value
    return False, None


def _missing(
    tool_name: str, declaration: dict[str, Any], producers: list[str] | None
) -> str:
    """What to tell the model when a required parameter cannot be filled.

    ``producers`` names the connected tools that publish the kind, so the model
    is told which one to run rather than left to work it out from the kind
    string. Empty means nothing connected publishes it at all — a wiring fault
    (:mod:`mcp_state.wiring`) rather than a recoverable turn. ``None`` means the
    caller supplied no map to look in, which is not the same claim.
    """
    lead = (
        f"{tool_name} needs {declaration['parameter']!r}, which is supplied from "
        f"session state ({declaration.get('kind') or 'a value'}) rather than by "
        "you, and nothing in this session has published it."
    )
    if producers is None:
        return f"{lead} If a tool produces it, run that one first."
    if not producers:
        return f"{lead} No connected tool publishes it, so it cannot be supplied here."
    if len(producers) == 1:
        return f"{lead} Run {producers[0]} first — it publishes this."
    return f"{lead} Run one of {', '.join(producers)} first — they publish this."


def _bindable(
    tool: BaseTool, published: Mapping[str, list[str]] | None
) -> list[dict[str, Any]]:
    """The declarations this client will act on for one tool.

    A declaration whose kind nothing connected publishes is dropped when the
    model may generate the value: the parameter then stays in the schema and
    the model fills it, which is what a client implementing none of this would
    do anyway. One that may *not* be model-generated is kept, so the parameter
    is hidden and the tool reports the gap — and :mod:`mcp_state.wiring` can
    withhold it entirely.
    """
    declarations = declarations_for(tool)
    if published is None:
        return declarations
    return [
        declaration
        for declaration in declarations
        if satisfiable(declaration, published) or not model_generatable(declaration)
    ]


def bind_injected(
    tool: BaseTool, published: Mapping[str, list[str]] | None = None
) -> BaseTool:
    """Return ``tool`` with declared parameters hidden, and handles offered.

    ``published`` maps each kind the connected tools publish to the tools that
    publish it (see :func:`mcp_state.middleware.publishers`). Without it every
    declaration is assumed satisfiable and an unfillable parameter cannot name
    what would fill it; :func:`bind_all_injected` supplies it.

    A tool with no declarations and no structured parameters is returned
    unchanged, so this is safe to map over every tool from every server.
    """
    declarations = _bindable(tool, published)
    declared = {item["parameter"] for item in declarations}
    args_schema = offer_handles(_prune(tool.args_schema, declared), frozenset(declared))
    if not declarations and args_schema is tool.args_schema:
        return tool

    schemas = {
        item["parameter"]: _property_schema(tool.args_schema, item["parameter"])
        for item in declarations
    }
    producers: dict[str, list[str] | None] = {
        item["parameter"]: None if published is None else published.get(wants(item), [])
        for item in declarations
    }
    inner: Callable[..., Any] = getattr(tool, "coroutine", None) or getattr(
        tool, "func"
    )

    async def call(
        injected_state: Annotated[
            dict[str, StateEntry] | None, InjectedState(TOOL_STATE_KEY)
        ] = None,
        runtime: Any = None,
        **arguments: Any,
    ) -> Any:
        arguments = dereference(arguments, injected_state)
        for declaration in declarations:
            parameter = declaration["parameter"]
            if parameter in arguments:
                continue  # an explicit value wins; never override a caller
            found, value = resolve(declaration, injected_state, schemas[parameter])
            if found:
                arguments[parameter] = value
            elif declaration.get("required", True):
                raise ToolException(
                    _missing(tool.name, declaration, producers[parameter])
                )
        return await inner(runtime=runtime, **arguments)

    # `metadata` is carried so the tool's `_meta` survives binding — a UI reads
    # its `ui://` view URI from there. `response_format` is the wrapped tool's
    # own, because `call` returns whatever the tool returned: adapter-loaded
    # tools are `content_and_artifact`, but a locally defined one need not be.
    return StructuredTool(
        name=tool.name,
        description=tool.description,
        args_schema=args_schema,
        coroutine=call,
        response_format=tool.response_format,
        metadata=tool.metadata,
        handle_tool_error=tool.handle_tool_error,
    )


def bind_all_injected(tools: list[BaseTool]) -> list[BaseTool]:
    """Apply :func:`bind_injected` across a whole toolset load.

    Resolves what is published once over the full set, so a model-generatable
    parameter with no publisher connected degrades to model-supplied rather
    than to a tool that always raises.
    """
    published = publishers(tools)
    return [bind_injected(tool, published) for tool in tools]
