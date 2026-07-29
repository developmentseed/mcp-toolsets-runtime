"""Fill caller-supplied tool parameters from agent state, not from the model.

The client half of :mod:`mcp_runtime.injected`. A server declares that a
parameter may be injected; :func:`bind_injected` is what acts on it.

For each declaration on a tool, it does two things:

1. **Removes the parameter from the schema the model sees.** The value never
   has to be generated, so it costs no output tokens and cannot be
   hallucinated or truncated.
2. **Fills it at call time** from ``tool_state`` (:mod:`mcp_state.state`),
   resolving by *kind* — so the tool that published the value may live in a
   different toolset on a different MCP server, and neither end names the
   other. The agent is the bus.

Both fall out of one mechanism. LangGraph reads ``InjectedState``
annotations off the tool's *coroutine* as well as its schema
(``langgraph.prebuilt.tool_node._get_all_injected_args``), so a wrapper
coroutine carrying one gets the whole ``tool_state`` dict handed to it at
call time, while ``args_schema`` — left as the server's raw JSON Schema,
minus the injected properties — is what reaches the model. Keeping it a
plain dict rather than round-tripping through pydantic preserves the
server's schema exactly, which matters now that MCP input schemas may use
the whole of JSON Schema 2020-12.

**A resolved value is validated against the parameter's own schema before
use.** A kind is a nominal type: two servers agreeing on the string
``geojson.AreaOfInterest`` does not make one's payload fit the other's
schema. A value that does not validate is treated as absent, so a mismatch
degrades to the model being asked (or a clear error) rather than the
consuming server receiving something it will reject with no one watching.
"""

from collections.abc import Callable
from typing import Annotated, Any

import jsonschema
from langchain_core.tools import BaseTool, StructuredTool, ToolException
from langgraph.prebuilt import InjectedState

from mcp_state.middleware import produced
from mcp_state.state import TOOL_STATE_KEY, StateEntry, entries_of_kind
from mcp_runtime.injected import INJECTED_META_KEY

# The wrapper's parameter carrying the injected state. Not in ``args_schema``,
# so the model never sees it; LangGraph fills it from the annotation below.
STATE_PARAM = "injected_state"


def wanted(declaration: dict[str, Any]) -> str:
    """What a declaration resolves against: a kind, or ``key:<stateKey>``."""
    if state_key := declaration.get("stateKey"):
        return f"key:{state_key}"
    return str(declaration.get("kind") or "")


def satisfiable(
    declaration: dict[str, Any], published: tuple[frozenset, frozenset]
) -> bool:
    """Whether anything connected publishes what this declaration asks for."""
    kinds, keys = published
    if state_key := declaration.get("stateKey"):
        return state_key in keys
    kind = declaration.get("kind")
    return bool(kind) and kind in kinds


def declarations_for(tool: BaseTool) -> list[dict[str, Any]]:
    """A tool's injected-parameter declarations, from its server ``_meta``.

    ``langchain_mcp_adapters`` preserves the MCP tool's ``_meta`` onto the
    converted LangChain tool's ``metadata``, which is what makes a
    server-side declaration reachable here at all.
    """
    meta = (getattr(tool, "metadata", None) or {}).get("_meta") or {}
    found = meta.get(INJECTED_META_KEY)
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
    survives byte-for-byte.
    """
    if not isinstance(args_schema, dict):
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

    Returns ``(found, value)``. An explicit ``stateKey`` wins when present —
    the declaring tool asked for a specific producer. Otherwise the most
    recently published entry of the declared ``kind`` that also validates
    against ``schema`` is used, so a stale or foreign-dialect value is passed
    over rather than injected.
    """
    if state_key := declaration.get("stateKey"):
        entry = (tool_state or {}).get(state_key)
        if entry is not None and _validates(entry.get("value"), schema):
            return True, entry["value"]
        return False, None

    kind = declaration.get("kind")
    if not kind:
        return False, None
    for _key, entry in entries_of_kind(tool_state, kind):
        value = entry.get("value")
        if _validates(value, schema):
            return True, value
    return False, None


def _missing(tool_name: str, declaration: dict[str, Any]) -> str:
    """What to tell the model when a required parameter cannot be filled."""
    source = declaration.get("stateKey") or declaration.get("kind") or "a value"
    return (
        f"{tool_name} needs {declaration['parameter']!r}, which is supplied from "
        f"session state ({source}) rather than by you, and nothing in this "
        "session has published it. If a tool produces it, run that one first."
    )


def bind_injected(
    tool: BaseTool, published: tuple[frozenset[str], frozenset[str]] | None = None
) -> BaseTool:
    """Return ``tool`` with its injected parameters hidden and auto-filled.

    Tools declaring nothing are returned unchanged, so this is safe to map
    over every tool from every server — including third-party ones, which
    carry no declarations.

    ``published`` is what the connected tools publish (see
    :func:`mcp_state.middleware.produced`). Given it, a declaration nothing can
    satisfy that set ``modelFallback`` is left alone entirely — the parameter
    stays in the schema and the model supplies it, which is what a client
    implementing none of this would do anyway. Without ``published`` every
    declaration is assumed satisfiable; :func:`bind_all_injected` supplies it.
    """
    declarations = declarations_for(tool)
    if published is not None:
        declarations = [
            declaration
            for declaration in declarations
            if satisfiable(declaration, published)
            or not declaration.get("modelFallback")
        ]
    if not declarations:
        return tool

    parameters = {item["parameter"] for item in declarations}
    schemas = {
        item["parameter"]: _property_schema(tool.args_schema, item["parameter"])
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
        for declaration in declarations:
            parameter = declaration["parameter"]
            if parameter in arguments:
                continue  # an explicit value wins; never override a caller
            found, value = resolve(declaration, injected_state, schemas[parameter])
            if found:
                arguments[parameter] = value
            elif declaration.get("required", True):
                raise ToolException(_missing(tool.name, declaration))
        return await inner(runtime=runtime, **arguments)

    # `metadata` is carried so the tool's `_meta` survives binding — a UI reads
    # its `ui://` view URI from there.
    return StructuredTool(
        name=tool.name,
        description=tool.description,
        args_schema=_prune(tool.args_schema, parameters),
        coroutine=call,
        response_format="content_and_artifact",
        metadata=tool.metadata,
        handle_tool_error=tool.handle_tool_error,
    )


def bind_all_injected(tools: list[BaseTool]) -> list[BaseTool]:
    """Apply :func:`bind_injected` across a whole toolset load.

    Resolves satisfiability once over the full set, so a ``modelFallback``
    parameter with no publisher connected degrades to model-supplied rather
    than to a tool that always raises.
    """
    published = produced(tools)
    return [bind_injected(tool, published) for tool in tools]
