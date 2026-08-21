"""Fill tool parameters from session state instead of from the model.

Three paths, applied by one function, in order of how little the model has to
do.

**Declared.** A server that tags a parameter with
:class:`mcp_runtime.declarations.Kind` gets the strong form: the parameter is
removed from the schema the model sees, and filled at call time from
``tool_state`` by matching the kind. The model neither generates the value nor
knows the parameter exists — zero tokens, and nothing to hallucinate.

**Undeclared.** Every other structured parameter gains a ``@state:<key>``
handle branch (:mod:`mcp_state.handles`), so the model can point at a stored
value by name rather than reproducing it. About ten tokens, and it needs
nothing from the server — the path an unmodified third-party tool takes.

**Narrowed.** A parameter its server tagged
:class:`~mcp_runtime.declarations.NotAuthored` keeps the handle branch and
loses the other one, so a handle is the only thing it accepts. Between the two
above in what it asks of the server and in what it guarantees: it names no
type, so nothing has to agree with anything, but unlike a plain handle branch
the model cannot decline to use it and write the value out instead.

All three fall out of one mechanism. LangGraph reads ``InjectedState``
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

from collections.abc import Callable, Container, Mapping, Sequence
from typing import Annotated, Any

import jsonschema
from langchain_core.tools import BaseTool, StructuredTool, ToolException
from langgraph.prebuilt import InjectedState

from mcp_runtime.declarations import CONSUMES_META_KEY, NOT_AUTHORED_META_KEY
from mcp_state.handles import (
    available,
    dereference_with_receipts,
    is_handle,
    offer_handles,
    unresolved,
    unresolved_message,
)
from mcp_state.middleware import publishers
from mcp_state.receipts import (
    BY_DECLARATION,
    INJECTED_ARTIFACT_KEY,
    Receipt,
    receipt_for,
)
from mcp_state.state import TOOL_STATE_KEY, StateEntry, entries_of_kind


class StateRefusal(ToolException):
    """A call this binding would not let through — not a failure of the tool.

    Its own type so it can be told apart from whatever the wrapped tool raises,
    which matters because the two want opposite handling. A refusal is
    *addressed to the model*: it names the parameter, what would fill it, and
    which tool publishes that, so the model can fix the call and try again. It
    is therefore delivered as the tool's **result** — a ``ToolMessage`` with
    ``status="error"`` — rather than raised.

    Raising it instead would end the run, and worse: the assistant message
    keeps its ``tool_calls`` while no ``ToolMessage`` answers them, which is a
    transcript most providers reject outright. One refusal would leave the
    thread unusable for every turn after it, and a model batching a publisher
    and its consumer into one step — ordinary behaviour — is enough to cause
    one.
    """


#: What LangChain lets a ``handle_tool_error`` callable return.
Handled = str | Sequence[str | dict[str, Any]]


def _refusals_are_results(
    inherited: bool | str | Callable[[ToolException], Handled] | None,
) -> Callable[[ToolException], Handled]:
    """Handle our refusals; leave the wrapped tool's own errors as they were.

    ``handle_tool_error`` is a property of the whole tool, so switching it on
    for the refusals would also switch it on for the tool underneath, quietly
    changing how *its* failures surface. This reproduces LangChain's own
    semantics for whatever the tool declared, and adds a case in front.
    """

    def handle(error: ToolException) -> Handled:
        if isinstance(error, StateRefusal):
            return str(error)
        if not inherited:
            raise error
        if inherited is True:
            return str(error)
        if isinstance(inherited, str):
            return inherited
        return inherited(error)

    return handle


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


def not_authored_for(tool: BaseTool) -> frozenset[str]:
    """The tool's parameters its server says a model must not write.

    Read from ``_meta`` the same way :func:`declarations_for` reads consumed
    kinds. Independent of that list: a parameter may be one, the other, both or
    neither, and the two say different things.
    """
    meta = (getattr(tool, "metadata", None) or {}).get("_meta") or {}
    found = meta.get(NOT_AUTHORED_META_KEY)
    if not isinstance(found, list):
        return frozenset()
    return frozenset(str(name) for name in found if isinstance(name, str))


def _authored(
    tool_name: str,
    parameter: str,
    tool_state: dict[str, StateEntry] | None,
    *,
    written: bool,
) -> str:
    """What to tell the model about a ``NotAuthored`` parameter it got wrong.

    ``written`` distinguishes the two ways to get it wrong, because the fix
    differs: a literal means the model tried to author the value and should
    point at a stored one instead, while an omission on a required parameter
    usually means nothing has produced one yet.

    Neither message names a tool that would publish the value, because
    :class:`~mcp_runtime.declarations.NotAuthored` names no kind and so gives
    nothing to look one up by. What the model gets instead is the listing of
    what *is* in state, on top of the tool descriptions it already holds.
    """
    lead = (
        f"{tool_name} was not called. {parameter!r} takes a value that already "
        "exists in this session; you cannot write one."
    )
    if written:
        lead = (
            f"{tool_name} was not called. {parameter!r} was given a value you "
            "wrote. It takes a reference to a value some tool already produced."
        )
    listing = available(tool_state)
    if not listing:
        return (
            f"{lead} Nothing has been published to session state yet, so run "
            "the tool that produces this first."
        )
    return "\n".join(
        [f"{lead} Pass @state:<key> naming one of:"] + [f"  {line}" for line in listing]
    )


#: How a ``$ref`` into the schema's own definitions is written.
DEFS_REF_PREFIX = "#/$defs/"


def _refs(node: Any) -> set[str]:
    """Every ``$ref`` target under ``node``, not descending into ``$defs``."""
    if isinstance(node, dict):
        found = {node["$ref"]} if isinstance(node.get("$ref"), str) else set()
        for key, value in node.items():
            if key != "$defs":
                found |= _refs(value)
        return found
    if isinstance(node, list):
        return {ref for item in node for ref in _refs(item)}
    return set()


def _prune_defs(args_schema: Any) -> Any:
    """The schema with definitions nothing references removed.

    :func:`_prune` and a handle-only rewrite both delete the only ``$ref`` to a
    definition without touching ``$defs``, which then travels to the model
    describing a type no parameter mentions — the whole cost of a richly typed
    parameter, with none of its benefit. Reachability is followed through
    ``$defs`` themselves, so a definition kept alive only by another kept one
    survives.

    Returns the original object when nothing is unreachable, so a caller can
    still tell by identity that the schema is untouched.
    """
    if not isinstance(args_schema, dict):
        return args_schema
    defs = args_schema.get("$defs")
    if not isinstance(defs, dict) or not defs:
        return args_schema

    def names(refs: set[str]) -> set[str]:
        return {
            ref[len(DEFS_REF_PREFIX) :]
            for ref in refs
            if ref.startswith(DEFS_REF_PREFIX)
        }

    outside = {key: value for key, value in args_schema.items() if key != "$defs"}
    frontier = names(_refs(outside))
    reachable: set[str] = set()
    while frontier:
        name = frontier.pop()
        if name in reachable or name not in defs:
            continue
        reachable.add(name)
        frontier |= names(_refs(defs[name]))

    if len(reachable) == len(defs):
        return args_schema
    pruned = dict(args_schema)
    kept = {name: schema for name, schema in defs.items() if name in reachable}
    if kept:
        pruned["$defs"] = kept
    else:
        pruned.pop("$defs", None)
    return pruned


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


def _required(args_schema: Any) -> frozenset[str]:
    """The parameters a server's own schema marks required."""
    if not isinstance(args_schema, dict):
        return frozenset()
    required = args_schema.get("required")
    if not isinstance(required, list):
        return frozenset()
    return frozenset(name for name in required if isinstance(name, str))


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
) -> tuple[str, StateEntry] | None:
    """Find the state entry satisfying one declaration.

    Returns the key it is stored under and the entry itself, or ``None``. The
    most recently published entry of the declared kind that also validates
    against ``schema`` is used, so a stale or foreign-dialect value is passed
    over rather than injected.

    The key and the entry both come back because the caller needs more than
    the value: the entry carries the kind and the publishing tool, which is
    what a receipt (:mod:`mcp_state.receipts`) records.
    """
    kind = declaration.get("kind")
    if not kind:
        return None
    for key, entry in entries_of_kind(tool_state, kind):
        if _validates(entry.get("value"), schema):
            return key, entry
    return None


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


def _with_receipts(
    result: Any, receipts: dict[str, Receipt], response_format: str
) -> Any:
    """The tool's return, with a record of what session state supplied.

    Recorded on the artifact rather than in the content, because the capture
    middleware rewrites content from the structured payload and would drop
    anything written there. A ``content``-only tool has no artifact to carry
    it, so its receipts go unrecorded rather than the wrapper changing the
    return shape the tool declared.
    """
    if not receipts or response_format != "content_and_artifact":
        return result
    if not (isinstance(result, tuple) and len(result) == 2):
        return result
    content, artifact = result
    if artifact is not None and not isinstance(artifact, dict):
        return result
    return content, {**(artifact or {}), INJECTED_ARTIFACT_KEY: receipts}


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

    A parameter its server tagged
    :class:`~mcp_runtime.declarations.NotAuthored` is narrowed instead: it keeps
    its place in the schema but accepts only a handle, and a call that writes a
    literal into one — or omits a required one — is refused before it reaches
    the server. Where a parameter carries both markers the declaration wins,
    since it removes the parameter from the schema entirely.

    ``published`` maps each kind the connected tools publish to the tools that
    publish it (see :func:`mcp_state.middleware.publishers`). Without it every
    declaration is assumed satisfiable and an unfillable parameter cannot name
    what would fill it; :func:`bind_all_injected` supplies it.

    A tool with no declarations, no narrowing and no structured parameters is
    returned unchanged, so this is safe to map over every tool from every
    server.
    """
    declarations = _bindable(tool, published)
    declared = {item["parameter"] for item in declarations}
    # A declared parameter wins where a tool tags one both ways: it leaves the
    # schema altogether, and narrowing something the model cannot see is noise.
    not_authored = not_authored_for(tool) - declared
    args_schema = _prune_defs(
        offer_handles(
            _prune(tool.args_schema, declared), frozenset(declared), not_authored
        )
    )
    if not declarations and not not_authored and args_schema is tool.args_schema:
        return tool
    required_not_authored = not_authored & _required(tool.args_schema)

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
    response_format = tool.response_format

    async def call(
        injected_state: Annotated[
            dict[str, StateEntry] | None, InjectedState(TOOL_STATE_KEY)
        ] = None,
        runtime: Any = None,
        **arguments: Any,
    ) -> Any:
        # Checked before anything is substituted, because that is the only
        # point a literal is still distinguishable from a resolved handle. The
        # narrowed schema should have prevented one, but a schema is a request
        # to a model rather than a guarantee from it.
        for parameter in sorted(not_authored):
            if parameter not in arguments:
                if parameter in required_not_authored:
                    raise StateRefusal(
                        _authored(tool.name, parameter, injected_state, written=False)
                    )
            elif not is_handle(arguments[parameter]):
                raise StateRefusal(
                    _authored(tool.name, parameter, injected_state, written=True)
                )
        arguments, receipts = dereference_with_receipts(arguments, injected_state)
        for declaration in declarations:
            parameter = declaration["parameter"]
            if parameter in arguments:
                continue  # an explicit value wins; never override a caller
            found = resolve(declaration, injected_state, schemas[parameter])
            if found is not None:
                key, entry = found
                arguments[parameter] = entry.get("value")
                receipts[parameter] = receipt_for(key, entry, BY_DECLARATION)
            elif declaration.get("required", True):
                raise StateRefusal(
                    _missing(tool.name, declaration, producers[parameter])
                )
        # Checked after both paths have filled what they can, so what is left
        # is genuinely unresolvable rather than merely not yet resolved.
        if leftover := unresolved(arguments):
            raise StateRefusal(unresolved_message(tool.name, leftover, injected_state))
        result = await inner(runtime=runtime, **arguments)
        return _with_receipts(result, receipts, response_format)

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
        # Not the wrapped tool's flag verbatim: a refusal has to reach the
        # model as a result whatever the tool would have done with its own
        # errors. See :func:`_refusals_are_results`.
        handle_tool_error=_refusals_are_results(tool.handle_tool_error),
    )


def bind_all_injected(tools: list[BaseTool]) -> list[BaseTool]:
    """Apply :func:`bind_injected` across a whole toolset load.

    Resolves what is published once over the full set, so a model-generatable
    parameter with no publisher connected degrades to model-supplied rather
    than to a tool that always raises.
    """
    published = publishers(tools)
    return [bind_injected(tool, published) for tool in tools]
