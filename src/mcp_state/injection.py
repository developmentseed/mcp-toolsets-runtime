"""Give a tool a value the model never wrote, and refuse the calls that cheat.

A tool parameter that could hold a structured value gains a second accepted
form: the string ``@state:<key>``, naming something already in session state
(:mod:`mcp_state.handles`). The model spends about ten tokens on the key
instead of thousands reproducing the value, the value itself never enters the
transcript, and none of it needs anything from the server — the path an
unmodified third-party tool takes.

A server that wants more says so with
:class:`~mcp_runtime.declarations.NotAuthored`, and the parameter is
**narrowed**: it keeps its place in the schema and loses the arm that accepted
a literal, so a handle is the only thing that fits. The difference is that the
model cannot decline the offer and write the value out instead.

Both are one mechanism. LangGraph reads ``InjectedState`` annotations off the
tool's *coroutine* as well as its schema
(``langgraph.prebuilt.tool_node._get_all_injected_args``), so a wrapper
coroutine carrying one gets the whole ``tool_state`` dict handed to it at call
time, while ``args_schema`` — the server's raw JSON Schema, with the handle
branches rewritten into it — is what reaches the model. Keeping it a plain dict
rather than round-tripping through pydantic preserves the server's schema
exactly, which matters now that MCP input schemas may use the whole of JSON
Schema 2020-12.

**Which stored value to use is the model's decision, and no heuristic's.** It
has the conversation; "the area the user just drew" is not something recency
can be relied on to know. What the client does is make the choice cheap to
express, make what is available legible, and refuse a call that got it wrong
in a way the model can act on.
"""

from collections.abc import Callable, Sequence
from typing import Annotated, Any

from langchain_core.tools import BaseTool, StructuredTool, ToolException
from langgraph.prebuilt import InjectedState

from mcp_runtime.declarations import NOT_AUTHORED_META_KEY
from mcp_state.handles import (
    available,
    dereference_with_receipts,
    is_handle,
    offer_handles,
    unresolved,
    unresolved_message,
)
from mcp_state.receipts import INJECTED_ARTIFACT_KEY, Receipt
from mcp_state.state import TOOL_STATE_KEY, StateEntry


class StateRefusal(ToolException):
    """A call this binding would not let through — not a failure of the tool.

    Its own type so it can be told apart from whatever the wrapped tool raises,
    which matters because the two want opposite handling. A refusal is
    *addressed to the model*: it names the parameter, says what was wrong with
    it, and lists what session state actually holds, so the model can fix the
    call and try again. It is therefore delivered as the tool's **result** — a
    ``ToolMessage`` with ``status="error"`` — rather than raised.

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


def not_authored_for(tool: BaseTool) -> frozenset[str]:
    """The tool's parameters its server says a model must not write.

    ``langchain_mcp_adapters`` preserves the MCP tool's ``_meta`` onto the
    converted LangChain tool's ``metadata``, which is what makes a server-side
    declaration reachable here at all.
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

    Neither message names a tool that would publish the value.
    :class:`~mcp_runtime.declarations.NotAuthored` says only that the model may
    not write it, so there is nothing here to look a producer up by. What the
    model gets instead is the listing of what *is* in state, on top of the tool
    descriptions it already holds.
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

    Narrowing a parameter deletes the only ``$ref`` to a definition without
    touching ``$defs``, which then travels to the model describing a type no
    parameter mentions — the whole cost of a richly typed parameter, with none
    of its benefit. Reachability is followed through ``$defs`` themselves, so a
    definition kept alive only by another kept one survives.

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


def _required(args_schema: Any) -> frozenset[str]:
    """The parameters a server's own schema marks required."""
    if not isinstance(args_schema, dict):
        return frozenset()
    required = args_schema.get("required")
    if not isinstance(required, list):
        return frozenset()
    return frozenset(name for name in required if isinstance(name, str))


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


def bind_injected(tool: BaseTool) -> BaseTool:
    """Return ``tool`` with handles offered, and narrowed where its server said.

    A parameter its server tagged
    :class:`~mcp_runtime.declarations.NotAuthored` keeps its place in the
    schema but accepts only a handle, and a call that writes a literal into one
    — or omits a required one — is refused before it reaches the server.

    A tool with no narrowing and no structured parameters is returned
    unchanged, so this is safe to map over every tool from every server.
    """
    not_authored = not_authored_for(tool)
    args_schema = _prune_defs(offer_handles(tool.args_schema, only=not_authored))
    if not not_authored and args_schema is tool.args_schema:
        return tool
    required_not_authored = not_authored & _required(tool.args_schema)

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
    """Apply :func:`bind_injected` across a whole toolset load."""
    return [bind_injected(tool) for tool in tools]
