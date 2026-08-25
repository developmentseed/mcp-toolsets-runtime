"""Move large values out of tool returns and into session state.

See :mod:`mcp_state.state` for the namespace the captured keys land in.

A tool's structured return reaches us as MCP ``structuredContent`` on the
message artifact. The tool message is rewritten to short text plus a
``[state updated: …]`` breadcrumb, and the payload moves to ``tool_state`` — so
a large value is *available* for the rest of the session without ever being
*in* the transcript.

Two ways a field gets captured:

**Declared.** The server said so, via :func:`publications` reading its
``_meta``. Every ``ToolResult`` data key is one; the declaration names the
qualified key the field lands under, which is what keeps two toolsets'
identically-named fields from overwriting each other.

**By size.** With ``capture_undeclared`` set, any field whose serialised form
exceeds it is captured whatever the server said, keyed by the tool that
returned it. This is what lets an unmodified third-party MCP server take part:
it declares nothing, and its large values still land somewhere a later tool
can be pointed at.

Declared capture wins where both apply — the server named the key.

Secret-shaped field names are refused either way. That is a backstop against a
toolset publishing something it should not, not a defence against a server
that means harm.

**The other direction.** A tool that was *given* a value from ``tool_state``
records a receipt on its artifact (:mod:`mcp_state.receipts`), which a host
reads to show a call as it ran. Nothing is echoed to the model: it wrote the
``@state:<key>`` handle itself. That runs before any of the capture checks, so
a consumer returning nothing
structured still reports what it received.

**What is left on the message.** The captured payload moves to ``tool_state``,
but the artifact is not simply discarded: it keeps the fields that were *not*
captured, plus a ``captured_state`` map naming where each captured field went.
A UI host reassembles the tool's full structured content from the two with
:func:`restore_structured`, which is what lets a ``ui://`` view keep rendering
under capture. The artifact never reaches the model either way — LangChain
sends ``content``, not ``artifact`` — so this costs no context; it exists
because the residue alone is not the result the view was written against.
"""

import json
import re
from collections.abc import Iterable, Mapping
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langgraph.types import Command

from mcp_runtime.declarations import PRODUCES_META_KEY, qualified
from mcp_state.handles import handle_key, is_handle
from mcp_state.receipts import (
    INJECTED_ARTIFACT_KEY,
    Receipt,
    receipts_of,
)
from mcp_state.state import MODEL_AUTHORED, TOOL_STATE_KEY, AgentState, StateEntry

MESSAGE_KEY = "message"

#: Default size, in serialised bytes, above which an undeclared field is
#: captured rather than left in the transcript. Roughly a screenful of JSON:
#: small enough that anything a model would struggle to reproduce is caught,
#: large enough that ordinary result objects are left alone.
DEFAULT_CAPTURE_BYTES = 2048

#: Field names that look like secrets are never stored, whatever a server
#: declares. A toolset should not be publishing these at all; this is the
#: backstop for when one does.
BLOCKED_KEY_PATTERN = re.compile(
    r"secret|token|password|passwd|credential|api[_-]?key|access[_-]?key"
    r"|private[_-]?key|authorization",
    re.IGNORECASE,
)

#: ``{tool name: {field: {"stateKey": ...}}}``
Published = dict[str, dict[str, dict[str, Any]]]

#: Metadata key naming the MCP server a tool was loaded from. Stamped by the
#: host at load (``langchain_mcp_adapters`` takes a ``server_name`` and puts it
#: nowhere on the tool), and read here so an undeclared capture is keyed the
#: same three-part way a declared one is.
SERVER_METADATA_KEY = "mcp_toolsets_server"


def call_inputs(arguments: Mapping[str, Any]) -> dict[str, str]:
    """Where each argument of one call came from.

    A handle is a reference the model wrote to a value some tool produced, so
    it resolves to that key; everything else the model authored this turn.
    Exact, and cheap: no value is compared, inspected or stored — only the
    argument's name and, where there is one, the key it pointed at.

    This is the *input* side deliberately. A tool owns what it returns, and
    the client is in no position to decide whether a returned value was really
    derived or merely echoed back — an equality test would catch the echo and
    miss every transformation, producing a label that is sometimes right with
    no way to tell which time. What a call was handed needs no such judgement.
    """
    return {
        name: handle_key(value) if is_handle(value) else MODEL_AUTHORED
        for name, value in arguments.items()
    }


def with_server_name(tool: BaseTool, server: str) -> BaseTool:
    """``tool`` with the server it came from recorded on its metadata."""
    return tool.model_copy(
        update={"metadata": {**(tool.metadata or {}), SERVER_METADATA_KEY: server}}
    )


def owners(tools: list[BaseTool]) -> dict[str, str]:
    """``{tool name: server}`` for the tools a host stamped at load."""
    found = {}
    for tool in tools:
        server = (getattr(tool, "metadata", None) or {}).get(SERVER_METADATA_KEY)
        if isinstance(server, str) and server:
            found[tool.name] = server
    return found


#: Artifact key under which a captured message records ``{field: stateKey}``
#: for every field that moved to ``tool_state``. Read it with
#: :func:`restore_structured` rather than by hand.
CAPTURED_ARTIFACT_KEY = "captured_state"


def publications(tools: list[BaseTool]) -> Published:
    """What each tool declares it publishes, from its server ``_meta``.

    ``langchain_mcp_adapters`` preserves the MCP tool's ``_meta`` onto the
    converted LangChain tool, which is what makes the server-side declaration
    reachable here. Tools declaring nothing are simply absent.
    """
    found: Published = {}
    for tool in tools:
        meta = (getattr(tool, "metadata", None) or {}).get("_meta") or {}
        declarations = meta.get(PRODUCES_META_KEY)
        if not declarations:
            continue
        found[tool.name] = {
            declaration["field"]: declaration
            for declaration in declarations
            if isinstance(declaration, dict) and declaration.get("field")
        }
    return found


def state_keys(published: Published) -> frozenset[str]:
    """Every qualified key any connected tool declares it may publish.

    Undeclared captures are absent — nothing knows their keys until they
    happen, which is why this is *not* an allowlist for reading. Pass it to
    :func:`mcp_state.inspect.make_inspect_state` and a read that misses can
    tell "declared, not published yet" from "no such key"; nothing is hidden
    either way.
    """
    return frozenset(
        declaration["stateKey"]
        for fields in published.values()
        for declaration in fields.values()
        if declaration.get("stateKey")
    )


def _from_artifact(artifact: Any) -> dict[str, Any] | None:
    """The MCP ``structured_content`` dict from a tool message artifact, if any."""
    if not isinstance(artifact, dict):
        return None
    content = artifact.get("structured_content")
    return content if isinstance(content, dict) else None


def _size(value: Any) -> int:
    """The serialised size of a value, or 0 if it will not serialise."""
    try:
        return len(json.dumps(value))
    except (TypeError, ValueError):
        return 0


def _breadcrumb(keys: list[str]) -> str:
    """The note naming what a tool just stored, and the two ways to use it.

    The two clauses are scoped apart deliberately. Written as one sentence
    about "the key", ``@state:`` — the more distinctive token of the two —
    generalises into *how you name session state*, and models then write it
    wherever a key belongs: as ``inspect_state``'s argument, or on a plain
    string parameter that never accepted a handle. So the read takes the bare
    key, and the handle is named as belonging to a parameter rather than to
    the key.
    """
    return (
        f"[state updated: {', '.join(keys)} — "
        "pass the bare key to inspect_state to read one; pass @state:<key> "
        "only to a tool parameter whose schema accepts it]"
    )


class StateCaptureMiddleware(AgentMiddleware[AgentState]):
    """Keep large tool output in session state rather than in the transcript.

    A tool opts in by returning ``{"message": <text for the model>, **data}``;
    each declared data key lands in ``tool_state`` under the qualified key its
    server declared. With ``capture_undeclared`` set, large fields are captured
    from any tool at all, including servers that declare nothing.

    Declaring :class:`~mcp_state.state.AgentState` as ``state_schema`` is what
    puts the ``tool_state`` channel on the graph, so adding this middleware is
    enough on its own — a host need not also pass ``state_schema`` to
    ``create_agent``, and one that passes its own ``AgentState`` subclass is
    merged with rather than overridden.

    Args:
        published: Per-tool declarations, from :func:`publications`.
        capture_undeclared: Size in serialised bytes above which an undeclared
            field is captured anyway. ``None`` disables it, leaving capture
            exactly as declared.
        owners: ``{tool name: server}``, from :func:`owners`. Only undeclared
            captures need it, and only to key them the same way a declared one
            is keyed.
    """

    state_schema = AgentState

    def __init__(
        self,
        published: Published | None = None,
        capture_undeclared: int | None = DEFAULT_CAPTURE_BYTES,
        owners: dict[str, str] | None = None,
    ) -> None:
        super().__init__()
        self._published = published or {}
        self._capture_undeclared = capture_undeclared
        self._owners = owners or {}

    def _updates(
        self, tool_name: str, payload: dict[str, Any], arguments: Mapping[str, Any]
    ) -> tuple[dict[str, StateEntry], dict[str, str]]:
        """The ``tool_state`` writes one return earns, and where each came from.

        The second element maps each captured field to the key it landed under.
        The caller needs it twice over: to describe what is *left* when no
        ``message`` told it what to say, and to record on the message where a
        UI host can find the value again.

        ``arguments`` are the model's, from the call this answers. Every entry
        one return writes carries the same :func:`call_inputs` record: it
        describes the *call*, not the field, because which output derives from
        which input is the tool's business and it does not say.
        """
        origin = call_inputs(arguments)

        def entry(value: Any) -> StateEntry:
            """One write, carrying where the producing call's arguments came from."""
            written = StateEntry(value=value, tool=tool_name)
            if origin:
                written["inputs"] = origin
            return written

        declarations = self._published.get(tool_name, {})
        threshold = self._capture_undeclared
        updates: dict[str, StateEntry] = {}
        sources: dict[str, str] = {}
        for field, value in payload.items():
            if field == MESSAGE_KEY or BLOCKED_KEY_PATTERN.search(field):
                continue
            if declaration := declarations.get(field):
                updates[declaration["stateKey"]] = entry(value)
                sources[field] = declaration["stateKey"]
            elif threshold is not None and _size(value) >= threshold:
                # A server that declared nothing still gets a three-part key,
                # so the model reads the same shape whatever produced it. The
                # owning server is only known if the host stamped it at load;
                # without that the key is the tool and the field alone.
                key = (
                    qualified(owner, tool_name, field)
                    if (owner := self._owners.get(tool_name))
                    else f"{tool_name}/{field}"
                )
                updates[key] = entry(value)
                sources[field] = key
        return updates, sources

    def _content(self, payload: dict[str, Any], captured: Iterable[str]) -> str:
        """The text the model sees in place of the return.

        A ``ToolResult`` says what it wants said, in ``message``. Any other
        structured return is reduced to the fields small enough to keep, so
        capturing a large one does not silently blank the result.
        """
        if isinstance(message := payload.get(MESSAGE_KEY), str):
            return message
        moved = set(captured)
        kept = {
            field: value
            for field, value in payload.items()
            if field not in moved and field != MESSAGE_KEY
        }
        return json.dumps(kept) if kept else ""

    async def awrap_tool_call(
        self, request: ToolCallRequest, handler: Any
    ) -> ToolMessage | Command[Any]:
        result = await handler(request)
        if not isinstance(result, ToolMessage):
            return result

        # Read before anything else: a tool that received state but returned
        # nothing structured still has a receipt to report, and every path
        # below this can return early.
        received = receipts_of(result.artifact)

        payload = _from_artifact(result.artifact)
        if payload is None:
            return result

        updates, sources = self._updates(
            request.tool_call["name"], payload, request.tool_call.get("args") or {}
        )
        if not updates and not isinstance(payload.get(MESSAGE_KEY), str):
            return result

        written = _breadcrumb(sorted(updates)) if updates else None
        notes = [note for note in (written,) if note]
        captured = result.model_copy(
            update={
                "content": _annotated(self._content(payload, sources), notes),
                "artifact": _residue(payload, sources, received),
            }
        )
        if not updates:
            return captured
        return Command(update={TOOL_STATE_KEY: updates, "messages": [captured]})


def _annotated(content: str, notes: list[str]) -> str:
    """Text content with each note below it, skipping any that are empty."""
    return "\n\n".join(part for part in (content, *notes) if part)


def _residue(
    payload: dict[str, Any],
    sources: dict[str, str],
    received: dict[str, Receipt],
) -> dict[str, Any]:
    """What stays on the message: the uncaptured fields, plus where the rest went.

    Secret-shaped fields are dropped here as well as from state — they are the
    one thing capture refuses to move, and leaving them on the artifact would
    hand them to a UI host that reassembles it. Receipts are carried through
    unchanged, so rewriting the artifact does not lose what state supplied.
    """
    residue: dict[str, Any] = {
        "structured_content": {
            field: value
            for field, value in payload.items()
            if field not in sources and not BLOCKED_KEY_PATTERN.search(field)
        },
        CAPTURED_ARTIFACT_KEY: sources,
    }
    if received:
        residue[INJECTED_ARTIFACT_KEY] = received
    return residue


def restore_structured(
    artifact: Any, tool_state: dict[str, StateEntry] | None
) -> dict[str, Any] | None:
    """A captured tool's full structured content, rebuilt from message + state.

    The inverse of capture, for a UI host: the fields small enough to have
    stayed on the artifact, with every captured field put back from
    ``tool_state`` under its original name. A view written against the tool's
    own return therefore needs no knowledge that capture happened.

    Works unchanged on an *uncaptured* message (no ``captured_state`` map, so
    the artifact's structured content is already whole), which is what lets a
    host call it on every tool result rather than branching. Returns ``None``
    when there is no structured content at all.

    A key that has since been overwritten by a later write resolves to the
    current value, not the one this tool returned. State holds one value per
    key by design; a host wanting the exact turn's payload should snapshot
    what this returns at render time, as ``mcp_agent.host`` does.
    """
    if not isinstance(artifact, dict):
        return None
    content = artifact.get("structured_content")
    restored = dict(content) if isinstance(content, dict) else {}
    captured = artifact.get(CAPTURED_ARTIFACT_KEY)
    if not isinstance(captured, dict):
        return restored or None
    for field, key in captured.items():
        if entry := (tool_state or {}).get(key):
            restored[field] = entry["value"]
    return restored or None
