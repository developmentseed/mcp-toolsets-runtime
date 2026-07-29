"""Move bulk values out of tool returns and into session state.

See :mod:`mcp_state.state` for the namespace the captured keys land in.

A tool's structured return reaches us as MCP ``structuredContent`` on the
message artifact. The tool message is rewritten to short text plus a
``[state updated: …]`` breadcrumb, and the payload moves to ``tool_state`` — so
a large value is *available* for the rest of the session without ever being
*in* the transcript.

Two ways a field gets captured:

**Declared.** The server said so, via :func:`publications` reading its
``_meta``. The declaration names the qualified key the field lands under and
the kind it publishes, which is what keeps two toolsets' identically-named
fields from overwriting each other.

**By size.** With ``capture_undeclared`` set, any field whose serialised form
exceeds it is captured whatever the server said, keyed by the tool that
returned it and labelled with whatever :func:`mcp_state.detect.detect_kind`
recognises. This is what lets an unmodified third-party MCP server take part:
it declares nothing, and its large values still land somewhere a later tool
can be pointed at.

Declared capture wins where both apply — a server that named a kind knows
better than a detector.

Secret-shaped field names are refused either way. That is a backstop against a
toolset publishing something it should not, not a defence against a server
that means harm.
"""

import json
import re
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langgraph.types import Command

from mcp_runtime.declarations import PRODUCES_META_KEY, qualified
from mcp_state.detect import detect_kind
from mcp_state.state import TOOL_STATE_KEY, StateEntry

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

#: ``{tool name: {field: {"stateKey": ..., "kind": ...}}}``
Published = dict[str, dict[str, dict[str, Any]]]


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

    Pass to :func:`mcp_state.inspect.make_inspect_state` so the model can only
    ever name a key some tool actually declares. Undeclared captures are not
    included — nothing knows their keys until they happen.
    """
    return frozenset(
        declaration["stateKey"]
        for fields in published.values()
        for declaration in fields.values()
        if declaration.get("stateKey")
    )


def published_kinds(tools: list[BaseTool]) -> frozenset[str]:
    """Every kind the connected tools declare they publish.

    What :func:`mcp_state.injection.bind_all_injected` decides satisfiability
    against. Kinds that only ever arrive by detection are absent, so a
    declared parameter is not assumed satisfiable on the strength of a value
    that may never appear.
    """
    return frozenset(
        declaration["kind"]
        for fields in publications(tools).values()
        for declaration in fields.values()
        if declaration.get("kind")
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
    return (
        f"[state updated: {', '.join(keys)} — "
        "inspect_state(key) reads a value; pass @state:<key> to a tool "
        "parameter to hand it the value directly]"
    )


class StateCaptureMiddleware(AgentMiddleware):
    """Keep bulky tool output in session state rather than in the transcript.

    A tool opts in by returning ``{"message": <text for the model>, **data}``;
    each declared data key lands in ``tool_state`` under the qualified key its
    server declared. With ``capture_undeclared`` set, large fields are captured
    from any tool at all, including servers that declare nothing.

    Args:
        published: Per-tool declarations, from :func:`publications`.
        capture_undeclared: Size in serialised bytes above which an undeclared
            field is captured anyway. ``None`` disables it, leaving capture
            exactly as declared.
    """

    def __init__(
        self,
        published: Published | None = None,
        capture_undeclared: int | None = DEFAULT_CAPTURE_BYTES,
    ) -> None:
        super().__init__()
        self._published = published or {}
        self._capture_undeclared = capture_undeclared

    def _updates(
        self, tool_name: str, payload: dict[str, Any]
    ) -> tuple[dict[str, StateEntry], set[str]]:
        """The ``tool_state`` writes one return earns, and the fields they came from.

        The field names come back too because the caller has to describe what
        is *left* when no ``message`` told it what to say.
        """
        declarations = self._published.get(tool_name, {})
        threshold = self._capture_undeclared
        updates: dict[str, StateEntry] = {}
        sources: set[str] = set()
        for field, value in payload.items():
            if field == MESSAGE_KEY or BLOCKED_KEY_PATTERN.search(field):
                continue
            if declaration := declarations.get(field):
                updates[declaration["stateKey"]] = StateEntry(
                    value=value, kind=declaration.get("kind"), tool=tool_name
                )
                sources.add(field)
            elif threshold is not None and _size(value) >= threshold:
                updates[qualified(tool_name, field)] = StateEntry(
                    value=value, kind=detect_kind(value), tool=tool_name
                )
                sources.add(field)
        return updates, sources

    def _content(self, payload: dict[str, Any], captured: set[str]) -> str:
        """The text the model sees in place of the return.

        A ``ToolResult`` says what it wants said, in ``message``. Any other
        structured return is reduced to the fields small enough to keep, so
        capturing a large one does not silently blank the result.
        """
        if isinstance(message := payload.get(MESSAGE_KEY), str):
            return message
        kept = {
            field: value
            for field, value in payload.items()
            if field not in captured and field != MESSAGE_KEY
        }
        return json.dumps(kept) if kept else ""

    async def awrap_tool_call(
        self, request: ToolCallRequest, handler: Any
    ) -> ToolMessage | Command[Any]:
        result = await handler(request)
        if not isinstance(result, ToolMessage):
            return result
        payload = _from_artifact(result.artifact)
        if payload is None:
            return result

        updates, sources = self._updates(request.tool_call["name"], payload)
        if not updates and not isinstance(payload.get(MESSAGE_KEY), str):
            return result

        content = self._content(payload, sources)
        if updates:
            breadcrumb = _breadcrumb(sorted(updates))
            content = f"{content}\n\n{breadcrumb}" if content else breadcrumb
        captured = result.model_copy(update={"content": content, "artifact": None})
        if not updates:
            return captured
        return Command(update={TOOL_STATE_KEY: updates, "messages": [captured]})
