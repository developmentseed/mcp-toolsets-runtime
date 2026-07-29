"""Capture the state updates a tool's ``ToolResult`` declares.

See :mod:`mcp_state.state` for the namespace the captured keys land in.

A tool's dict return reaches us as MCP ``structuredContent`` on the message
artifact (:mod:`mcp_runtime.fastmcp_output` derives the output schema, so
every toolset served by this runtime emits it). The tool message is rewritten
to the ``message`` plus a ``[state updated: …]`` breadcrumb, and the payload
moves to ``tool_state`` — so a large value is *available* for the rest of the
session without ever being *in* the transcript.

Which fields may be captured is decided by the *server*, per tool, via
:func:`publications` reading its ``_meta``. A declaration carries more than
permission: it names the qualified key the field lands under and the kind it
publishes — both what injection resolves on, and what keeps two toolsets'
identically-named fields from overwriting each other.

Capture is therefore opt-in twice over: a server that declares nothing
publishes nothing, so third-party MCP servers pass through untouched even if
their returns happen to look like a ``ToolResult``.
"""

import re
from typing import Any, NamedTuple

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langgraph.types import Command

from mcp_runtime.injected import PRODUCES_META_KEY
from mcp_state.state import TOOL_STATE_KEY, StateEntry

MESSAGE_KEY = "message"

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


class PublishedTargets(NamedTuple):
    """What connected tools publish, in the two forms a declaration asks for."""

    kinds: frozenset[str]
    keys: frozenset[str]


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
    """Every qualified key any connected tool may publish.

    Pass to :func:`mcp_state.inspect.make_inspect_state` so the model can only
    ever name a key some tool actually declares.
    """
    return frozenset(
        declaration["stateKey"]
        for fields in published.values()
        for declaration in fields.values()
        if declaration.get("stateKey")
    )


def published_targets(tools: list[BaseTool]) -> PublishedTargets:
    """The kinds and the qualified keys the connected tools publish."""
    declarations = [
        declaration
        for fields in publications(tools).values()
        for declaration in fields.values()
    ]
    return PublishedTargets(
        kinds=frozenset(
            declaration["kind"]
            for declaration in declarations
            if declaration.get("kind")
        ),
        keys=frozenset(
            declaration["stateKey"]
            for declaration in declarations
            if declaration.get("stateKey")
        ),
    )


def _from_artifact(artifact: Any) -> dict[str, Any] | None:
    """The MCP ``structured_content`` dict from a tool message artifact, if any."""
    if not isinstance(artifact, dict):
        return None
    content = artifact.get("structured_content")
    return content if isinstance(content, dict) else None


def _breadcrumb(keys: list[str]) -> str:
    return (
        f"[state updated: {', '.join(keys)} — "
        "inspect_state(key) reads a value; pattern=/path= narrow it]"
    )


class StateCaptureMiddleware(AgentMiddleware):
    """Apply tool-declared state updates and keep bulky payloads out of chat.

    A tool opts in by returning ``{"message": <text for the model>, **data}``;
    each data key lands in ``tool_state`` under the qualified key its server
    declared, provided it is not secret-shaped. The tool message becomes the
    ``message`` plus a breadcrumb naming the stored keys.
    """

    def __init__(self, published: Published | None = None) -> None:
        super().__init__()
        self._published = published or {}

    def _updates(
        self, tool_name: str, payload: dict[str, Any]
    ) -> dict[str, StateEntry]:
        """The ``tool_state`` writes one return earns, keyed by qualified key."""
        declarations = self._published.get(tool_name, {})
        updates: dict[str, StateEntry] = {}
        for field, value in payload.items():
            if field == MESSAGE_KEY or BLOCKED_KEY_PATTERN.search(field):
                continue
            declaration = declarations.get(field)
            if declaration is None:
                continue
            updates[declaration["stateKey"]] = StateEntry(
                value=value, kind=declaration.get("kind"), tool=tool_name
            )
        return updates

    async def awrap_tool_call(
        self, request: ToolCallRequest, handler: Any
    ) -> ToolMessage | Command[Any]:
        result = await handler(request)
        if not isinstance(result, ToolMessage):
            return result
        payload = _from_artifact(result.artifact)
        if payload is None or not isinstance(payload.get(MESSAGE_KEY), str):
            return result

        updates = self._updates(request.tool_call["name"], payload)
        content = payload[MESSAGE_KEY]
        if updates:
            content = f"{content}\n\n{_breadcrumb(sorted(updates))}"
        captured = result.model_copy(update={"content": content, "artifact": None})
        if not updates:
            return captured
        return Command(update={TOOL_STATE_KEY: updates, "messages": [captured]})
