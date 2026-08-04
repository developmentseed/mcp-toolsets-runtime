"""What a chat host needs to present a turn, with no UI framework in it.

A host embedding this agent has to answer two questions each turn: which tool
results carry a ``ui://`` view and what to feed it (:func:`view_props`, over
:func:`view_bundles` and :func:`view_uri_for`), and what a tool call was
actually given once session state filled in the parameters the model never saw
(:func:`step_input`).

Nothing here imports a UI framework, so it is reachable from a base install —
:mod:`mcp_agent.web` is one host built on it, and its Chainlit dependency is an
optional extra. Importing that module registers Chainlit's ``@cl.on_chat_start``
and friends onto the importing process, which is not something a host wants as a
side effect of reading a tool's ``_meta``.
"""

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from mcp_agent.main import with_credential_support
from mcp_state import Receipt, describe, receipts_of, restore_structured, supplied
from mcp_state.state import StateEntry

# The _meta convention a UI-capable host reads (mcp-ui / Apps-SDK style):
# tool.metadata["_meta"]["ui"]["resourceUri"] names a ui:// resource to render.
VIEW_META_KEY = "ui"

# How many turns of view props :func:`remember_views` keeps. Bounded because a
# snapshot can hold a large image data URI.
MAX_VIEW_HISTORY = 20


async def view_bundles(
    connections: dict[str, Any], required: dict[str, list[str]] | None
) -> dict[str, str]:
    """Read every ``ui://`` view bundle the toolsets serve, as ``{uri: html}``.

    Standard MCP: the runtime registers each view as a ``ui://<toolset>/<view>``
    resource and stamps the owning tool's ``_meta`` with that URI. A tool with
    no view has no such resource, so it renders as text exactly as before.
    """
    client = MultiServerMCPClient(with_credential_support(connections, required))
    try:
        blobs = await client.get_resources()
    except Exception:  # noqa: BLE001 - views are optional; degrade to text
        return {}
    return {
        uri: blob.as_string()
        for blob in blobs
        if (uri := str(blob.metadata.get("uri", ""))).startswith("ui://")
    }


def view_uri_for(tool: BaseTool | None) -> str | None:
    """The ``ui://`` resource a tool declares via its ``_meta``, if any."""
    meta = (getattr(tool, "metadata", None) or {}).get("_meta") or {}
    ui = meta.get(VIEW_META_KEY)
    return ui.get("resourceUri") if isinstance(ui, dict) else None


def tool_name(message: ToolMessage, new_messages: list[BaseMessage]) -> str | None:
    """The name of the tool a ToolMessage answers (its own, or from the call)."""
    if message.name:
        return message.name
    for candidate in new_messages:
        if isinstance(candidate, AIMessage):
            for call in candidate.tool_calls:
                if call["id"] == message.tool_call_id:
                    return call["name"]
    return None


def view_props(
    new_messages: list[BaseMessage],
    tool_outputs: dict[str, ToolMessage],
    view_html: dict[str, str],
    tools_by_name: dict[str, BaseTool],
    tool_state: dict[str, StateEntry] | None = None,
) -> list[dict[str, Any]]:
    """This turn's view props (``{"html", "data"}``), newest first.

    One entry per tool result whose tool declares a ``ui://`` resource that the
    session actually holds HTML for; every other result renders as text alone.

    Each view is fed its tool's *whole* structured content, reassembled by
    :func:`~mcp_state.restore_structured` from what stayed on the message and
    what capture moved into ``tool_state``. A view is written against its
    tool's return, so it must not be able to tell whether the value took the
    long way round — and with state off there is nothing to put back, which is
    the same call with an empty map.
    """
    views: list[dict[str, Any]] = []
    for message in tool_outputs.values():
        tool = tools_by_name.get(tool_name(message, new_messages) or "")
        uri = view_uri_for(tool)
        html = view_html.get(uri) if uri else None
        if not html:
            continue
        data = restore_structured(message.artifact, tool_state)
        views.insert(0, {"html": html, "data": data})
    return views


def remember_views(
    history: dict[str, list[dict[str, Any]]],
    turn_id: str,
    views: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Record a turn's views, dropping the oldest beyond :data:`MAX_VIEW_HISTORY`."""
    history[turn_id] = views
    for stale in list(history)[:-MAX_VIEW_HISTORY]:  # bound memory, keep newest
        del history[stale]
    return history


def _from_state(receipt: Receipt, tool_state: dict[str, StateEntry] | None) -> str:
    """One session-state-supplied parameter, as a line that can be traced back.

    The value itself is deliberately not shown — it is in state precisely
    because it is too big for a transcript, and a step panel is no different.
    Its shape stands in for it.
    """
    parts = [f"← {receipt['key']}", receipt.get("kind") or "untyped"]
    if entry := (tool_state or {}).get(receipt["key"]):
        parts.append(describe(entry.get("value")))
    if tool := receipt.get("tool"):
        parts.append(f"from {tool}")
    return " · ".join(parts)


def step_input(
    arguments: dict[str, Any],
    result: ToolMessage | None,
    tool_state: dict[str, StateEntry] | None,
) -> dict[str, Any]:
    """A tool call's arguments, plus whatever session state filled in.

    A declared parameter is removed from the schema the model sees, so it is
    absent from the arguments the model produced. Showing those alone would
    present the call as having run without the value that decided its result.
    """
    received = supplied(receipts_of(getattr(result, "artifact", None)), arguments)
    if not received:
        return arguments
    return {
        **arguments,
        **{
            parameter: _from_state(receipt, tool_state)
            for parameter, receipt in received.items()
        },
    }
