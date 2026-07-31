"""Chainlit chat UI over the same agent as the ``mcp-agent`` CLI.

Run locally with ``uv run mcp-agent-web``, or deploy it (see ``charts/mcp-chat``)
as a public chat over the toolsets behind an index.

**Bring your own model.** Unlike the CLI, the web app holds *no* provider key:
each user supplies a ``provider:model`` and their API key in the chat's
settings panel, so a hosted deployment stores no secret to leak or meter. Env
values (``PROVIDER_MODEL`` / ``PROVIDER_API_KEY`` in the environment or .env)
only *pre-fill* those fields for local single-user use. ``MCP_URL`` (index root
or single MCP endpoint, default ``http://localhost:8000/mcp``) and
``CHAINLIT_PORT`` (default 8080) come from :class:`WebSettings`.

Per-user credentials work the same way: every credential header the connected
toolsets advertise gets a field in the same panel; values ride HTTP headers on
the MCP calls — only to the toolsets that declared them — so the model and the
chat history never see them. The agent is built per session once a model + key
are provided; credentials apply per message via ``user_credentials``, the same
mechanism a public multi-user API would use with one shared agent.

Tool views render in the right-hand side panel (``ElementSidebar``) rather than
inline in the transcript, so the conversation stays one readable column and the
visualizations get the room they need — see :func:`render_views`.

Session state (``MCP_AGENT_STATE``, on by default — see :mod:`mcp_agent.main`)
keeps large tool values out of the model's context. Views are unaffected: a
captured value is reassembled for rendering from ``tool_state``, so the panel
shows the payload in full while the model only ever saw the summary.
"""

import os
import sys
import uuid
from pathlib import Path
from typing import Any

import chainlit as cl
from chainlit.input_widget import InputWidget, TextInput
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from mcp_agent.main import (
    DEFAULT_ELEMENTS_DIR,
    HOST_ELEMENTS,
    build_agent,
    connect_error_hint,
    fetch_connections,
    first_leaf,
    run_turn,
    user_credentials,
    with_credential_support,
)
from mcp_state import restore_structured
from mcp_state.state import StateEntry

# The _meta convention a UI-capable host reads (mcp-ui / Apps-SDK style):
# tool.metadata["_meta"]["ui"]["resourceUri"] names a ui:// resource to render.
VIEW_META_KEY = "ui"

# Settings-panel field ids for the user-supplied model. Kept out of the
# environment so a hosted deployment holds no provider secret (BYOM).
MODEL_FIELD = "PROVIDER_MODEL"
API_KEY_FIELD = "PROVIDER_API_KEY"

# Title of the side panel the views render in.
VIEW_PANEL_TITLE = "Visualizations"

# Per-turn snapshots of view props ({"html", "data"}) so a past turn's panel can
# be recalled from its reply's "Show in panel" action — the panel only ever
# holds the latest turn's views, so an overwritten one can't be rebuilt
# otherwise. Bounded because a snapshot can hold a large image data URI.
_VIEW_HISTORY = "view_history"
_MAX_VIEW_HISTORY = 20
_SHOW_VIEWS_ACTION = "show_views"


class WebSettings(BaseSettings):
    """Web-UI configuration — everything the deployment could hold is optional.

    Unlike the CLI's ``AgentSettings``, the model and its key are *not* required
    here: the hosted chat is bring-your-own-model, so each user supplies
    ``provider:model`` and their API key in the settings panel and the server
    stores no provider secret. Env values, when present (local dev / .env), only
    pre-fill those fields.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    provider_model: str | None = None
    provider_api_key: SecretStr | None = None
    mcp_url: str = "http://localhost:8000/mcp"
    chainlit_port: int = Field(default=8080, ge=1, le=65535)


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


@cl.on_chat_start
async def start() -> None:
    settings = WebSettings()
    cl.user_session.set("mcp_url", settings.mcp_url)
    _, required = await fetch_connections(settings.mcp_url)
    cl.user_session.set("required", required)
    header_names = sorted(
        {name for names in (required or {}).values() for name in names}
    )
    fields: list[InputWidget] = [
        TextInput(
            id=MODEL_FIELD,
            label="Model — provider:model",
            placeholder="e.g. anthropic:claude-3-5-haiku-latest",
            initial=settings.provider_model or "",
        ),
        TextInput(
            id=API_KEY_FIELD,
            label="Model API key (kept in this session only)",
            initial="",
        ),
        *[TextInput(id=name, label=name) for name in header_names],
    ]
    await cl.ChatSettings(fields).send()

    prefilled_key = (
        settings.provider_api_key.get_secret_value()
        if settings.provider_api_key
        else ""
    )
    if settings.provider_model and prefilled_key:
        await ensure_agent(settings.provider_model, prefilled_key)
    else:
        await cl.Message(
            "**Bring your own model.** Open ⚙ settings (by the message box) and "
            "set a **Model** (`provider:model`) and your **API key** to connect. "
            "Your key stays in this browser session — it is never sent to the "
            "model, written to logs, or stored on the server."
        ).send()


async def ensure_agent(model: str, api_key: str) -> None:
    """(Re)build the session's agent for its MCP url with the user's model.

    Called when a model + key arrive (from the settings panel, or env pre-fill).
    Stores the agent and view bundles on the session and greets with what
    connected; connection failures surface in the chat, not the logs. Existing
    chat history is preserved across a model switch.
    """
    mcp_url: str = cl.user_session.get("mcp_url") or "http://localhost:8000/mcp"
    required: dict[str, list[str]] | None = cl.user_session.get("required")
    try:
        agent, connections, tools, withheld = await build_agent(
            mcp_url, model, SecretStr(api_key)
        )
    except Exception as error:  # noqa: BLE001 - surface in the UI, not the logs
        await cl.Message(
            f"Could not connect with model `{model}`: {first_leaf(error)}."
            f"{connect_error_hint(mcp_url)}"
        ).send()
        return
    cl.user_session.set("agent", agent)
    cl.user_session.set("provider_model", model)
    cl.user_session.set("messages", cl.user_session.get("messages") or [])
    cl.user_session.set("view_html", await view_bundles(connections, required))
    cl.user_session.set("tools_by_name", {tool.name: tool for tool in tools})
    await cl.Message(
        f"Connected to **{len(connections)}** server(s) "
        f"({', '.join(connections)}) with **{len(tools)}** tools, using "
        f"**{model}**: {', '.join(tool.name for tool in tools)}."
    ).send()
    needing = ", ".join(
        f"{toolset} ({', '.join(names)})"
        for toolset, names in sorted((required or {}).items())
        if names
    )
    if needing:
        await cl.Message(
            f"Some tools act on your behalf and need credentials: {needing}. "
            "Set them in the settings panel (⚙ by the message box); each is "
            "sent only to the toolset that declares it, never to the model."
        ).send()
    if withheld:
        listed = "\n".join(f"- `{item}`" for item in withheld)
        await cl.Message(
            f"**{len(withheld)} tool(s) are not available** — each needs a value "
            "no connected toolset publishes, and its own author said a model "
            f"must not invent one:\n\n{listed}\n\nConnecting the toolset that "
            "produces it makes them available again."
        ).send()


@cl.on_settings_update
async def apply_settings(values: dict[str, Any]) -> None:
    """Apply the settings panel: (re)build on a model change, else set credentials.

    The model + key connect (or re-connect) the agent; every other field is a
    toolset credential header, stored for the next turn. A credentials-only
    edit must not rebuild — that would reconnect the MCP servers needlessly — so
    the agent is rebuilt only when the model changes or none exists yet.
    """
    model = str(values.get(MODEL_FIELD) or "").strip()
    api_key = str(values.get(API_KEY_FIELD) or "").strip()
    headers = {
        name: value.strip()
        for name, value in values.items()
        if name not in (MODEL_FIELD, API_KEY_FIELD)
        and isinstance(value, str)
        and value.strip()
    }
    cl.user_session.set("credentials", headers or None)

    have_agent = cl.user_session.get("agent") is not None
    model_changed = model != (cl.user_session.get("provider_model") or "")
    if model and api_key and (not have_agent or model_changed):
        await ensure_agent(model, api_key)
    elif not have_agent:
        await cl.Message(
            "Set both a **Model** and an **API key** in ⚙ settings to connect."
        ).send()
    else:
        await cl.Message(
            "Credentials updated — your next tool calls will use them."
        ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    agent = cl.user_session.get("agent")
    if agent is None:
        await cl.Message(
            "Not connected yet — set your **Model** and **API key** in "
            "⚙ settings first (or check MCP_URL and reload)."
        ).send()
        return
    messages: list[BaseMessage] = cl.user_session.get("messages") or []
    credentials: dict[str, str] | None = cl.user_session.get("credentials")
    # Round-tripped rather than rebuilt: the agent is invoked once per turn, so
    # a value captured on an earlier turn only stays reachable by being handed
    # back in. None until something is captured (or forever, with state off).
    tool_state: dict[str, StateEntry] | None = cl.user_session.get("tool_state")
    try:
        with user_credentials(credentials):
            history, new_messages, tool_state = await run_turn(
                agent, messages, message.content, tool_state
            )
    except Exception as error:  # noqa: BLE001 - surface in the UI, keep chatting
        await cl.Message(f"Error: {error}").send()
        return
    cl.user_session.set("messages", history)
    cl.user_session.set("tool_state", tool_state)

    tool_messages = [msg for msg in new_messages if isinstance(msg, ToolMessage)]
    tool_outputs = {msg.tool_call_id: msg for msg in tool_messages}
    for msg in new_messages:
        for call in getattr(msg, "tool_calls", None) or []:
            async with cl.Step(name=call["name"]) as step:
                step.input = call["args"]
                result = tool_outputs.get(call["id"])
                step.output = str(result.content) if result else ""

    # One id per turn, keying this turn's view snapshot so the reply's "Show in
    # panel" action can bring those visualizations back after later turns have
    # replaced the panel.
    turn_id = uuid.uuid4().hex
    produced_views = await render_views(new_messages, tool_outputs, turn_id, tool_state)
    actions = [_show_views_action(turn_id)] if produced_views else []
    await cl.Message(str(history[-1].content), actions=actions).send()


def _tool_name(message: ToolMessage, new_messages: list[BaseMessage]) -> str | None:
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
        tool = tools_by_name.get(_tool_name(message, new_messages) or "")
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
    """Record a turn's views, dropping the oldest beyond ``_MAX_VIEW_HISTORY``."""
    history[turn_id] = views
    for stale in list(history)[:-_MAX_VIEW_HISTORY]:  # bound memory, keep newest
        del history[stale]
    return history


async def show_views(views: list[dict[str, Any]]) -> None:
    """Replace the side panel with ``McpView`` elements built from ``views``.

    Fresh ``CustomElement`` instances are built on each call so recalling a past
    turn mounts cleanly rather than reusing a spent element id.
    """
    elements = [cl.CustomElement(name="McpView", props=view) for view in views]
    await cl.ElementSidebar.set_title(VIEW_PANEL_TITLE)
    await cl.ElementSidebar.set_elements(elements)


async def render_views(
    new_messages: list[BaseMessage],
    tool_outputs: dict[str, ToolMessage],
    turn_id: str,
    tool_state: dict[str, StateEntry] | None = None,
) -> bool:
    """Render a UI view for each tool result whose tool declares one.

    A tool's ``_meta`` names a ``ui://`` resource; its HTML (read once at
    connect) goes into a sandboxed iframe, fed the tool's ``structuredContent``
    (from the ToolMessage's ``artifact``, plus anything session state captured
    out of it). Interactions inside come back as user messages via
    ``sendUserMessage``, so the loop advances the chat.

    Views go to the right-hand ``ElementSidebar``, not inline in the chat, so
    the transcript stays one readable column. The panel shows only the *current*
    turn's views: a turn that produces any view replaces the panel's contents,
    so it never accumulates, while a turn with none leaves the panel as-is (a
    text follow-up doesn't wipe the last map). Each turn's props are snapshotted
    first, so its reply's "Show in panel" action can recall them; the return
    value tells the caller whether to offer that action.
    """
    view_html: dict[str, str] = cl.user_session.get("view_html") or {}
    tools_by_name: dict[str, BaseTool] = cl.user_session.get("tools_by_name") or {}
    if not view_html:
        return False
    views = view_props(new_messages, tool_outputs, view_html, tools_by_name, tool_state)
    if not views:
        return False
    cl.user_session.set(
        _VIEW_HISTORY,
        remember_views(cl.user_session.get(_VIEW_HISTORY) or {}, turn_id, views),
    )
    await show_views(views)
    return True


def _show_views_action(turn_id: str) -> cl.Action:
    """A reply button that recalls that turn's views into the panel."""
    return cl.Action(
        name=_SHOW_VIEWS_ACTION,
        payload={"turn": turn_id},
        label="Show in panel",
        icon="panel-right",
        tooltip="Bring this turn's visualizations back to the panel",
    )


@cl.action_callback(_SHOW_VIEWS_ACTION)
async def on_show_views(action: cl.Action) -> None:
    """Repopulate the panel from a past turn's snapshot (or note it's gone)."""
    history: dict[str, list[dict[str, Any]]] = cl.user_session.get(_VIEW_HISTORY) or {}
    views = history.get(str(action.payload.get("turn")))
    if views:
        await show_views(views)
    else:
        await cl.Message("Those visualizations are no longer available.").send()


def main() -> None:
    """Console entry point (``mcp-agent-web``).

    No provider key is required to start: the model is bring-your-own, set per
    session in the UI. Only ``CHAINLIT_PORT`` is read here at boot.
    """
    from chainlit.cli import run_chainlit

    settings = WebSettings()

    # Chainlit loads the "McpView" element from ./public/elements at render time;
    # nudge (don't fail) if it's missing so tool views don't silently no-op.
    missing = [n for n in HOST_ELEMENTS if not (DEFAULT_ELEMENTS_DIR / n).is_file()]
    if missing:
        print(
            f"warning: Chainlit host element(s) {missing} not found under "
            f"{DEFAULT_ELEMENTS_DIR}/; tool views will not render. Install with: "
            f"mcp-agent install-elements",
            file=sys.stderr,
        )

    os.environ["CHAINLIT_PORT"] = str(settings.chainlit_port)
    run_chainlit(str(Path(__file__).resolve()))
