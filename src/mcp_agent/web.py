"""Chainlit chat UI over the same agent as the ``mcp-agent`` CLI.

Run locally with ``uv run mcp-agent-web``, or deploy it (see ``charts/mcp-chat``)
as a public chat over the toolsets behind an index.

**Bring your own model.** Unlike the CLI, the web app is *configured* with no
provider key: each user supplies a ``provider:model`` and their API key in the
chat's settings panel, so the deployment itself holds no secret to leak or
meter. Env values (``PROVIDER_MODEL`` / ``PROVIDER_API_KEY`` in the environment
or .env) only *pre-fill* those fields for local single-user use. ``MCP_URL``
(index root or single MCP endpoint, default ``http://localhost:8000/mcp``) and
``CHAINLIT_PORT`` (default 8080) come from :class:`WebSettings`.

What that does *not* mean is that the key stays on the client. Chainlit sends
settings values to the server and keeps them on the session — a process-memory
dict keyed by session id — so a user's key does reach this process and lives
here for the session's lifetime. "The deployment holds no key" is a statement
about configuration, not about where the key is while someone is chatting. The
user-facing copy in :func:`start` is worded to say exactly that, and claims
nothing further; keep it honest if this changes. Session scoping is also why a
new chat asks for the key again: a new session id has no settings behind it.

.. warning::
   **Configuring a Chainlit data layer persists these keys.** On chat end
   ``WebsocketSession.to_persistable()`` copies ``chat_settings`` — the API key
   field included — into the thread metadata it writes. Chainlit drops ``env``
   there unless ``persist_user_env`` is set, but has no equivalent guard for
   ``chat_settings``, and ``clean_metadata`` only serializes and size-caps. So
   a deployment that adds a data layer to get chat history silently starts
   writing every user's provider key to its database. Nothing here configures
   one, which is the only reason the shipped default is safe. If you add one,
   clear the field off ``context.session.chat_settings`` once the agent is
   built — the key is already bound into the model by then.

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

try:
    import chainlit as cl
    from chainlit.input_widget import InputWidget, TextInput
except ModuleNotFoundError as exc:  # pragma: no cover - depends on the install
    if exc.name != "chainlit":
        raise
    raise ModuleNotFoundError(
        "mcp_agent.web needs Chainlit, which the [web] extra installs: "
        "pip install 'mcp-toolsets-runtime[web]'. The [agent] extra carries "
        "mcp_agent.main and mcp_agent.host, neither of which needs a UI "
        "framework — see mcp_agent.host if you are writing a host of your own.",
        name="chainlit",
    ) from exc
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.tools import BaseTool
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from mcp_agent.main import (
    DEFAULT_ELEMENTS_DIR,
    HOST_ELEMENTS,
    Checkpointing,
    build_agent,
    connect_error_hint,
    fetch_connections,
    first_leaf,
    new_thread_id,
    resolve_credentials,
    run_turn,
    user_credentials,
)
from mcp_agent.host import (
    VIEW_META_KEY,
    remember_views,
    step_input,
    tool_name,
    view_bundles,
    view_props,
    view_uri_for,
)
from mcp_state.state import StateEntry

# Re-exported: these used to live here, and a host importing them from this
# module keeps working. New code should take them from mcp_agent.host, which
# needs no Chainlit.
__all__ = [
    "VIEW_META_KEY",
    "remember_views",
    "step_input",
    "tool_name",
    "view_bundles",
    "view_props",
    "view_uri_for",
]

# Settings-panel field ids for the user-supplied model. Kept out of the
# environment so a hosted deployment holds no provider secret (BYOM).
MODEL_FIELD = "PROVIDER_MODEL"
API_KEY_FIELD = "PROVIDER_API_KEY"

# Title of the side panel the views render in.
VIEW_PANEL_TITLE = "Visualizations"

# Session key for the per-turn snapshots of view props ({"html", "data"}), so a
# past turn's panel can be recalled from its reply's "Show in panel" action —
# the panel only ever holds the latest turn's views, so an overwritten one
# cannot be rebuilt otherwise. remember_views bounds how many are kept.
_VIEW_HISTORY = "view_history"
_SHOW_VIEWS_ACTION = "show_views"

# Chainlit's callbacks hang off this module, so the process has no other scope
# to put a process-scoped resource in. This binding never changes; the object
# owns its own state and is opened and closed by the two hooks below, so the
# connection pool has a lifetime rather than merely a first use.
CHECKPOINTING = Checkpointing()


@cl.on_app_startup
async def open_checkpointer() -> None:
    """Connect the checkpointer before the first session, not on first message.

    Chainlit logs and swallows an exception raised here, so this cannot be the
    thing that rejects a bad configuration — :func:`main` validates the value
    before Chainlit starts, and what is left to fail here is the connection
    itself. A database that is down then surfaces on the first message, which
    is the right place for something that may simply come back.
    """
    await CHECKPOINTING.open()


@cl.on_app_shutdown
async def close_checkpointer() -> None:
    await CHECKPOINTING.aclose()


class WebSettings(BaseSettings):
    """Web-UI configuration — everything the deployment could hold is optional.

    Unlike the CLI's ``AgentSettings``, the model and its key are *not* required
    here: the hosted chat is bring-your-own-model, so each user supplies
    ``provider:model`` and their API key in the settings panel and the server
    stores no provider secret. Env values, when present (local dev / .env), only
    pre-fill those fields.

    ``extra="allow"`` so credential headers named only by the connected
    toolsets survive from a ``.env`` into ``model_extra``, where
    :func:`~mcp_agent.main.resolve_credentials` reads them.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    provider_model: str | None = None
    provider_api_key: SecretStr | None = None
    mcp_url: str = "http://localhost:8000/mcp"
    chainlit_port: int = Field(default=8080, ge=1, le=65535)


@cl.on_chat_start
async def start() -> None:
    settings = WebSettings()
    cl.user_session.set("mcp_url", settings.mcp_url)
    # One checkpointer thread per chat session. Minted here rather than reusing
    # Chainlit's session id so the conversation is addressable by something we
    # own — with a durable checkpointer it outlives the websocket that made it.
    cl.user_session.set("thread_id", new_thread_id())
    _, required = await fetch_connections(settings.mcp_url)
    cl.user_session.set("required", required)
    # Headers the deployment can already satisfy from its own environment get
    # no field and no prompt: they are applied to every turn from here on.
    from_env = resolve_credentials(required, {}, settings.model_extra)
    cl.user_session.set("env_credentials", from_env or None)
    header_names = sorted(
        {name for names in (required or {}).values() for name in names} - set(from_env)
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
            label="Model API key (held for this chat session only)",
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
            "set a **Model** (`provider:model`) and your **API key** to connect.\n\n"
            "Your key is sent to this server and held in the chat session's "
            "memory, where it is used only to authenticate your calls to the "
            "provider you chose. It never becomes part of the conversation the "
            "model reads. It is dropped when the session ends — so a new chat "
            "asks for it again."
        ).send()


async def ensure_agent(model: str, api_key: str) -> None:
    """(Re)build the session's agent for its MCP url with the user's model.

    Called when a model + key arrive (from the settings panel, or env pre-fill).
    Stores the agent and view bundles on the session and greets with what
    connected; connection failures surface in the chat, not the logs.

    A rebuilt agent resumes the session's existing conversation: the transcript
    and its ``tool_state`` belong to the thread in the checkpointer, not to the
    agent object, so switching model mid-conversation keeps both.
    """
    mcp_url: str = cl.user_session.get("mcp_url") or "http://localhost:8000/mcp"
    try:
        built = await build_agent(
            mcp_url, model, SecretStr(api_key), checkpointer=await CHECKPOINTING.saver()
        )
    except Exception as error:  # noqa: BLE001 - surface in the UI, not the logs
        await cl.Message(
            f"Could not connect with model `{model}`: {first_leaf(error)}."
            f"{connect_error_hint(mcp_url)}"
        ).send()
        return
    agent, connections, tools, withheld, required = built
    # The declaration the agent was actually wired with, which is the one the
    # panel and the per-turn credentials must agree with. `start` read it
    # before any agent existed, to draw the panel; this is the authority.
    cl.user_session.set("required", required)
    cl.user_session.set("agent", agent)
    cl.user_session.set("provider_model", model)
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
    from_env: dict[str, str] = cl.user_session.get("env_credentials") or {}
    if needing and not all(
        name in from_env for names in (required or {}).values() for name in names
    ):
        await cl.Message(
            f"Some tools act on your behalf and need credentials: {needing}. "
            "Set them in the settings panel (⚙ by the message box); each is "
            "sent only to the toolset that declares it, never to the model."
        ).send()
    if from_env:
        await cl.Message(
            f"Supplied by this deployment's environment: "
            f"{', '.join(f'`{name}`' for name in sorted(from_env))}."
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
    # A panel value wins over the environment fallback for the same header.
    credentials = {
        **(cl.user_session.get("env_credentials") or {}),
        **(cl.user_session.get("credentials") or {}),
    }
    thread_id: str = cl.user_session.get("thread_id") or new_thread_id()
    try:
        with user_credentials(credentials or None):
            turn = await run_turn(agent, message.content, thread_id)
    except Exception as error:  # noqa: BLE001 - surface in the UI, keep chatting
        await cl.Message(f"Error: {error}").send()
        return
    new_messages, tool_state = turn.new_messages, turn.sidecar
    tool_messages = [msg for msg in new_messages if isinstance(msg, ToolMessage)]
    tool_outputs = {msg.tool_call_id: msg for msg in tool_messages}
    for msg in new_messages:
        for call in getattr(msg, "tool_calls", None) or []:
            async with cl.Step(name=call["name"]) as step:
                result = tool_outputs.get(call["id"])
                step.input = step_input(call["args"], result, tool_state)
                # ``.text``, not ``str(.content)``: a server returning content
                # blocks would otherwise put a Python repr in the step.
                step.output = result.text if result else ""

    # One id per turn, keying this turn's view snapshot so the reply's "Show in
    # panel" action can bring those visualizations back after later turns have
    # replaced the panel.
    turn_id = uuid.uuid4().hex
    produced_views = await render_views(new_messages, tool_outputs, turn_id, tool_state)
    actions = [_show_views_action(turn_id)] if produced_views else []
    answer = turn.answer
    if turn.citations:
        answer += "\n\nSources: " + ", ".join(f"`{ref}`" for ref in turn.citations)
    await cl.Message(answer, actions=actions).send()


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
    session in the UI. ``CHAINLIT_PORT`` and ``MCP_AGENT_CHECKPOINT`` are read
    here at boot.
    """
    from chainlit.cli import run_chainlit

    settings = WebSettings()

    # Before Chainlit takes the process over: it logs and swallows whatever the
    # startup hook raises, so a checkpoint target that is simply not one has to
    # be rejected here or the app would serve and fail every message instead.
    try:
        CHECKPOINTING.validate()
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from None

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
