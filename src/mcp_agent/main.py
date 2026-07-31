"""Chat with every toolset behind an mcp-toolsets index URL.

Point ``mcp-agent`` at an index root (anything serving a ``connections`` map
shaped for ``MultiServerMCPClient``) or directly at a single MCP endpoint;
it loads every server's tools and lets a chat model drive them in an
interactive chat.

The model is provider-agnostic and no provider ships by default: pick one by
setting ``PROVIDER_MODEL`` to a ``provider:model`` string for LangChain's
``init_chat_model`` (e.g. ``openai:gpt-4o-mini``,
``anthropic:claude-3-5-haiku-latest``) and installing that provider's package
(``uv add langchain-openai``). ``PROVIDER_API_KEY`` (the chosen provider's key)
and ``PROVIDER_MODEL`` are read from the environment or a ``.env`` file.

**Session state is on.** Large tool values are kept out of the model's context
by :mod:`mcp_state` — captured into ``tool_state`` on the way back, injected
into the tools that take them on the way out (see ``docs/SESSION-STATE.md``).
Set ``MCP_AGENT_STATE=0`` to build the plain agent instead: no capture, no
injection, every value through the transcript as before.
"""

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from importlib import resources
from pathlib import Path
from typing import Annotated, Any, cast

import httpx
import typer
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from mcp.client.streamable_http import create_mcp_http_client
from mcp.shared.exceptions import McpError
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from pydantic import Field, SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict
from rich.console import Console
from rich.markdown import Markdown

from mcp_state import (
    AgentState,
    StateCaptureMiddleware,
    Unsatisfiable,
    bind_all_injected,
    make_inspect_state,
    partition_usable,
    publications,
    state_keys,
)
from mcp_state.state import TOOL_STATE_KEY, StateEntry

# How to set PROVIDER_MODEL when it is missing; shown in the error message.
PROVIDER_HELP = (
    "Set PROVIDER_MODEL (e.g. openai:gpt-4o-mini) and PROVIDER_API_KEY in the "
    "environment or .env, and install the provider package "
    "(e.g. uv add langchain-openai)."
)
SYSTEM_PROMPT = (
    "You are a helpful assistant with tools from one or more MCP toolsets. "
    "Use them whenever they can ground your answer; otherwise answer directly."
)

app = typer.Typer(no_args_is_help=True, help=__doc__)
console = Console()

# Chainlit host elements shipped as package data (src/mcp_agent/elements/). The
# web agent renders tool views via a Chainlit CustomElement Chainlit loads from
# <app-root>/public/elements/, which defaults to ./public/elements relative to
# where you launch it.
HOST_ELEMENTS = ("McpView.jsx",)
DEFAULT_ELEMENTS_DIR = Path("public/elements")


def install_host_elements(target: Path) -> list[Path]:
    """Copy the packaged Chainlit host element(s) into ``target``.

    Returns the paths written. Deterministic and idempotent: it always writes
    the version shipped with the installed package, so an upgrade + reinstall
    refreshes the element with no drift. Meant to be run at build time (see the
    ``install-elements`` command) rather than as a runtime side effect.
    """
    target.mkdir(parents=True, exist_ok=True)
    source = resources.files("mcp_agent") / "elements"
    written = []
    for name in HOST_ELEMENTS:
        dest = target / name
        dest.write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")
        written.append(dest)
    return written


class AgentSettings(BaseSettings):
    """Agent configuration, validated from the environment or a .env file.

    The CLI takes the URL and model as arguments; the web UI (``web.py``)
    reads ``MCP_URL`` and ``PROVIDER_MODEL`` from here instead.
    ``PROVIDER_MODEL`` and ``PROVIDER_API_KEY`` are required — there is no
    default provider; ``PROVIDER_API_KEY`` is passed straight to
    ``init_chat_model``.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    provider_api_key: SecretStr
    provider_model: str
    mcp_url: str = "http://localhost:8000/mcp"
    chainlit_port: int = Field(default=8080, ge=1, le=65535)


class StateSettings(BaseSettings):
    """Whether the agent keeps large tool values out of the model's context.

    Its own settings class, not a field on :class:`AgentSettings`, because
    :func:`build_agent` serves both the CLI and the web host and only the CLI
    can construct ``AgentSettings`` (the web host is bring-your-own-model, so
    it holds no ``provider_api_key`` to satisfy it).

    On by default: an agent driving toolsets that declare what they publish
    should use those declarations, and the failure mode of leaving it off is
    silent — a large value burns context on every subsequent turn and nothing
    reports it. ``MCP_AGENT_STATE=0`` opts out, for a host that renders tool
    results straight from the transcript or wires its own middleware.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    mcp_agent_state: bool = True


def connections_from(url: str, payload: Any) -> dict[str, Any]:
    """Extract an index payload's connections map, else treat url as one server."""
    if isinstance(payload, dict) and isinstance(payload.get("connections"), dict):
        return payload["connections"]
    return {"server": {"transport": "streamable_http", "url": url}}


def credential_headers_from(payload: Any) -> dict[str, list[str]] | None:
    """Per-toolset credential header names from an index payload.

    ``None`` means the payload was not an index (a direct single-server URL),
    so no declarations are available.
    """
    if not (isinstance(payload, dict) and isinstance(payload.get("toolsets"), list)):
        return None
    return {
        entry["name"]: [
            header.lower() for header in entry.get("credential_headers", [])
        ]
        for entry in payload["toolsets"]
        if isinstance(entry, dict) and entry.get("name")
    }


# Connection failures an agent should report rather than crash on.
CONNECT_ERRORS = (httpx.HTTPError, OSError, McpError)


def first_leaf(error: BaseException) -> BaseException:
    """Unwrap (possibly nested) ExceptionGroups to the first real exception."""
    while isinstance(error, BaseExceptionGroup):
        error = error.exceptions[0]
    return error


def connect_error_hint(url: str) -> str:
    """A nudge for the most common misconfiguration: a missing /mcp path."""
    if url.rstrip("/").endswith("/mcp"):
        return ""
    return (
        " Hint: single-toolset servers serve MCP under /mcp "
        "(e.g. http://localhost:8000/mcp); only an index is served at the root."
    )


def health_url_for(url: str) -> str | None:
    """Derive a direct MCP endpoint's sibling /health URL, if there is one."""
    base = url.rstrip("/")
    return base.removesuffix("/mcp") + "/health" if base.endswith("/mcp") else None


async def single_server_credential_headers(
    client: httpx.AsyncClient, url: str
) -> dict[str, list[str]] | None:
    """Ask a direct MCP endpoint's /health which credential headers it reads.

    Returns ``None`` when there is no health route or it doesn't advertise
    credentials (e.g. a non-mcp-toolsets server).
    """
    health_url = health_url_for(url)
    if health_url is None:
        return None
    try:
        health = (await client.get(health_url)).json()
        headers = health.get("credential_headers")
    except (httpx.HTTPError, ValueError, AttributeError):
        return None
    if not isinstance(headers, list):
        return None
    return {"server": [str(header).lower() for header in headers]}


async def fetch_connections(
    url: str,
) -> tuple[dict[str, Any], dict[str, list[str]] | None]:
    """Resolve a URL to a MultiServerMCPClient config plus credential needs."""
    async with httpx.AsyncClient(follow_redirects=True, timeout=10.0) as client:
        try:
            payload = (await client.get(url)).json()
        except (httpx.HTTPError, ValueError):
            payload = None
        connections = connections_from(url, payload)
        required = credential_headers_from(payload)
        if required is None:
            required = await single_server_credential_headers(client, url)
    return connections, required


_credentials: ContextVar[dict[str, str] | None] = ContextVar(
    "user_credentials", default=None
)


@contextmanager
def user_credentials(headers: dict[str, str] | None) -> Iterator[None]:
    """Provide the calling user's credential headers for the duration.

    This is how an agent passes a user's secrets to the tools without the
    model ever seeing them: they ride the MCP transport, not the conversation.
    The agent is built once; wrap each turn (``run_turn``) in this and the
    tool calls made inside read the values at request time, so one long-lived
    agent serves many users with different credentials.
    """
    token = _credentials.set(headers)
    try:
        yield
    finally:
        _credentials.reset(token)


def credential_client_factory(allowed: list[str] | None) -> Any:
    """Build an httpx client factory injecting the current user's credentials.

    Only headers named in ``allowed`` (the toolset's advertised declaration)
    are injected, so unrelated toolsets never receive them; ``None`` means no
    declaration was discoverable (a server the user pointed at directly) and
    every provided header is sent.
    """
    wanted = None if allowed is None else {header.lower() for header in allowed}

    def factory(
        headers: dict[str, str] | None = None,
        timeout: httpx.Timeout | None = None,
        auth: httpx.Auth | None = None,
    ) -> httpx.AsyncClient:
        provided = _credentials.get() or {}
        send = {
            header: value
            for header, value in provided.items()
            if wanted is None or header.lower() in wanted
        }
        return create_mcp_http_client(
            headers={**(headers or {}), **send}, timeout=timeout, auth=auth
        )

    return factory


def with_credential_support(
    connections: dict[str, Any], required: dict[str, list[str]] | None
) -> dict[str, Any]:
    """Wire each connection to inject per-user credentials at call time."""
    return {
        name: {
            **connection,
            "httpx_client_factory": credential_client_factory(
                None if required is None else required.get(name, [])
            ),
        }
        for name, connection in connections.items()
    }


def with_session_state(
    model: Any, tools: list[BaseTool]
) -> tuple[Any, list[Unsatisfiable]]:
    """Build the agent with :mod:`mcp_state` wired in, and say what it dropped.

    The four pieces are interdependent and all four are needed (the pattern
    ``docs/CONSUMING.md`` documents for anyone assembling their own agent):
    ``AgentState`` adds the ``tool_state`` namespace, the middleware captures
    into it, ``bind_all_injected`` reads back out of it, and ``inspect_state``
    lets the model read a value it was only told the key of.

    ``partition_usable`` withholds a tool whose required parameter nothing
    connected can fill and a model may not invent — calling it could only
    raise. The returned list of :class:`~mcp_state.wiring.Unsatisfiable` is a
    wiring report for the caller to surface; it is empty in a sound
    deployment.
    """
    published = publications(tools)
    agent_tools, withheld = partition_usable(bind_all_injected(tools))
    agent = create_agent(
        model,
        [*agent_tools, make_inspect_state(state_keys(published))],
        system_prompt=SYSTEM_PROMPT,
        state_schema=AgentState,
        middleware=[StateCaptureMiddleware(published)],
    )
    return agent, withheld


async def build_agent(
    url: str, model: str, api_key: SecretStr, session_state: bool | None = None
) -> tuple[Any, dict[str, Any], list[BaseTool], list[Unsatisfiable]]:
    """Discover the servers behind ``url`` and build a tool-calling agent.

    Built once per process/session: per-user credentials are not baked in but
    read from :func:`user_credentials` on every tool call. ``model`` is a
    ``provider:model`` string for :func:`init_chat_model` and ``api_key`` is
    that provider's key, so the agent is provider-agnostic.

    ``session_state`` keeps large tool values out of the model's context; it
    defaults to :class:`StateSettings` (``MCP_AGENT_STATE``, on unless set
    otherwise). Pass it explicitly to ignore the environment.

    Returns the agent, the connections it discovered, the tools as loaded
    (*before* binding — a UI reads each tool's ``_meta`` off these, and
    binding is an agent-side concern), and any tools withheld as uncallable.
    """
    if session_state is None:
        session_state = StateSettings().mcp_agent_state
    connections, required = await fetch_connections(url)
    tools = await MultiServerMCPClient(
        with_credential_support(connections, required)
    ).get_tools()
    chat_model = init_chat_model(model, api_key=api_key.get_secret_value())
    if not session_state:
        return (
            create_agent(chat_model, tools, system_prompt=SYSTEM_PROMPT),
            connections,
            tools,
            [],
        )
    agent, withheld = with_session_state(chat_model, tools)
    return agent, connections, tools, withheld


async def run_turn(
    agent: Any,
    messages: list[BaseMessage],
    text: str,
    tool_state: dict[str, StateEntry] | None = None,
) -> tuple[list[BaseMessage], list[BaseMessage], dict[str, StateEntry] | None]:
    """Run one chat turn; return the history, this turn's new messages, and state.

    ``tool_state`` has to make the round trip explicitly. The agent is invoked
    per turn with a state dict built here, so anything the caller does not pass
    back in is gone — a value captured on one turn would be unreachable on the
    next, and injection would silently find nothing to inject.

    ``None`` means this agent has no state namespace: either nothing has been
    captured yet, or session state is off entirely. Passing it straight back is
    correct in both cases, so a caller round-trips whatever it was handed and
    never needs to know which agent it is driving.
    """
    state: dict[str, Any] = {"messages": [*messages, HumanMessage(text)]}
    if tool_state is not None:
        state[TOOL_STATE_KEY] = tool_state
    result = await agent.ainvoke(cast(Any, state))
    history: list[BaseMessage] = result["messages"]
    return history, history[len(messages) + 1 :], result.get(TOOL_STATE_KEY)


async def chat_loop(url: str, model: str, api_key: SecretStr) -> None:
    try:
        agent, connections, tools, withheld = await build_agent(url, model, api_key)
    except* CONNECT_ERRORS as group:
        console.print(
            f"[red]Could not reach the MCP server(s) behind {url}: "
            f"{first_leaf(group)}[/red]"
        )
        if hint := connect_error_hint(url):
            console.print(f"[yellow]{hint.strip()}[/yellow]")
        raise typer.Exit(1) from None

    console.print(
        f"Connected to [bold]{len(connections)}[/bold] server(s): "
        f"{', '.join(connections)}"
    )
    console.print(f"[dim]{len(tools)} tools: {', '.join(t.name for t in tools)}[/dim]")
    for item in withheld:
        console.print(f"[yellow]withholding {item}[/yellow]")
    console.print("[dim]Type a message, or quit to exit.[/dim]")

    messages: list[BaseMessage] = []
    tool_state: dict[str, StateEntry] | None = None
    while True:
        try:
            line = console.input("[bold cyan]you>[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line in ("quit", "exit"):
            break
        try:
            messages, new_messages, tool_state = await run_turn(
                agent, messages, line, tool_state
            )
        except Exception as error:  # noqa: BLE001 - keep the chat alive
            console.print(f"[red]{error}[/red]")
            continue
        for message in new_messages:
            for call in getattr(message, "tool_calls", None) or []:
                console.print(f"[dim]→ {call['name']} {call['args']}[/dim]")
        console.print(Markdown(str(messages[-1].content)))


@app.command()
def chat(
    url: Annotated[
        str | None,
        typer.Argument(
            help="Index URL serving a connections map, or a single MCP endpoint. "
            "Defaults to MCP_URL from the environment / .env.",
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            help="Chat model as provider:model (e.g. openai:gpt-4o-mini). "
            "Overrides PROVIDER_MODEL from the environment / .env.",
        ),
    ] = None,
) -> None:
    """Discover the MCP servers behind URL and chat with their tools.

    URL and model are read from the environment / .env (the same place as
    PROVIDER_API_KEY) when omitted, so the agent can be configured entirely
    by a .env file; CLI arguments override it.
    """
    try:
        settings = AgentSettings(provider_model=model) if model else AgentSettings()
    except ValidationError:
        console.print(f"[red]{PROVIDER_HELP}[/red]")
        raise typer.Exit(1) from None
    asyncio.run(
        chat_loop(
            url or settings.mcp_url, settings.provider_model, settings.provider_api_key
        )
    )


@app.command("install-elements")
def install_elements(
    target: Annotated[
        Path,
        typer.Argument(
            help="Directory to install the Chainlit host element(s) into, "
            "typically your app root's public/elements.",
        ),
    ] = DEFAULT_ELEMENTS_DIR,
) -> None:
    """Copy the packaged Chainlit host element(s) into a chainlit app root.

    The web agent (``mcp-agent-web``) renders tool views via a Chainlit
    CustomElement named "McpView", which Chainlit loads from
    ``<app-root>/public/elements/``. Run this at build time — e.g. in your
    Dockerfile: ``RUN mcp-agent install-elements`` — so the element is present
    without the package writing to your filesystem at runtime.
    """
    for dest in install_host_elements(target):
        console.print(f"[green]installed[/green] {dest}")
