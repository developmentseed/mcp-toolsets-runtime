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

**Conversations are checkpointed**, so a caller keeps a ``thread_id`` rather
than a message list, and both the transcript and ``tool_state`` persist under
it. ``MCP_AGENT_CHECKPOINT`` selects the store: ``memory`` (the default — fine
for local dev, demos and a single-replica deployment) or a PostgreSQL URL for
anything that has to survive a restart or run more than one replica. A caller
embedding this can pass any LangGraph checkpointer to :func:`build_agent`
instead.
"""

import asyncio
import os
import uuid
from collections.abc import Iterator, Sequence
from contextlib import AsyncExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Annotated, Any, NamedTuple, cast

import httpx
import typer
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from mcp.client.streamable_http import create_mcp_http_client
from mcp.shared.exceptions import McpError
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from pydantic import Field, SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict
from rich.console import Console
from rich.markdown import Markdown

from mcp_state import (
    SESSION_STATE_PROMPT,
    StateCaptureMiddleware,
    bind_all_injected,
    describe_receipt,
    make_inspect_state,
    owners,
    publications,
    receipts_of,
    state_keys,
    with_server_name,
)
from mcp_state.state import TOOL_STATE_KEY, StateEntry

# How to set PROVIDER_MODEL when it is missing; shown in the error message.
PROVIDER_HELP = (
    "Set PROVIDER_MODEL (e.g. openai:gpt-4o-mini) and PROVIDER_API_KEY in the "
    "environment or .env, and install the provider package "
    "(e.g. uv add langchain-openai)."
)
# The demo agent's own instruction, free of any session-state vocabulary — it
# is all the plain (MCP_AGENT_STATE=0) agent gets, since a prompt describing
# notes that never appear would only mislead the model.
BASE_PROMPT = (
    "You are a helpful assistant with tools from one or more MCP toolsets. "
    "Use them whenever they can ground your answer; otherwise answer directly."
)
# What the state-wired agent runs with: the instruction above plus the
# host-agnostic fragment explaining breadcrumbs, @state:<key> handles,
# host-filled parameters, inspect_state, and the provenance the state notes
# record. A host with its own prompt composes the same way — see
# :mod:`mcp_state.prompt`.
SYSTEM_PROMPT = f"{BASE_PROMPT}\n\n{SESSION_STATE_PROMPT}"

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

    ``extra="allow"`` so credential headers named only by the connected
    toolsets (``X_DEMO_TOKEN`` and friends) survive from a ``.env`` into
    ``model_extra``, where :func:`resolve_credentials` reads them.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="allow")

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


#: ``MCP_AGENT_CHECKPOINT`` value selecting the in-process store.
MEMORY_CHECKPOINT = "memory"

#: URL schemes understood as "a PostgreSQL checkpointer". Matched explicitly so
#: a typo'd value fails with a list of what is supported, rather than being
#: handed to a driver as a DSN and failing somewhere less legible.
POSTGRES_SCHEMES = ("postgres://", "postgresql://")


def checkpoint_target(value: str) -> str:
    """Validate a ``MCP_AGENT_CHECKPOINT`` value, returning it stripped.

    Pure and I/O-free, so it can run before anything is started — which is the
    point: "that is not a checkpointer" is a startup error, and separating it
    from "the database did not answer" means the two fail in the places they
    can actually be reported.
    """
    target = value.strip()
    if target == MEMORY_CHECKPOINT or target.startswith(POSTGRES_SCHEMES):
        return target
    raise ValueError(
        f"MCP_AGENT_CHECKPOINT={target!r} is not a checkpointer: use "
        f"{MEMORY_CHECKPOINT!r} or a URL starting with "
        f"{' / '.join(POSTGRES_SCHEMES)}."
    )


class CheckpointSettings(BaseSettings):
    """Where conversation state is kept.

    ``memory`` (the default) holds threads in the process: everything is lost
    on restart and a second replica knows nothing of the first, which is fine
    for local dev, demos, and a single-replica deployment nobody expects to
    resume. A PostgreSQL URL survives both, and needs the
    ``[checkpointing-postgres]`` extra.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    mcp_agent_checkpoint: str = MEMORY_CHECKPOINT


class Checkpointing:
    """Owns one checkpointer and whatever it holds open.

    A checkpointer is *process*-scoped, not agent-scoped: the web host builds
    an agent per session and again on every model change, and a Postgres saver
    owns a connection pool, so building one per agent would open pools without
    bound. Conversations are kept apart by ``thread_id``, never by separate
    savers.

    That scope is a lifetime, not a global — this is an object an entry point
    creates, holds for as long as it serves turns, and closes. Use it as an
    async context manager where there is a scope to hang it on::

        async with Checkpointing() as checkpointing:
            built = await build_agent(url, model, key,
                                      checkpointer=await checkpointing.saver())

    A framework that owns the process instead (Chainlit, whose callbacks hang
    off a module) can hold one and drive :meth:`open` / :meth:`aclose` from its
    startup and shutdown hooks, which is what ``mcp_agent.web`` does.

    ``target`` defaults to ``MCP_AGENT_CHECKPOINT``. The value is only read
    when a saver is actually built, so constructing this is free and cannot
    fail on a misconfigured environment.
    """

    def __init__(self, target: str | None = None) -> None:
        self._target = target
        self._saver: BaseCheckpointSaver | None = None
        self._resources = AsyncExitStack()

    async def saver(self) -> BaseCheckpointSaver:
        """The checkpointer, built on first use and reused after."""
        if self._saver is None:
            self._saver = await self._build()
        return self._saver

    def validate(self) -> str:
        """Check the configured target without opening anything.

        For an entry point that wants a bad value to fail before it starts
        serving — see ``mcp_agent.web.main``, where the alternative is a chat
        that greets the user and then fails the moment they say anything.
        """
        return checkpoint_target(
            self._target
            if self._target is not None
            else CheckpointSettings().mcp_agent_checkpoint
        )

    async def _build(self) -> BaseCheckpointSaver:
        target = self.validate()
        if target == MEMORY_CHECKPOINT:
            return InMemorySaver()
        return await self._postgres(target)

    async def _postgres(self, url: str) -> BaseCheckpointSaver:
        """A Postgres saver held open until this object is closed.

        ``from_conn_string`` is an async context manager owning a connection
        pool, and the agent needs it for as long as it serves turns — hence
        the exit stack, rather than a ``with`` block that would close the pool
        before the first message arrived.
        """
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        except ImportError:
            raise RuntimeError(
                "MCP_AGENT_CHECKPOINT names a PostgreSQL URL but the driver is "
                "not installed. Add the extra: "
                "mcp-toolsets-runtime[checkpointing-postgres]."
            ) from None
        saver = await self._resources.enter_async_context(
            AsyncPostgresSaver.from_conn_string(url)
        )
        await saver.setup()  # idempotent: creates the checkpoint tables if absent
        return saver

    async def open(self) -> BaseCheckpointSaver:
        """Build the saver eagerly, so a bad DSN fails at startup not mid-chat."""
        return await self.saver()

    async def aclose(self) -> None:
        """Release the connection pool, if one was ever opened."""
        await self._resources.aclose()
        self._resources = AsyncExitStack()
        self._saver = None

    async def __aenter__(self) -> "Checkpointing":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()


def new_thread_id() -> str:
    """A fresh conversation id. One per CLI run, one per web session."""
    return uuid.uuid4().hex


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


def credential_env_var(header: str) -> str:
    """Env var a credential header falls back to (``x-demo-token`` -> ``X_DEMO_TOKEN``)."""
    return header.upper().replace("-", "_")


def headers_from_flags(flags: list[str]) -> dict[str, str]:
    """Parse repeated ``--header NAME=VALUE`` options into a header map.

    Names are lowercased to match the toolsets' advertised (lowercase)
    declarations; a later flag wins over an earlier one for the same name.
    """
    parsed: dict[str, str] = {}
    for flag in flags:
        name, sep, value = flag.partition("=")
        name = name.strip().lower()
        if not sep or not name:
            raise typer.BadParameter(
                f"expected NAME=VALUE, got {flag!r}", param_hint="--header"
            )
        parsed[name] = value
    return parsed


def resolve_credentials(
    required: dict[str, list[str]] | None,
    flags: dict[str, str],
    dotenv_extra: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Resolve credential headers from explicit flags, then the environment.

    A flag always wins. For each header a connected toolset advertises
    (``required``) and no flag supplied, the matching env var (see
    :func:`credential_env_var`) is used when set — the real process
    environment first, then ``dotenv_extra``, so an exported variable beats a
    ``.env`` one. ``dotenv_extra`` takes ``BaseSettings.model_extra``, whose
    keys are lower-cased.

    Flags naming headers no toolset declared still pass through: with a direct
    URL ``required`` is ``None`` and the per-toolset gate in
    :func:`credential_client_factory` forwards everything anyway.
    """
    resolved = dict(flags)
    for header in {name for names in (required or {}).values() for name in names}:
        if header in resolved:
            continue
        var = credential_env_var(header)
        value = os.environ.get(var) or (dotenv_extra or {}).get(var.lower())
        if value:
            resolved[header] = str(value)
    return resolved


def receipt_lines(arguments: dict[str, Any], result: BaseMessage | None) -> list[str]:
    """What a printed tool call shows as a handle, resolved to what it named.

    The argument above reads ``@state:raster-ops/clip/aoi``; this says which
    tool published that and confirms the key existed. One line each, or none
    for a tool that took nothing from state.
    """
    received = receipts_of(getattr(result, "artifact", None))
    return [
        describe_receipt(parameter, receipt)
        for parameter, receipt in sorted(received.items())
    ]


def with_session_state(
    model: Any,
    tools: list[BaseTool],
    checkpointer: BaseCheckpointSaver | None = None,
    system_prompt: str = SYSTEM_PROMPT,
    extra_tools: Sequence[BaseTool] = (),
    middleware: Sequence[Any] = (),
) -> Any:
    """Build the agent with :mod:`mcp_state` wired in.

    The three pieces are interdependent and all three are needed (the pattern
    ``docs/CONSUMING.md`` documents for anyone assembling their own agent): the
    middleware brings the ``tool_state`` namespace and captures into it,
    ``bind_all_injected`` reads back out of it, and ``inspect_state`` lets the
    model read a value it was only told the key of.

    ``system_prompt``, ``extra_tools`` and ``middleware`` are the seams a host
    with its own prompt, its own local tools, or its own callbacks builds on.
    ``system_prompt`` is used verbatim — a host replacing it should append
    :data:`mcp_state.SESSION_STATE_PROMPT` to its own instructions, since this
    function always wires the machinery that fragment describes.
    ``extra_tools`` are added as given — they are the host's own, not MCP
    tools, so they are neither bound to session state nor checked against it.
    ``middleware`` runs after :class:`~mcp_state.StateCaptureMiddleware`.
    """
    published = publications(tools)
    return create_agent(
        model,
        [
            *bind_all_injected(tools),
            make_inspect_state(state_keys(published)),
            *extra_tools,
        ],
        system_prompt=system_prompt,
        middleware=[
            StateCaptureMiddleware(published, owners=owners(tools)),
            *middleware,
        ],
        checkpointer=checkpointer,
    )


class BuiltAgent(NamedTuple):
    """An agent and what it was built from.

    ``tools`` are as loaded, *before* binding — a UI reads each tool's
    ``_meta`` off these, and binding is an agent-side concern. ``required`` is
    the per-toolset
    credential-header declaration discovered along with the connections, or
    ``None`` for a direct URL that advertised none; a caller resolving
    credentials wants the same declaration the agent was wired with, rather
    than a second lookup that could disagree with it.
    """

    agent: Any
    connections: dict[str, Any]
    tools: list[BaseTool]
    required: dict[str, list[str]] | None


async def build_agent(
    url: str,
    model: str,
    api_key: SecretStr,
    session_state: bool | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    system_prompt: str | None = None,
    extra_tools: Sequence[BaseTool] = (),
    middleware: Sequence[Any] = (),
) -> BuiltAgent:
    """Discover the servers behind ``url`` and build a tool-calling agent.

    Built once per process/session: per-user credentials are not baked in but
    read from :func:`user_credentials` on every tool call. ``model`` is a
    ``provider:model`` string for :func:`init_chat_model` and ``api_key`` is
    that provider's key, so the agent is provider-agnostic.

    ``session_state`` keeps large tool values out of the model's context; it
    defaults to :class:`StateSettings` (``MCP_AGENT_STATE``, on unless set
    otherwise). Pass it explicitly to ignore the environment.

    ``checkpointer`` is where conversations live, keyed by ``thread_id``. Any
    LangGraph saver does — pass a configured ``AsyncPostgresSaver`` (or your
    own) and this makes no assumptions about it.

    Omitted, each call gets a *fresh in-process* saver. That is right for
    building one agent and wrong for building several, since two agents with
    separate savers cannot see each other's threads: a caller that rebuilds an
    agent (on a model change, say) and expects the conversation to survive
    must own one :class:`Checkpointing` and pass its saver every time, as both
    entry points here do. There is deliberately no shared default doing it
    invisibly — which saver an agent writes to is worth being able to see.

    ``system_prompt``, ``extra_tools`` and ``middleware`` let a host layer its
    own prompt, local tools and callbacks over the discovered MCP tools; see
    :func:`with_session_state`. They apply whether or not session state is on.
    Left as ``None``, the prompt follows the wiring: the state-wired agent
    gets :data:`SYSTEM_PROMPT` (the base instruction plus
    :data:`mcp_state.SESSION_STATE_PROMPT`), the plain agent just
    :data:`BASE_PROMPT`. A host passing its own prompt with state on should
    make the same composition — the fragment is what tells the model how
    handles, filled parameters and the state notes work.

    Returns a :class:`BuiltAgent`.
    """
    if session_state is None:
        session_state = StateSettings().mcp_agent_state
    if system_prompt is None:
        # The default prompt has to match the wiring: only the state-wired
        # agent is told about breadcrumbs, handles and filled parameters.
        system_prompt = SYSTEM_PROMPT if session_state else BASE_PROMPT
    if checkpointer is None:
        checkpointer = InMemorySaver()
    connections, required = await fetch_connections(url)
    # Loaded per server rather than in one call, so each tool can be stamped
    # with where it came from: `langchain_mcp_adapters` takes a `server_name`
    # and records it nowhere on the tool it builds, and an undeclared capture
    # needs it to key a value the same way a declared one is keyed.
    client = MultiServerMCPClient(with_credential_support(connections, required))
    tools = [
        with_server_name(tool, server)
        for server in connections
        for tool in await client.get_tools(server_name=server)
    ]
    chat_model = init_chat_model(model, api_key=api_key.get_secret_value())
    if not session_state:
        return BuiltAgent(
            create_agent(
                chat_model,
                [*tools, *extra_tools],
                system_prompt=system_prompt,
                middleware=list(middleware),
                checkpointer=checkpointer,
            ),
            connections,
            tools,
            required,
        )
    agent = with_session_state(
        chat_model,
        tools,
        checkpointer,
        system_prompt=system_prompt,
        extra_tools=extra_tools,
        middleware=middleware,
    )
    return BuiltAgent(agent, connections, tools, required)


@dataclass
class TurnResult:
    """What one :func:`run_turn` produced.

    ``history`` is the thread's full transcript, ``new_messages`` just this
    turn's replies. ``answer`` is the final message flattened to plain text.
    ``sidecar`` is the thread's ``tool_state``, which a UI needs to render this
    turn's views (:func:`mcp_state.restore_structured`); it is ``None`` when
    the agent has no state namespace.
    """

    history: list[BaseMessage]
    new_messages: list[BaseMessage]
    answer: str
    sidecar: dict[str, StateEntry] | None = None
    #: Ids the model cited on ``reference`` content blocks, in first-seen
    #: order; empty when the answer is plain text carrying no blocks.
    citations: list[str] = field(default_factory=list)


def _block_citations(block: dict[str, Any]) -> Iterator[str]:
    """Every citation id one standardised content block carries.

    The standard annotation is ``{"type": "citation", ...}``, which is what
    :func:`answer_citations` gets by reading ``content_blocks``. A ``Citation``
    need not carry every field, so identity falls back ``id`` -> ``title`` ->
    ``url`` -> ``cited_text``: a citation with only an excerpt is still worth
    surfacing.

    ``reference``/``reference_ids`` is Mistral's, and survives translation
    because opaque reference ids have nowhere to go in ``Citation`` — it has
    ``url``, ``title`` and ``cited_text``, none of which an id is.
    """
    reference = block.get("reference")
    if isinstance(reference, dict):
        yield from reference.get("reference_ids") or []
    for annotation in block.get("annotations") or []:
        if not isinstance(annotation, dict) or annotation.get("type") != "citation":
            continue
        extras = annotation.get("extras")
        if isinstance(extras, dict) and extras.get("reference_ids"):
            yield from extras["reference_ids"]
            continue
        for name in ("id", "title", "url", "cited_text"):
            if value := annotation.get(name):
                yield value
                break


def answer_citations(message: BaseMessage) -> list[str]:
    """Ids a message cited, in first-seen order and without duplicates.

    Read off ``content_blocks``, not ``content``: on ``content`` every provider
    keeps its own shape — Anthropic a native ``citations`` key, OpenAI
    ``url_citation`` annotations, Mistral ``reference`` blocks — so matching
    there means matching one provider and silently missing the rest.
    ``content_blocks`` is the standardised view, where all of them arrive as
    ``citation`` annotations.

    Falls back to ``content`` where that accessor is unavailable, and plain
    string content carries no blocks and yields nothing either way.
    """
    blocks = getattr(message, "content_blocks", None)
    if not isinstance(blocks, list):
        blocks = message.content
    if not isinstance(blocks, list):
        return []
    ordered: dict[str, None] = {}
    for block in blocks:
        if not isinstance(block, dict):
            continue
        for ref in _block_citations(block):
            if isinstance(ref, str) and ref:
                ordered.setdefault(ref, None)
    return list(ordered)


async def run_turn(
    agent: Any,
    text: str,
    thread_id: str,
    config: dict[str, Any] | None = None,
) -> TurnResult:
    """Run one chat turn on ``thread_id``.

    Only the new message is sent. The thread's transcript and its ``tool_state``
    both live in the checkpointer, so a caller keeps an id rather than a message
    list, and a value captured on one turn is still there on the next — which
    was previously the caller's job to arrange, and silent to get wrong.

    ``config`` is the runnable config handed to ``ainvoke``, for a host
    attaching per-turn callbacks or metadata (tracing, say). ``thread_id`` is
    merged into its ``configurable`` and wins over any set there.

    The thread is read before the turn purely to know where this turn's messages
    begin — its length is the boundary, and it is cheap next to the model call.
    """
    merged = dict(config or {})
    merged["configurable"] = {
        **(merged.get("configurable") or {}),
        "thread_id": thread_id,
    }
    before = await agent.aget_state(cast(Any, merged))
    seen = len((getattr(before, "values", None) or {}).get("messages") or [])
    result = await agent.ainvoke(
        cast(Any, {"messages": [HumanMessage(text)]}), cast(Any, merged)
    )
    history: list[BaseMessage] = result["messages"]
    # +1 skips the HumanMessage just added: "new" means the agent's replies.
    new_messages = history[seen + 1 :]
    last = new_messages[-1] if new_messages else None
    return TurnResult(
        history=history,
        new_messages=new_messages,
        # ``.text`` flattens list-structured content (a model may interleave
        # text with reference blocks); ``str(.content)`` would render the raw
        # block list as a Python repr.
        answer=str(last.text) if last is not None else "",
        sidecar=result.get(TOOL_STATE_KEY),
        citations=answer_citations(last) if last is not None else [],
    )


async def chat_loop(
    url: str,
    model: str,
    api_key: SecretStr,
    headers: dict[str, str] | None = None,
    dotenv_extra: dict[str, Any] | None = None,
) -> None:
    """One run of the interactive CLI chat.

    The whole process is one scope here — one agent, one conversation — so the
    checkpointer is a local held for the duration and closed on the way out,
    which is the shape :class:`Checkpointing` exists for.
    """
    checkpointing = Checkpointing()
    try:
        checkpointing.validate()
    except ValueError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(2) from None
    async with checkpointing:
        await _chat_loop(
            url,
            model,
            api_key,
            await checkpointing.open(),
            headers or {},
            dotenv_extra,
        )


async def _chat_loop(
    url: str,
    model: str,
    api_key: SecretStr,
    checkpointer: BaseCheckpointSaver,
    headers: dict[str, str],
    dotenv_extra: dict[str, Any] | None = None,
) -> None:
    try:
        built = await build_agent(url, model, api_key, checkpointer=checkpointer)
    except* CONNECT_ERRORS as group:
        console.print(
            f"[red]Could not reach the MCP server(s) behind {url}: "
            f"{first_leaf(group)}[/red]"
        )
        if hint := connect_error_hint(url):
            console.print(f"[yellow]{hint.strip()}[/yellow]")
        raise typer.Exit(1) from None

    credentials = resolve_credentials(built.required, headers, dotenv_extra) or None
    console.print(
        f"Connected to [bold]{len(built.connections)}[/bold] server(s): "
        f"{', '.join(built.connections)}"
    )
    console.print(
        f"[dim]{len(built.tools)} tools: {', '.join(t.name for t in built.tools)}[/dim]"
    )
    if credentials:
        console.print(f"[dim]credentials: {', '.join(sorted(credentials))}[/dim]")
    if missing := sorted(
        {name for names in (built.required or {}).values() for name in names}
        - set(credentials or {})
    ):
        console.print(
            f"[yellow]no value for {', '.join(missing)} "
            "(pass --header NAME=VALUE, or set the matching env var)[/yellow]"
        )
    console.print("[dim]Type a message, or quit to exit.[/dim]")

    # One thread for the run. With an in-process checkpointer that is the whole
    # lifetime anyway; against Postgres it is what a `--thread` option would
    # resume, which is left for whoever wants it.
    thread_id = new_thread_id()
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
            with user_credentials(credentials):
                turn = await run_turn(built.agent, line, thread_id)
        except Exception as error:  # noqa: BLE001 - keep the chat alive
            console.print(f"[red]{error}[/red]")
            continue
        results = {
            msg.tool_call_id: msg
            for msg in turn.new_messages
            if isinstance(msg, ToolMessage)
        }
        for message in turn.new_messages:
            for call in getattr(message, "tool_calls", None) or []:
                console.print(f"[dim]→ {call['name']} {call['args']}[/dim]")
                for line in receipt_lines(call["args"], results.get(call["id"])):
                    console.print(f"[dim]  {line}[/dim]")
        console.print(Markdown(turn.answer))
        if turn.citations:
            console.print(f"[dim]Sources: {', '.join(turn.citations)}[/dim]")


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
    header: Annotated[
        list[str] | None,
        typer.Option(
            "--header",
            help="Credential header as NAME=VALUE, repeatable. Any header a "
            "toolset advertises and no flag supplies falls back to its env "
            "var (x-demo-token -> X_DEMO_TOKEN).",
        ),
    ] = None,
) -> None:
    """Discover the MCP servers behind URL and chat with their tools.

    URL and model are read from the environment / .env (the same place as
    PROVIDER_API_KEY) when omitted, so the agent can be configured entirely
    by a .env file; CLI arguments override it.

    Credential headers ride the MCP transport, so the model never sees them.
    """
    try:
        settings = AgentSettings(provider_model=model) if model else AgentSettings()
    except ValidationError:
        console.print(f"[red]{PROVIDER_HELP}[/red]")
        raise typer.Exit(1) from None
    asyncio.run(
        chat_loop(
            url or settings.mcp_url,
            settings.provider_model,
            settings.provider_api_key,
            headers_from_flags(header or []),
            settings.model_extra,
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
