"""The agent over HTTP as a whole service, for a deployment with no app of its own.

:mod:`mcp_agent_api.routes` is an ``APIRouter`` and deliberately nothing else.
This is the other end: the application around it — the lifespan that connects to
the MCP servers, the checkpointer those conversations live in, and CORS for a
browser client served from somewhere else.

Two ways in, and the difference is who owns the agent::

    # everything from the environment
    from mcp_agent_api.app import app

    # an agent built your way
    from mcp_agent_api.app import create_app
    app = create_app(build=my_async_factory)

``uvicorn mcp_agent_api.app:app`` serves the first. Requires the ``[api]``
extra.

**The factory is async and runs in the lifespan.**
:func:`~mcp_agent.main.build_agent` connects to MCP servers, and
:func:`create_app` is called at import time, when there is no loop to do that
on. So ``build`` is awaited during startup rather than passed a finished agent,
and a caller that already has one passes ``lambda: already_built`` wrapped in a
coroutine. Until it returns, the routes answer ``503``.

**Settings are read in two places, on purpose.** CORS is middleware, which
FastAPI only accepts before startup — so those are read by :func:`create_app`,
and they are all optional, which is what makes a module-level ``app`` safe to
import anywhere. The provider model and its key are required, and are read
*inside* the lifespan by the default factory: a missing ``PROVIDER_MODEL`` then
fails the process at startup with a legible error, rather than at import of a
module something merely touched.
"""

import os
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from mcp_agent.main import AgentSettings, Checkpointing, build_agent
from mcp_agent_api.routes import Built, TurnContext, create_router

#: Comma-separated origins a browser client may call this API from. Empty (the
#: default) adds no CORS middleware at all, which is right for an API behind
#: the same origin as its UI, or reached through a dev server's proxy.
CORS_ORIGINS_VAR = "MCP_AGENT_API_CORS_ORIGINS"

#: Builds the agent the routes serve. Async because connecting to MCP servers
#: is; see the module docstring.
Builder = Callable[[], Awaitable[Built]]


def cors_origins(value: str | None = None) -> list[str]:
    """Origins from a comma-separated string, or from the environment.

    A plain string rather than pydantic-settings' list parsing, which wants
    JSON for a complex type — ``MCP_AGENT_API_CORS_ORIGINS=https://a,https://b``
    is what someone writes in a Helm values file without being told to.
    """
    raw = os.environ.get(CORS_ORIGINS_VAR, "") if value is None else value
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def from_environment(checkpointing: Checkpointing) -> Builder:
    """The default factory: an agent from ``AgentSettings``, checkpointed.

    Constructed per app rather than read at import, so the environment is
    sampled when the process starts serving and a test can point the same
    application at a different one.
    """

    async def build() -> Built:
        settings = AgentSettings()
        return await build_agent(
            settings.mcp_url,
            settings.provider_model,
            settings.provider_api_key,
            checkpointer=await checkpointing.saver(),
        )

    return build


def create_app(
    build: Builder | None = None,
    *,
    origins: Sequence[str] | None = None,
    prefix: str = "",
    checkpoint: str | None = None,
    turn_context: TurnContext | None = None,
) -> FastAPI:
    """The API as an application, with the agent built during startup.

    ``build`` defaults to :func:`from_environment`. A caller supplying one owns
    whatever it builds against — including its own checkpointer, in which case
    ``checkpoint`` is not consulted, because nothing here asks for a saver
    unless the default factory does.

    ``origins`` defaults to ``MCP_AGENT_API_CORS_ORIGINS``. No origins means no
    CORS middleware rather than a middleware permitting nothing, so a
    same-origin deployment carries none of it.

    ``prefix`` is passed to :func:`~mcp_agent_api.routes.create_router`, for
    mounting the whole service under a path, and so is ``turn_context`` — a
    deployment that wants its runs traced needs that seam whether or not it
    owns the application around them.
    """
    checkpointing = Checkpointing(checkpoint)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Connect on the way up, and let go of it all on the way down.

        A failure here stops the process. That is the intended behaviour for a
        service whose entire purpose is the agent: a container that starts,
        reports healthy and answers ``503`` forever is harder to notice than
        one that refuses to start.
        """
        async with checkpointing:
            app.state.built = await (build or from_environment(checkpointing))()
            try:
                yield
            finally:
                app.state.built = None

    app = FastAPI(title="mcp-agent-api", lifespan=lifespan)
    if allowed := (cors_origins() if origins is None else list(origins)):
        app.add_middleware(
            CORSMiddleware,
            allow_origins=allowed,
            allow_credentials=True,
            allow_methods=["*"],
            # Credential headers are named by the connected toolsets and so are
            # not knowable here; the routes take only the ones a toolset
            # declared, which is where that filtering belongs.
            allow_headers=["*"],
        )

    def provider() -> Built:
        """The built agent, or a raise the routes turn into ``503``."""
        built = getattr(app.state, "built", None)
        if built is None:
            raise RuntimeError("the agent is still connecting")
        return built

    app.include_router(
        create_router(provider, prefix=prefix, turn_context=turn_context)
    )
    return app


#: The application ``uvicorn mcp_agent_api.app:app`` serves. Built at import,
#: which reads only the optional settings — see the module docstring.
app = create_app()
