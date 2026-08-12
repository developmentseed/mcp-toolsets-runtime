"""``create_app``: the routes with a lifespan, a checkpointer and CORS around them.

The lifespan is the point of this module, so these run the application through
``httpx``'s ASGI transport *with* startup and shutdown, rather than mounting a
router at a request handler and never entering one.
"""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest

from mcp_agent_api.app import CORS_ORIGINS_VAR, cors_origins, create_app
from mcp_agent_api.routes import Built
from tests.mcp_agent_api.test_routes import _ask, _built


def _build(built: Built | None = None) -> Any:
    """A factory over an already-built agent, as a caller supplying one has."""
    agent = built if built is not None else _built()

    async def build() -> Built:
        return agent

    return build


@asynccontextmanager
async def _running(app: Any) -> AsyncIterator[httpx.AsyncClient]:
    """The app with its lifespan actually run, and a client onto it.

    ``httpx``'s ASGI transport speaks only the HTTP scope, so on its own it
    never starts anything — every test here would then be testing an
    application whose agent was never built. This drives the ``lifespan``
    scope by hand, which is all a server does, and is a smaller thing to own
    than a dependency for it.
    """
    inbox: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    outbox: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    scope = {"type": "lifespan", "asgi": {"version": "3.0", "spec_version": "2.0"}}
    task = asyncio.create_task(app(scope, inbox.get, outbox.put))

    await inbox.put({"type": "lifespan.startup"})
    if (await outbox.get())["type"] == "lifespan.startup.failed":
        await task  # Starlette re-raises whatever the lifespan raised.
        raise AssertionError("startup failed without an error")  # pragma: no cover
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://api"
        ) as client:
            yield client
    finally:
        await inbox.put({"type": "lifespan.shutdown"})
        await outbox.get()
        await task


async def test_the_agent_is_built_during_startup_and_served_after() -> None:
    started = []

    async def build() -> Built:
        started.append(True)
        return _built()

    app = create_app(build)
    assert not started, "the factory must not run at create_app time"

    async with _running(app) as client:
        assert started, "the factory must have run during the lifespan"
        response = await client.post("/runs", json=_ask())
        assert response.status_code == 200


async def test_a_request_before_startup_is_503_not_a_crash() -> None:
    """The honest answer while a lifespan is still connecting to MCP servers."""
    app = create_app(_build())
    # Deliberately not through `_running`, so `app.state.built` was never set
    # — which is the state a request racing a slow startup meets.
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://api"
    ) as client:
        response = await client.post("/runs", json=_ask())
    assert response.status_code == 503
    assert "still connecting" in response.json()["detail"]


async def test_a_factory_that_raises_stops_the_process() -> None:
    """Better a container that will not start than one that 503s forever."""

    async def build() -> Built:
        raise RuntimeError("no index at that URL")

    with pytest.raises(RuntimeError, match="no index at that URL"):
        async with _running(create_app(build)):
            pass  # pragma: no cover - the lifespan raises before this runs


async def test_the_agent_is_released_on_shutdown() -> None:
    app = create_app(_build())
    async with _running(app):
        assert app.state.built is not None
    assert app.state.built is None


def test_the_routes_are_all_mounted() -> None:
    """Read off the schema rather than ``app.routes``, which is not flat."""
    assert set(create_app(_build()).openapi()["paths"]) == {
        "/runs",
        "/threads/{thread_id}",
        "/threads/{thread_id}/state/{key}",
        "/views/{toolset}/{view}",
    }


def test_a_prefix_moves_every_route() -> None:
    assert set(create_app(_build(), prefix="/v1").openapi()["paths"]) == {
        "/v1/runs",
        "/v1/threads/{thread_id}",
        "/v1/threads/{thread_id}/state/{key}",
        "/v1/views/{toolset}/{view}",
    }


async def test_no_origins_means_no_cors_middleware() -> None:
    """A same-origin deployment should carry none of it."""
    app = create_app(_build(), origins=[])
    assert not any(
        "CORSMiddleware" in str(middleware) for middleware in app.user_middleware
    )


async def test_an_origin_is_answered_with_the_cors_header() -> None:
    app = create_app(_build(), origins=["https://ui.example"])
    async with _running(app) as client:
        response = await client.post(
            "/runs",
            json=_ask(),
            headers={"Origin": "https://ui.example"},
        )
        assert response.headers["access-control-allow-origin"] == "https://ui.example"


async def test_a_preflight_is_answered_before_the_agent_exists() -> None:
    """CORS is middleware, so it runs whether or not the lifespan has finished."""
    app = create_app(_build(), origins=["https://ui.example"])
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://api"
    ) as client:
        response = await client.options(
            "/runs",
            headers={
                "Origin": "https://ui.example",
                "Access-Control-Request-Method": "POST",
            },
        )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://ui.example"


def test_origins_are_read_from_a_comma_separated_string() -> None:
    assert cors_origins("https://a, https://b") == ["https://a", "https://b"]
    assert cors_origins("") == []
    assert cors_origins(" , ") == []


def test_origins_fall_back_to_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(CORS_ORIGINS_VAR, "https://from-env")
    assert cors_origins() == ["https://from-env"]


def test_creating_the_app_needs_no_provider_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Why a module-level ``app`` is safe to import.

    ``PROVIDER_MODEL`` and ``PROVIDER_API_KEY`` are required by
    ``AgentSettings``, and the default factory reads them — but not until the
    lifespan runs.
    """
    monkeypatch.delenv("PROVIDER_MODEL", raising=False)
    monkeypatch.delenv("PROVIDER_API_KEY", raising=False)
    assert create_app() is not None
