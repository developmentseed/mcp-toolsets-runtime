"""Serving the bundled web client: the page, its configuration, its assets.

These build a bundle of their own in ``tmp_path`` rather than reading the real
one. The real one exists only where Node has run, which is CI's ``js`` job and
the release workflow — the Python job has no Node, and a test that skipped
itself there would cover nothing on the platform that publishes the wheel.
"""

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from mcp_agent_api.ui import (
    CONFIG_ELEMENT_ID,
    ENV_PREFIX,
    UiConfig,
    available,
    config_from_environment,
    create_ui_router,
    mount_ui,
    render_index,
)

#: Big enough to be worth compressing, and compressible.
SCRIPT = b"console.log('hello');\n" * 200


def _bundle(root: Path, *, config_element: bool = True) -> Path:
    """A built client, as Vite leaves one."""
    element = (
        f'<script id="{CONFIG_ELEMENT_ID}" type="application/json">'
        '{ "api": "/api" }</script>'
        if config_element
        else ""
    )
    root.mkdir(parents=True, exist_ok=True)
    (root / "index.html").write_text(
        "<!doctype html><html><head>"
        f"{element}"
        '<script type="module" src="./assets/app-abc123.js"></script>'
        "</head><body><div id=root></div></body></html>"
    )
    (root / "assets").mkdir(exist_ok=True)
    (root / "assets" / "app-abc123.js").write_bytes(SCRIPT)
    (root / "assets" / "tiny.css").write_text("b{color:red}")
    return root


def _client(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://ui"
    )


def _app(root: Path, **kwargs: Any) -> FastAPI:
    app = FastAPI()
    app.include_router(create_ui_router(root=root, **kwargs))
    return app


# --- the page --------------------------------------------------------------


async def test_the_page_carries_the_deployments_configuration(tmp_path: Path):
    """The whole point of rewriting the element rather than fetching it: the
    first render already has the real title.
    """
    root = _bundle(tmp_path / "ui")
    app = _app(root, config=UiConfig(api="", title="Rainfall", examples=("try me",)))

    async with _client(app) as client:
        page = (await client.get("/")).text

    written = page.split(f'id="{CONFIG_ELEMENT_ID}"')[1].split(">")[1].split("<")[0]
    assert json.loads(written) == {
        "api": "",
        "title": "Rainfall",
        "tagline": "",
        "greeting": "",
        "examples": ["try me"],
        "accent": "",
    }


async def test_the_page_is_never_cached(tmp_path: Path):
    """It names this build's hashed assets, and a deploy removes the old ones."""
    app = _app(_bundle(tmp_path / "ui"))
    async with _client(app) as client:
        response = await client.get("/")
    assert response.headers["cache-control"] == "no-store"


async def test_a_bundle_without_the_element_is_still_served(tmp_path: Path):
    """An index.html from somewhere else gets the client's own defaults rather
    than a failure: this rewrites a page, it does not require one.
    """
    app = _app(_bundle(tmp_path / "ui", config_element=False))
    async with _client(app) as client:
        response = await client.get("/")
    assert response.status_code == 200
    assert CONFIG_ELEMENT_ID not in response.text


def test_only_the_first_configuration_element_is_rewritten():
    """A second one would be the bundle's own JSON, not ours."""
    html = (
        f'<script id="{CONFIG_ELEMENT_ID}" type="application/json">{{}}</script>'
        '<script id="other" type="application/json">{"keep": true}</script>'
    )
    out = render_index(html, UiConfig(title="One"))
    assert '{"keep": true}' in out
    assert "One" in out


# --- assets ----------------------------------------------------------------


async def test_an_asset_is_compressed_for_a_browser_that_takes_it(tmp_path: Path):
    """Compressed here because one of the two deployment shapes this runtime
    targets — an AWS load balancer — will not do it, and the bundle is half a
    megabyte.
    """
    app = _app(_bundle(tmp_path / "ui"))
    async with _client(app) as client:
        response = await client.get(
            "/assets/app-abc123.js", headers={"accept-encoding": "gzip"}
        )

    # httpx decodes the body on the way in, so what proves the wire was
    # compressed is the header and the length that came with it.
    assert response.headers["content-encoding"] == "gzip"
    assert int(response.headers["content-length"]) < len(SCRIPT)
    assert response.content == SCRIPT
    assert response.headers["vary"] == "Accept-Encoding"


async def test_an_asset_is_sent_whole_to_a_browser_that_does_not(tmp_path: Path):
    app = _app(_bundle(tmp_path / "ui"))
    async with _client(app) as client:
        response = await client.get(
            "/assets/app-abc123.js", headers={"accept-encoding": "identity"}
        )
    assert "content-encoding" not in response.headers
    assert response.content == SCRIPT


async def test_a_small_asset_is_not_compressed(tmp_path: Path):
    """Below a kilobyte the header costs more than the compression saves."""
    app = _app(_bundle(tmp_path / "ui"))
    async with _client(app) as client:
        response = await client.get(
            "/assets/tiny.css", headers={"accept-encoding": "gzip"}
        )
    assert "content-encoding" not in response.headers


async def test_assets_are_cached_forever(tmp_path: Path):
    """The filenames are content-hashed, so a changed asset is a changed URL."""
    app = _app(_bundle(tmp_path / "ui"))
    async with _client(app) as client:
        response = await client.get("/assets/app-abc123.js")
    assert "immutable" in response.headers["cache-control"]


async def test_an_unknown_asset_is_a_404_and_not_a_path(tmp_path: Path):
    """The names come from a listing made at mount, so nothing a request says
    is ever joined onto a path.
    """
    app = _app(_bundle(tmp_path / "ui"))
    async with _client(app) as client:
        assert (await client.get("/assets/nothing.js")).status_code == 404
        escape = await client.get("/assets/..%2f..%2fetc%2fpasswd")
        assert escape.status_code == 404


# --- mounting --------------------------------------------------------------


async def test_it_can_be_mounted_under_a_path(tmp_path: Path):
    app = _app(_bundle(tmp_path / "ui"), path="/chat")
    async with _client(app) as client:
        assert (await client.get("/chat/")).status_code == 200
        assert (await client.get("/chat/assets/tiny.css")).status_code == 200


async def test_a_bare_prefix_redirects_to_the_page(tmp_path: Path):
    """The bundle's asset URLs are relative, so ``/chat`` would resolve them
    against ``/`` and 404 every one of them.
    """
    app = _app(_bundle(tmp_path / "ui"), path="/chat")
    async with _client(app) as client:
        response = await client.get("/chat")
    assert response.status_code == 308
    assert response.headers["location"] == "/chat/"


async def test_mounting_leaves_the_api_routes_alone(tmp_path: Path):
    """Mounted at the root beside the agent's own routes, which is the default
    deployment: two exact paths, no wildcard, nothing shadowed.
    """
    app = FastAPI()

    @app.post("/runs")
    async def runs() -> dict[str, str]:
        return {"ran": "yes"}

    mount_ui(app, root=_bundle(tmp_path / "ui"))

    async with _client(app) as client:
        assert (await client.post("/runs")).json() == {"ran": "yes"}
        assert (await client.get("/")).status_code == 200


def test_there_is_nothing_to_mount_without_a_build(tmp_path: Path):
    """The one place a missing client is reported, and it says what to run."""
    assert not available(tmp_path / "nothing")
    with pytest.raises(RuntimeError, match="build-js"):
        create_ui_router(root=tmp_path / "nothing")


# --- configuration ---------------------------------------------------------


def test_examples_can_be_written_as_lines(monkeypatch: pytest.MonkeyPatch):
    """A Helm values file writes a list of questions as a block scalar far more
    readably than as embedded JSON, and both end up in one variable.
    """
    monkeypatch.setenv(f"{ENV_PREFIX}EXAMPLES", "  what is here?\n\nclip it\n")
    assert config_from_environment().examples == ("what is here?", "clip it")


def test_examples_can_be_written_as_json(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(f"{ENV_PREFIX}EXAMPLES", '["one", "two"]')
    assert config_from_environment().examples == ("one", "two")


def test_the_environment_configures_the_client(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(f"{ENV_PREFIX}TITLE", "DevSeed")
    monkeypatch.setenv(f"{ENV_PREFIX}ACCENT", "#123456")
    config = config_from_environment(api="/api")
    assert (config.title, config.accent, config.api) == ("DevSeed", "#123456", "/api")
    # Unset fields keep the client's own defaults rather than becoming empty.
    assert config.tagline == ""


# --- the application around it ---------------------------------------------


async def test_the_whole_service_serves_the_client_when_it_has_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """``uvicorn mcp_agent_api.app:app`` is meant to be a working chat, not an
    API waiting for someone to write a front end.
    """
    monkeypatch.setattr("mcp_agent_api.ui.UI_ROOT", _bundle(tmp_path / "ui"))
    from mcp_agent_api.app import create_app

    async with _client(create_app(_never_built)) as client:
        assert (await client.get("/")).status_code == 200


async def test_an_api_deployment_can_turn_the_client_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("mcp_agent_api.ui.UI_ROOT", _bundle(tmp_path / "ui"))
    from mcp_agent_api.app import create_app

    async with _client(create_app(_never_built, ui=False)) as client:
        assert (await client.get("/")).status_code == 404


def test_asking_for_a_client_that_is_not_installed_fails_at_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """``ui=True`` is a deployment saying the page is the point. Better to
    refuse to start than to serve a 404 as a home page.
    """
    monkeypatch.setattr("mcp_agent_api.ui.UI_ROOT", tmp_path / "nothing")
    from mcp_agent_api.app import create_app

    with pytest.raises(RuntimeError, match="build-js"):
        create_app(_never_built, ui=True)


async def test_the_client_is_pointed_at_the_prefix_the_routes_moved_to(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A deployment that mounts the API under ``/api`` and the page at the root
    is one application, and the page has to know which half is which.
    """
    monkeypatch.setattr("mcp_agent_api.ui.UI_ROOT", _bundle(tmp_path / "ui"))
    from mcp_agent_api.app import create_app

    async with _client(create_app(_never_built, prefix="/api")) as client:
        page = (await client.get("/")).text

    written = page.split(f'id="{CONFIG_ELEMENT_ID}"')[1].split(">")[1].split("<")[0]
    assert json.loads(written)["api"] == "/api"


async def test_liveness_answers_before_the_agent_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The two probes fail differently, which is why there are two: readiness
    is 503 while the lifespan connects, and restarting for that would be wrong.
    """
    monkeypatch.setattr("mcp_agent_api.ui.UI_ROOT", tmp_path / "nothing")
    from mcp_agent_api.app import create_app

    app = create_app(_never_built)
    async with _client(app) as client:
        assert (await client.get("/health/liveness")).status_code == 200
        assert (await client.get("/health/readiness")).status_code == 503


async def _never_built() -> Any:
    """A factory the tests here never let run: no lifespan is driven."""
    raise AssertionError("the lifespan should not have run")  # pragma: no cover


async def test_the_page_and_its_assets_answer_head(tmp_path: Path):
    """A GET-only route 405s a HEAD, and the things that send one — uptime
    checks, caching proxies, link checkers — report that as the page being
    broken. Starlette's own static files answer it; FastAPI's `get` does not.
    """
    app = _app(_bundle(tmp_path / "ui"))
    async with _client(app) as client:
        page = await client.head("/")
        asset = await client.head("/assets/app-abc123.js")

    assert (page.status_code, asset.status_code) == (200, 200)
    assert page.headers["cache-control"] == "no-store"
    assert "immutable" in asset.headers["cache-control"]
