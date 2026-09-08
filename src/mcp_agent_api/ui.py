"""The bundled web client, as routes you mount beside the agent's own.

The wheel carries a built single-page client for :mod:`mcp_agent_api.routes` —
the transcript, the tool calls and receipts as they happen, the session-state
panel, and the ``ui://`` views in their frames. A deployment that wants a
usable chat installs the ``[api]`` extra and serves it; nothing else is needed,
and no Node runs anywhere near the image.

Two ways in, mirroring the split between :mod:`~mcp_agent_api.routes` and
:mod:`~mcp_agent_api.app`::

    # the whole service, UI included
    from mcp_agent_api.app import create_app
    app = create_app()

    # an application of your own
    from mcp_agent_api.ui import mount_ui
    mount_ui(my_app, api="/api")

**The client is tied to the routes, not to this application.** It speaks the
six routes in :mod:`~mcp_agent_api.routes` and the AG-UI wire in
:mod:`~mcp_agent_api.events`, and nothing else. So a host that mounts
``create_router`` into its own FastAPI application — its own auth, its own
middleware, a different path — serves this same client by pointing ``api`` at
wherever it mounted them. What forces a fork is diverging from the *routes*,
not from the application around them.

**Configuration reaches the browser in the page, not in a second request.**
``index.html`` carries a ``<script type="application/json">`` element whose
contents this module rewrites at mount time. A client that fetched its own
configuration would render once with placeholder text and again with the real
thing, and every deployment would wear the flash.

**Assets are compressed here rather than at the edge.** The bundle is around
half a megabyte and compresses to under a third of that, and one of the two
deployment shapes this runtime targets — an AWS application load balancer —
cannot compress a response at all. Doing it once at mount, in memory, is a
smaller thing to own than a build-time compression plugin and works wherever
the image runs. It is deliberately *not* ``GZipMiddleware``: that would also
compress ``POST /runs``, which is a token stream, and buffering a token stream
is the one thing a chat must not do.
"""

import gzip
import json
import mimetypes
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response

#: Where the built client lands. Written by ``js/agent-ui``'s Vite build and
#: force-included into the wheel; absent from a checkout nobody has built.
UI_ROOT = Path(__file__).parent / "ui"

#: The element ``index.html`` reserves for configuration, rewritten on mount.
#: Matched by id rather than by position: a bundler is free to reorder the
#: head, and a regex over "the first script tag" would eventually take the
#: bundle's own.
CONFIG_ELEMENT_ID = "mcp-agent-ui-config"

_CONFIG_ELEMENT = re.compile(
    rf'(<script[^>]*\bid="{CONFIG_ELEMENT_ID}"[^>]*>)(.*?)(</script>)',
    re.DOTALL,
)

#: Prefix of the environment variables :func:`config_from_environment` reads.
ENV_PREFIX = "MCP_AGENT_UI_"

#: How long a browser may keep an asset. The filenames are content-hashed by
#: the bundler, so a changed asset is a changed URL and this can be forever.
IMMUTABLE = "public, max-age=31536000, immutable"

#: Below this, compressing costs more than it saves.
COMPRESS_OVER = 1024


@dataclass(frozen=True)
class UiConfig:
    """What a deployment gets to say about the client without rebuilding it.

    Text and one colour. Anything structural — a different layout, a map
    beside the transcript — is a change to the client, and the source is in
    this repository under ``js/agent-ui`` to be forked or contributed to.

    ``api`` is the only field the client cannot run without: the base it
    prefixes onto the six routes. Empty means same origin at the root, which
    is what :func:`~mcp_agent_api.app.create_app` serves.
    """

    api: str = ""
    title: str = "MCP Toolsets"
    #: One line under the title. Empty renders nothing rather than a gap.
    tagline: str = ""
    #: The opening screen's paragraph. Empty leaves the client to describe
    #: what is connected from ``GET /connections``, which is usually better.
    greeting: str = ""
    #: Questions offered as buttons on the opening screen. A deployment knows
    #: what its toolsets can answer; the client cannot guess it.
    examples: tuple[str, ...] = ()
    #: A CSS colour for the accent. Empty keeps the client's own.
    accent: str = ""

    def as_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))


def _examples(raw: str) -> tuple[str, ...]:
    """Example questions from one environment variable.

    A JSON array where the value parses as one, and otherwise one question per
    line — because a Helm values file writes a list of strings as a block
    scalar far more readably than as embedded JSON.
    """
    raw = raw.strip()
    if not raw:
        return ()
    if raw.startswith("["):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return tuple(str(item) for item in parsed)
    return tuple(line.strip() for line in raw.splitlines() if line.strip())


def config_from_environment(api: str = "") -> UiConfig:
    """A :class:`UiConfig` from ``MCP_AGENT_UI_*``, for a deployment with no code.

    Both deployment targets configure a container the same way, so the whole
    surface is environment variables: ``MCP_AGENT_UI_TITLE``, ``_TAGLINE``,
    ``_GREETING``, ``_EXAMPLES`` and ``_ACCENT``.
    """
    defaults = UiConfig()

    def read(name: str, fallback: str) -> str:
        return os.environ.get(f"{ENV_PREFIX}{name}", fallback)

    return UiConfig(
        api=api,
        title=read("TITLE", defaults.title),
        tagline=read("TAGLINE", defaults.tagline),
        greeting=read("GREETING", defaults.greeting),
        examples=_examples(read("EXAMPLES", "")),
        accent=read("ACCENT", defaults.accent),
    )


def available(root: Path | None = None) -> bool:
    """Whether a built client is in this installation.

    False in a checkout where ``scripts/build-js`` has not run, and in a wheel
    built without Node. Callers mounting it optionally check this rather than
    catching the error.
    """
    return (root or UI_ROOT).joinpath("index.html").is_file()


@dataclass
class _Asset:
    """One built file, held in memory with its compressed form beside it."""

    body: bytes
    media_type: str
    compressed: bytes | None = None

    def response(self, accept_encoding: str) -> Response:
        if self.compressed is not None and "gzip" in accept_encoding:
            return Response(
                self.compressed,
                media_type=self.media_type,
                headers={
                    "Content-Encoding": "gzip",
                    "Cache-Control": IMMUTABLE,
                    # Without this a shared cache may hand the compressed body
                    # to a client that never asked for one.
                    "Vary": "Accept-Encoding",
                },
            )
        return Response(
            self.body,
            media_type=self.media_type,
            headers={"Cache-Control": IMMUTABLE, "Vary": "Accept-Encoding"},
        )


def _load_assets(root: Path) -> dict[str, _Asset]:
    """Every file under ``assets/``, read and compressed once.

    A dictionary keyed by filename rather than a path join per request: the
    set is fixed at build time, so a name that is not in it is a 404 and no
    request can reach outside the directory whatever it asks for.
    """
    assets: dict[str, _Asset] = {}
    directory = root / "assets"
    if not directory.is_dir():
        return assets
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        body = path.read_bytes()
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        compressed = gzip.compress(body, 9) if len(body) > COMPRESS_OVER else None
        # A compressed form that is no smaller is a waste of a header.
        if compressed is not None and len(compressed) >= len(body):
            compressed = None
        assets[path.name] = _Asset(body, media_type, compressed)
    return assets


def render_index(html: str, config: UiConfig) -> str:
    """``index.html`` with the deployment's configuration written into it.

    The element is left alone if it is not there — an ``index.html`` from
    somewhere else is still served, it simply gets the client's own defaults.
    """
    return _CONFIG_ELEMENT.sub(
        lambda match: match.group(1) + config.as_json() + match.group(3), html, count=1
    )


def create_ui_router(
    config: UiConfig | None = None,
    *,
    path: str = "/",
    root: Path | None = None,
) -> APIRouter:
    """Routes serving the built client: the page, and its assets.

    Two routes and no wildcard, which is what lets this be mounted at ``/``
    beside the agent's own routes without shadowing them. The client keeps its
    state in the query string rather than in paths, so there is no history
    fallback to serve.

    ``path`` mounts it somewhere other than the root. The bundle references its
    assets relatively, so the page must be reached with a trailing slash for
    them to resolve; a request without one is redirected rather than served a
    page whose assets 404.
    """
    root = root or UI_ROOT
    if not available(root):
        raise RuntimeError(
            f"no built client at {root}: run scripts/build-js, or install a "
            "wheel built with one (see mcp_agent_api.ui)"
        )

    config = config or UiConfig()
    index = render_index((root / "index.html").read_text(), config)
    assets = _load_assets(root)

    path = "/" + path.strip("/")
    page = path if path == "/" else path + "/"
    router = APIRouter()

    @router.get(page, include_in_schema=False)
    async def read_index() -> HTMLResponse:
        """The page. Never cached: it names the hashed assets of *this* build,
        and a stale one names files that a deploy has already removed.
        """
        return HTMLResponse(index, headers={"Cache-Control": "no-store"})

    if page != path:

        @router.get(path, include_in_schema=False)
        async def redirect_to_page() -> RedirectResponse:
            return RedirectResponse(page, status_code=308)

    @router.get(page + "assets/{name}", include_in_schema=False)
    async def read_asset(name: str, request: Request) -> Response:
        """One built asset, compressed if the browser said it could take it."""
        asset = assets.get(name)
        if asset is None:
            raise HTTPException(404, f"no asset {name}")
        return asset.response(request.headers.get("accept-encoding", ""))

    return router


def mount_ui(
    app: FastAPI,
    *,
    api: str = "",
    path: str = "/",
    config: UiConfig | None = None,
    root: Path | None = None,
) -> None:
    """Serve the built client from ``app``.

    ``api`` is where this application serves the agent's routes, as the browser
    reaches them — ``""`` for the root of the same origin, ``"/api"`` behind a
    proxy that prefixes them. ``config`` defaults to
    :func:`config_from_environment`, which is how a container is configured.
    """
    app.include_router(
        create_ui_router(config or config_from_environment(api), path=path, root=root)
    )
