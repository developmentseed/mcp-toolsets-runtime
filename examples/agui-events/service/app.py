"""The application: `create_app`, and what a deployment adds around it.

`create_app` gives you the agent's six routes, a lifespan that builds it, CORS,
the two health probes an orchestrator asks for, and — when the installation has
one — the bundled web client at the root. Which is nearly everything, and the
point worth copying: the runtime hands you a `FastAPI`, and it is still yours.

This example adds one thing to it, and it is about the example rather than
about deployments: `MCP_AGENT_UI_*` defaults describing the four toolsets
`service/servers.py` starts, so the page this process serves on :8765 opens
with the questions the README talks about instead of a blank box. A deployment
sets those in its container; they are in this file because the toolsets are.

Serving the client under `npm run dev` is Vite's job, and Vite does not run
that rewrite — so the development page opens on what `GET /connections` says
is connected, which is the fallback every deployment gets before it configures
anything.
"""

import logging
import os

from fastapi import FastAPI

from mcp_agent_api.app import cors_origins, create_app
from service.agent import build
from service.settings import get_settings

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(name)s: %(message)s")
for noisy in ("mcp", "httpx", "uvicorn.access", "sse_starlette", "langchain"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

settings = get_settings()

#: What the four servers in `service/servers.py` can actually answer. A real
#: deployment sets these in its container; this one has them in the file
#: because the toolsets are in the file too.
os.environ.setdefault("MCP_AGENT_UI_TITLE", "mcp_agent_api")
os.environ.setdefault(
    "MCP_AGENT_UI_GREETING",
    "Four MCP servers are connected: dataset search, raster clipping with a "
    "ui:// view, contour smoothing this deployment cannot offer, and a "
    "third-party server that knows nothing about any of this.",
)
os.environ.setdefault(
    "MCP_AGENT_UI_EXAMPLES",
    "\n".join(
        [
            "find rainfall datasets and clip chirps to that area",
            "smooth the contours",
            "sketch a rough boundary around the Severn catchment and call it Severn",
            "a few paragraphs about anything",
        ]
    ),
)

app: FastAPI = create_app(build, origins=cors_origins(settings.allowed_origins))
