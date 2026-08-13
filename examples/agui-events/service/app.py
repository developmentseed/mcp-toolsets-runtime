"""The application: `create_app`, plus the routes an operator needs.

`create_app` gives you the agent's four routes, a lifespan that builds it, and
CORS. Everything a deployment adds on top goes here — which for this example is
the pair of health probes any orchestrator asks for, and is the point worth
copying: the runtime hands you a `FastAPI`, and it is still yours.
"""

import logging

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from mcp_agent_api.app import cors_origins, create_app
from service.agent import build
from service.settings import get_settings

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(name)s: %(message)s")
for noisy in ("mcp", "httpx", "uvicorn.access", "sse_starlette", "langchain"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

settings = get_settings()

app: FastAPI = create_app(build, origins=cors_origins(settings.allowed_origins))


@app.get("/health/liveness", tags=["Health"])
def liveness() -> JSONResponse:
    """Is the process up. Answers before the agent exists, and must."""
    return JSONResponse({"status": "alive"})


@app.get("/health/readiness", tags=["Health"])
def readiness() -> JSONResponse:
    """Is the agent built and serving.

    Separate from liveness because they fail differently: this one is 503 for
    as long as the lifespan is still connecting to the MCP servers, and an
    orchestrator that conflated the two would restart a process that was doing
    nothing wrong.
    """
    if getattr(app.state, "built", None) is None:
        return JSONResponse({"status": "connecting"}, status_code=503)
    return JSONResponse({"status": "ready"})
