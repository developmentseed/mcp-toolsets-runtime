"""Keeping a served agent's tools and views current.

The agent is built at startup from the tools the toolsets listed then, and a
toolset redeployed after that would reach the model with its old descriptions
until the API restarted. :func:`keep_fresh` builds the agent again every
``MCP_AGENT_API_REFRESH_SECONDS`` and swaps it in when what the model sees of
the tools has changed. :class:`~mcp_agent_api.routes.ViewCache` re-reads the
view bundles on the same interval.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import FastAPI

    from mcp_agent_api.routes import Built

logger = logging.getLogger(__name__)

#: Seconds between refreshes. ``0`` never refreshes: the agent and the views
#: are read once, at startup and on first use.
REFRESH_VAR = "MCP_AGENT_API_REFRESH_SECONDS"
DEFAULT_REFRESH_SECONDS = 30 * 60

#: Builds the agent the routes serve. Async because connecting to MCP servers is.
Builder = Callable[[], Awaitable["Built"]]


def refresh_seconds(value: str | None = None) -> float | None:
    """The interval from a string or the environment, or ``None`` for never."""
    raw = os.environ.get(REFRESH_VAR, "") if value is None else value
    seconds = float(raw) if raw.strip() else DEFAULT_REFRESH_SECONDS
    return seconds if seconds > 0 else None


def fingerprint(built: Built) -> str:
    """What the model sees of the tools: their names, descriptions and arguments."""
    tools = [(t.name, t.description, t.args) for t in built.tools]
    return hashlib.sha256(json.dumps(tools, default=str).encode()).hexdigest()


async def keep_fresh(app: FastAPI, build: Builder, every: float) -> None:
    """Rebuild ``app.state.built`` every ``every`` seconds, when its tools changed.

    Run as a task beside the lifespan's ``yield``. A turn already running keeps
    the agent it started with. A failed build keeps the current agent, so a
    toolset that is down for a moment does not take the others with it.
    """
    seen = fingerprint(app.state.built)
    while True:
        await asyncio.sleep(every)
        try:
            built = await build()
        except Exception:
            logger.warning(
                "tool refresh failed; keeping the current agent", exc_info=True
            )
            continue
        if (latest := fingerprint(built)) != seen:
            app.state.built, seen = built, latest
            logger.info("the toolsets' tools changed; agent rebuilt")
