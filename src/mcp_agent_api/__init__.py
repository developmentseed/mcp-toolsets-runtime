"""The agent `mcp_agent` builds, over HTTP.

Three layers, each importable without the one above it:

``mcp_agent_api.events``
    A pure async generator turning one turn into AG-UI events. Imports no
    FastAPI — for a consumer with its own transport or framework.

Layers above this one (an ``APIRouter``, and an app factory) arrive in later
releases. Needs the ``[api]`` extra.
"""

from mcp_agent_api.events import (
    ANSWER_CITATIONS,
    MCP_VIEW,
    STATE_CONSUMED,
    STATE_PUBLISHED,
    TOOLS_WITHHELD,
    agui_events,
    state_metadata,
)

__all__ = [
    "ANSWER_CITATIONS",
    "MCP_VIEW",
    "STATE_CONSUMED",
    "STATE_PUBLISHED",
    "TOOLS_WITHHELD",
    "agui_events",
    "state_metadata",
]
