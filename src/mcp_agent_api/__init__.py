"""The agent `mcp_agent` builds, over HTTP.

Three layers, each importable without the one above it:

``mcp_agent_api.events``
    A pure async generator turning one turn into AG-UI events. Imports no
    FastAPI — for a consumer with its own transport or framework.

``mcp_agent_api.routes``
    An ``APIRouter`` over a built agent, for a deployment bringing its own
    FastAPI application. Its response models are exported here too, so a Python
    client can validate against the shapes the routes document rather than
    re-declaring them: :class:`~mcp_agent_api.routes.ThreadResponse`,
    :class:`~mcp_agent_api.routes.TurnsResponse`,
    :class:`~mcp_agent_api.routes.StateValueResponse` and the
    :class:`~mcp_agent_api.routes.StateEntryInfo` all three share.

``mcp_agent_api.app``
    ``create_app`` and a module-level ``app``: the lifespan, the checkpointer
    and CORS around those routes, for a deployment that wants the service
    rather than the parts. Not re-exported here — importing it builds an
    application, which a consumer of the layers below should not pay for.

Needs the ``[api]`` extra.
"""

from mcp_agent_api.events import (
    ANSWER_CITATIONS,
    MCP_VIEW,
    STATE_CONSUMED,
    STATE_NAMESPACE,
    STATE_PUBLISHED,
    TOOLS_WITHHELD,
    agui_events,
    state_metadata,
)
from mcp_agent_api.routes import (
    Built,
    RunRequest,
    StateEntryInfo,
    StateValueResponse,
    ThreadResponse,
    TurnContext,
    TurnInfo,
    TurnsResponse,
    create_router,
)

__all__ = [
    "ANSWER_CITATIONS",
    "MCP_VIEW",
    "STATE_CONSUMED",
    "STATE_NAMESPACE",
    "STATE_PUBLISHED",
    "TOOLS_WITHHELD",
    "Built",
    "RunRequest",
    "StateEntryInfo",
    "StateValueResponse",
    "ThreadResponse",
    "TurnContext",
    "TurnInfo",
    "TurnsResponse",
    "agui_events",
    "create_router",
    "state_metadata",
]
