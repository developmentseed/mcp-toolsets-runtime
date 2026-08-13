"""The `build` factory `create_app` awaits during startup.

This is the seam a deployment writes for itself. `mcp_agent_api.app` defaults
to one calling the runtime's `build_agent` against an index URL; anything with
its own sequence supplies this instead — dss reimplements the whole thing for
Langfuse tracing and skill validation, and this example does it to stand up its
own MCP servers first.

What comes back is read by attribute, never by position: the routes want
``.agent``, ``.connections``, ``.tools``, ``.withheld`` and ``.required``, and
the runtime's `BuiltAgent` and dss's carry those five in different orders.
"""

import logging

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver

from mcp_agent.main import BuiltAgent, with_session_state
from service import model, servers

logger = logging.getLogger(__name__)


async def build() -> BuiltAgent:
    """Connect to the MCP servers and wire session state into an agent.

    Called once, inside the lifespan, because connecting is async and the app
    is constructed at import. Until it returns, every route answers 503.
    """
    connections = await servers.start()
    logger.info("connected %d MCP server(s)", len(connections))

    tools = await MultiServerMCPClient(connections).get_tools()
    chat, named = model.build()

    # `with_session_state` is the three-piece pattern docs/CONSUMING.md
    # documents: the capture middleware, `bind_all_injected` reading back out,
    # and `inspect_state` for a value the model was only told the key of. It
    # also withholds any tool whose required parameter nothing can fill.
    agent, withheld = with_session_state(chat, tools, InMemorySaver())
    logger.info(
        "built on %s: %d tool(s), %d withheld", named, len(tools), len(withheld)
    )

    # An in-process checkpointer, so threads live as long as this process. A
    # deployment passes a configured saver — `mcp_agent.main.Checkpointing`
    # builds one from MCP_AGENT_CHECKPOINT, PostgreSQL included.
    return BuiltAgent(agent, connections, tools, withheld, None)
