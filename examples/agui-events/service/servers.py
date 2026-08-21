"""The MCP servers the agent connects to — this example's stand-in for an index.

A deployment does not do this. It points at an already-running index and
discovers what is behind it; `mcp_agent.main.build_agent` takes that URL. This
module exists so the example is one command instead of five, and it is the only
part of `service/` that a real service would not have.

Four servers on ephemeral ports:

``dataset-search``
    publishes a 38 kB area of interest as a `ToolResult` data key
``raster-ops``
    takes one by handle, and carries a ``ui://`` view
``contour-ops``
    takes a value nothing here publishes, so its calls are refused
``terrain``
    a raw FastMCP server that declares nothing at all
"""

import asyncio
import socket
import threading
from typing import Any

import uvicorn

from mcp_runtime.server import build_server

#: toolset name -> module, for the servers built from this runtime.
TOOLSETS = {
    "dataset-search": "dataset_search.tools",
    "raster-ops": "clip_view.tools",  # the example's clip_raster, plus a view
    "contour-ops": "contour_ops.tools",  # nothing here publishes what it takes
}

#: The one server with no idea this project exists.
FOREIGN = "terrain"


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _serve(app: Any, port: int) -> None:
    """Run one ASGI app on a daemon thread, so the caller keeps its loop."""
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    threading.Thread(target=uvicorn.Server(config).run, daemon=True).start()


async def _wait_for(port: int, attempts: int = 80) -> None:
    for _ in range(attempts):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                return
        except OSError:
            await asyncio.sleep(0.1)
    raise RuntimeError(f"127.0.0.1:{port} never came up")


async def start() -> dict[str, dict[str, str]]:
    """Start all four and return connections, in `MultiServerMCPClient`'s shape.

    The return value is what a deployment would have got from its index, so
    everything downstream of here is what a deployment does.
    """
    from foreign_server import mcp as foreign_mcp

    ports = {name: free_port() for name in [*TOOLSETS, FOREIGN]}
    for toolset, module in TOOLSETS.items():
        server = build_server(
            toolset=toolset,
            module_name=module,
            host="127.0.0.1",
            port=ports[toolset],
        )
        _serve(server.streamable_http_app(), ports[toolset])
    _serve(foreign_mcp.streamable_http_app(), ports[FOREIGN])

    for port in ports.values():
        await _wait_for(port)

    return {
        name: {"transport": "streamable_http", "url": f"http://127.0.0.1:{port}/mcp"}
        for name, port in ports.items()
    }
