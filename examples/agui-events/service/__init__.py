"""The example's backend, laid out the way a deployment of this runtime is.

Four modules, each the counterpart of something a real service has:

``settings``
    One `BaseSettings` and a cached accessor, read once at import.
``servers``
    The MCP servers the agent connects to. **This is the example's own
    scaffolding** — a deployment points `MCP_URL` at a running index instead,
    and this module is what stands in for one.
``model``
    The chat model, from `PROVIDER_MODEL` and `PROVIDER_API_KEY`.
``agent``
    The `build` factory `create_app` awaits during startup.
``app``
    `create_app` plus the operational routes a deployment adds around it, and
    the module-level `app` uvicorn serves.

Run it with ``python -m service`` from this directory, or
``uvicorn service.app:app``.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXAMPLE = HERE.parent
REPO = EXAMPLE.parents[1]
SESSION_STATE = REPO / "examples" / "session-state"

# Two of the four servers are examples/session-state's, which is a directory
# rather than a package — neither it nor the toolsets under it are importable
# until they are on the path. The two servers this example adds sit beside it.
# Done here so it has happened before any submodule imports them.
sys.path[:0] = [
    str(SESSION_STATE),
    str(SESSION_STATE / "toolsets"),
    str(EXAMPLE / "toolsets"),
]
