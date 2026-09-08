"""Put the built web client into the distributions, when there is one to put.

``js/agent-ui`` builds into ``src/mcp_agent_api/ui``, which is generated and so
is not in version control — and hatchling's file selection follows version
control. A plain ``force-include`` entry would fix that and then fail the build
for anyone who has not run ``scripts/build-js``, which is everyone running
``uv build`` in a fresh clone.

So: include it when it exists, say nothing when it does not, and let
:mod:`mcp_agent_api.ui` be the one place that reports a missing client — at
mount, where the message can be about what to do. The release workflow builds
the client before ``uv build`` and then checks the wheel actually carries it,
which is the check that matters, because a wheel published without it would
install a chat with no page.

**Both targets, because ``uv build`` builds the wheel from the sdist.** Leaving
the client out of the sdist would leave it out of the wheel that came from one,
and the failure would be a published package rather than a build error.
"""

from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

#: Where the Vite build writes, relative to the project root.
CLIENT = Path("src") / "mcp_agent_api" / "ui"


class BuiltClientHook(BuildHookInterface):
    """Force-include the built client if ``scripts/build-js`` has run."""

    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        root = Path(self.root)
        client = root / CLIENT
        if not client.is_dir():
            return
        # A wheel is rooted at the package, an sdist at the project. The files
        # are the same either way; only what they are named inside differs.
        inside = root / "src" if self.target_name == "wheel" else root
        for path in sorted(client.rglob("*")):
            if path.is_file():
                build_data["force_include"][str(path)] = str(path.relative_to(inside))
