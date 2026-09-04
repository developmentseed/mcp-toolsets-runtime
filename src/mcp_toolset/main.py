"""Scaffold a new toolset: ``mcp-toolset new [--with-ui] <name>`` (kebab-case).

Run it from a consumer repo's root (one that installs ``mcp-toolsets-runtime``).
It lays down ``toolsets/<name>/`` — a Python package exporting ``TOOLS`` for the
runtime to serve — and, with ``--with-ui``, a Vite + React ``ui/`` project with
one example view wired to the example tool via a ``VIEWS`` export. The view's
bridge comes from ``@developmentseed/mcp-view`` (no vendored ``host.ts``).

The generated toolset depends on ``mcp-toolsets-runtime``; the workspace root's
``[tool.uv.sources]`` (its git tag / index pin) resolves it, so the new member
inherits whatever version the repo already tracks.
"""

import re
import shutil
import subprocess
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

app = typer.Typer(no_args_is_help=True, help=__doc__)
console = Console()


@app.callback()
def _root() -> None:
    """Keep ``new`` an explicit subcommand (and leave room for more)."""


# kebab-case: lowercase alnum groups joined by single hyphens, no leading/trailing.
NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")

# Templates use __NAME__ / __PKG__ sentinels and are rendered with str.replace
# (not str.format) so TS/JSON braces and ${VITE} vars survive verbatim.
PYPROJECT = """\
[project]
name = "__NAME__"
version = "0.1.0"
description = "TODO: describe the __NAME__ toolset."
requires-python = ">=3.12,<3.14"
dependencies = [
    "langchain-core<2.0.0,>=1.4.6",
    "mcp-toolsets-runtime",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
"""

# Appended for --with-ui: ship the vite-built (git-ignored) bundles in the wheel.
PYPROJECT_UI_EXTRA = """
[tool.hatch.build.targets.wheel]
# Views are built by ui/ (vite) into the package's views/ dir; they are
# git-ignored build artifacts, so force them into the wheel.
artifacts = ["src/__PKG__/views/*.html"]
"""

# A toolset's deployment config is written for whichever target the repo
# actually has. The two are not translations of each other: Helm values name
# Kubernetes concepts, and the AWS file names Fargate ones, so a repo keeps the
# file it can act on and nothing that points at a directory it does not have.
CHARTS_DIR = "charts"
INFRA_DIR = "infra"

TOOLSET_YAML = """\
# Optional Helm values overrides for this toolset (see charts/mcp-toolset/values.yaml).
{}
"""

TOOLSET_AWS_YAML = """\
# Optional deployment overrides for this toolset, read by the stack in infra/.
#
#   size:    a Fargate cpu/memory combination; omit for the smallest
#   env:     plain environment variables
#   secrets: environment variables read from Parameter Store, one path each
#
# size: { cpu: 256, memory: 512 }
# env:
#   EXAMPLE_URL: https://example.org
# secrets:
#   API_TOKEN: /mcp-toolsets/<instance>/__NAME__/api-token
{}
"""


def deployment_templates(root: Path) -> list[tuple[str, str]]:
    """Pick the deployment config files this repo can actually act on.

    A repo that chose one target at bootstrap has pruned the other, so writing
    both would leave a file naming a directory that is gone — inert, and only
    noticed when someone edits it and nothing happens. The template repo has
    both and gets both.

    A repo with neither still gets the Helm file: that is what every consumer
    got before this existed, and a repo laid out some third way is not one this
    can guess for.
    """
    files = []
    if (root / CHARTS_DIR).is_dir():
        files.append(("toolset.yaml", TOOLSET_YAML))
    if (root / INFRA_DIR).is_dir():
        files.append(("toolset.aws.yaml", TOOLSET_AWS_YAML))
    return files or [("toolset.yaml", TOOLSET_YAML)]


INIT_PY = '"""__NAME__ toolset."""\n'

TOOLS_PY = '''\
"""LangChain tools for __NAME__."""

from langchain_core.tools import tool

from mcp_runtime.tool_result import ToolResult


@tool
def example(query: str) -> ToolResult:
    """TODO: describe what this tool does (the docstring is the MCP schema)."""
    return ToolResult(message=f"you said: {query}")


TOOLS = [example]
'''

TOOLS_PY_UI = '''\
"""LangChain tools for __NAME__."""

from typing import NotRequired

from langchain_core.tools import tool

from mcp_runtime.tool_result import ToolResult


class ExampleResult(ToolResult):
    """A greeting plus who it was for — the data the panel view renders."""

    name: NotRequired[str]


@tool
def example(name: str = "world") -> ExampleResult:
    """TODO: describe what this tool does (the docstring is the MCP schema)."""
    return ExampleResult(message=f"Hello, {name}!", name=name)


TOOLS = [example]

# Map a tool to a UI view. The view is built from ui/ into
# src/__PKG__/views/<view-id>.html and rendered by any MCP Apps host (Claude.ai,
# ChatGPT, ...), fed this tool's structuredContent. Views are progressive
# enhancement: the message + data above still stand alone in a plain client.
VIEWS = {"example": "panel"}
'''

TEST_PY = """\
from __PKG__.tools import example


def test_example():
    assert example.invoke({"query": "hi"}) == {"message": "you said: hi"}
"""

TEST_PY_UI = """\
from __PKG__.tools import example


def test_example():
    assert example.invoke({"name": "dev"}) == {"message": "Hello, dev!", "name": "dev"}
"""

UI_PACKAGE_JSON = """\
{
  "name": "__NAME__-ui",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "typecheck": "tsc --noEmit",
    "build": "tsc --noEmit && VIEW=panel vite build",
    "dev": "VIEW=${VIEW:-panel} vite"
  },
  "dependencies": {
    "@developmentseed/mcp-view": "^0.1.0",
    "react": "^18.3.1",
    "react-dom": "^18.3.1"
  },
  "devDependencies": {
    "@types/node": "^22.10.2",
    "@types/react": "^18.3.12",
    "@types/react-dom": "^18.3.1",
    "@vitejs/plugin-react": "^4.3.4",
    "typescript": "^5.6.3",
    "vite": "^6.0.7",
    "vite-plugin-singlefile": "^2.1.0"
  }
}
"""

UI_TSCONFIG = """\
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "isolatedModules": true,
    "moduleDetection": "force",
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true
  },
  "include": ["src", "vite.config.ts"]
}
"""

UI_VITE_CONFIG = """\
import { resolve } from "node:path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { viteSingleFile } from "vite-plugin-singlefile";

// Build one self-contained view per pass (VIEW=<id>), inlined into a single
// HTML file in the Python package's views/ dir, where the runtime serves it as
// ui://<toolset>/<id>. Add a view: a new <id>.html + src/<id>.tsx, a VIEWS
// entry in tools.py, and a build pass in package.json's "build" script.
const view = process.env.VIEW;
if (!view) {
  throw new Error("set VIEW=<view-id> (e.g. VIEW=panel vite build)");
}

export default defineConfig({
  plugins: [react(), viteSingleFile()],
  build: {
    outDir: resolve(__dirname, "../src/__PKG__/views"),
    emptyOutDir: false,
    rollupOptions: {
      input: { [view]: resolve(__dirname, `${view}.html`) },
    },
  },
});
"""

UI_PANEL_HTML = """\
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>panel</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/panel.tsx"></script>
  </body>
</html>
"""

UI_PANEL_TSX = """\
import { StrictMode, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { onData, sendMessage } from "@developmentseed/mcp-view";

import "./styles.css";

// Mirror the tool's ToolResult (ExampleResult in tools.py). The host injects
// the whole structuredContent as this view's payload.
interface ExampleResult {
  message: string;
  name?: string;
}

function App() {
  const [data, setData] = useState<ExampleResult | null>(null);
  useEffect(() => {
    onData<ExampleResult>(setData);
  }, []);

  if (!data) return <div className="panel">Loading…</div>;
  return (
    <div className="panel">
      <p>{data.message}</p>
      {/* Interactions advance the chat: this becomes a user message, which the
          model reads and can act on with another tool. */}
      <button onClick={() => sendMessage(`Greet ${data.name ?? "everyone"} again`)}>
        Again
      </button>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
"""

UI_STYLES_CSS = """\
:root {
  color-scheme: light dark;
}
body {
  margin: 0;
  font: 14px/1.5 system-ui, -apple-system, sans-serif;
}
.panel {
  padding: 16px;
  display: flex;
  flex-direction: column;
  align-items: start;
  gap: 12px;
}
.panel button {
  border: 0;
  border-radius: 8px;
  padding: 8px 14px;
  background: #2563eb;
  color: #fff;
  font-weight: 600;
  cursor: pointer;
}
.panel button:hover {
  filter: brightness(1.08);
}
"""


def _render(template: str, name: str, pkg: str) -> str:
    return template.replace("__NAME__", name).replace("__PKG__", pkg)


def scaffold(root: Path, name: str, with_ui: bool) -> list[Path]:
    """Write a new toolset under ``root/toolsets/<name>``; return files written.

    Pure filesystem work — no ``uv``/network — so it's unit-testable. Raises
    ``ValueError`` on a non-kebab name or an existing toolset directory.
    """
    if not NAME_RE.match(name):
        raise ValueError(
            "toolset name must be kebab-case (lowercase letters, digits, hyphens)"
        )
    pkg = name.replace("-", "_")
    toolset = root / "toolsets" / name
    if toolset.exists():
        raise ValueError(f"{toolset} already exists")

    pkg_dir = toolset / "src" / pkg
    pkg_dir.mkdir(parents=True)
    (toolset / "tests").mkdir()

    def write(rel: str, template: str) -> Path:
        dest = toolset / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(_render(template, name, pkg), encoding="utf-8")
        return dest

    pyproject = PYPROJECT + (PYPROJECT_UI_EXTRA if with_ui else "")
    written = [
        write("pyproject.toml", pyproject),
        *(write(rel, template) for rel, template in deployment_templates(root)),
        write(f"src/{pkg}/__init__.py", INIT_PY),
        write(f"src/{pkg}/tools.py", TOOLS_PY_UI if with_ui else TOOLS_PY),
        write(f"tests/test_{pkg}.py", TEST_PY_UI if with_ui else TEST_PY),
    ]
    if with_ui:
        written += [
            write("ui/package.json", UI_PACKAGE_JSON),
            write("ui/tsconfig.json", UI_TSCONFIG),
            write("ui/vite.config.ts", UI_VITE_CONFIG),
            write("ui/panel.html", UI_PANEL_HTML),
            write("ui/src/panel.tsx", UI_PANEL_TSX),
            write("ui/src/styles.css", UI_STYLES_CSS),
        ]
    return written


@app.command()
def new(
    name: Annotated[str, typer.Argument(help="Toolset name (kebab-case).")],
    with_ui: Annotated[
        bool,
        typer.Option("--with-ui", help="Also scaffold a Vite + React ui/ view."),
    ] = False,
) -> None:
    """Scaffold ``toolsets/<name>/`` in the current repo and register it."""
    root = Path.cwd()
    try:
        written = scaffold(root, name, with_ui)
    except ValueError as error:
        console.print(f"[red]error: {error}[/red]")
        raise typer.Exit(1) from None

    pkg = name.replace("-", "_")
    for path in written:
        console.print(f"[green]created[/green] {path.relative_to(root)}")

    # Register it in the workspace (adds it to the root's dependencies). Best
    # effort: the scaffold above is the real work, so a missing/failed uv only
    # downgrades to a printed manual step.
    uv = shutil.which("uv")
    if uv is not None:
        try:
            subprocess.run([uv, "add", name], check=True)  # noqa: S603
            console.print(f"[green]registered[/green] {name} in the workspace")
        except (subprocess.CalledProcessError, OSError):
            console.print(f"[yellow]run `uv add {name}` to register it[/yellow]")
    else:
        console.print(
            f"[yellow]uv not found — run `uv add {name}` to register it[/yellow]"
        )

    console.print(
        f"\nNext: edit toolsets/{name}/src/{pkg}/tools.py, then run your tests."
    )
    if with_ui:
        console.print(
            "Build the view before serving (needs node):\n"
            f"  cd toolsets/{name}/ui && npm install && npm run build\n"
            f"Then: TOOLSET={name} uv run mcp-serve"
        )
