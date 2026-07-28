# Extract the shared runtime into `mcp-toolsets-runtime`

## Goal

Stop hand-porting the runtime between [`developmentseed/mcp-toolsets`](https://github.com/developmentseed/mcp-toolsets) (the
template) and `ecmwf/dss-agentic-ai-services` (generated from it). Publish the
shared runtime once, from this repo — the **Python** packages as a single
installable distribution, and the **TS view bridge** alongside it — consumed by
both repos via pinned versions (git URL now, PyPI/npm later).

## Decisions (locked)

- **One collapsed Python distribution** named `mcp-toolsets-runtime`, exposing
  the existing top-level modules `mcp_runtime`, `mcp_cli`, `mcp_agent` unchanged,
  so consumer `import` lines never change — only the dependency declaration does.
- **Single Python version** for the whole package (all three modules move together).
- **`[agent]` optional extra** for the chainlit/langchain web host, so the base
  install (runtime + cli) stays lean for tool-serving images.
- **TS bridge in scope** (new vs. the first draft): `host.ts` becomes an
  importable npm package; `McpView.jsx` ships with `mcp_agent`. Both ends
  version in **lockstep** with the Python package (shared wire protocol).
- **uv** for all Python work; **npm** for the JS package (matches the existing
  toolset `ui/` tooling).
- **Ruff config lives in `pyproject.toml`**, not in the scripts. `scripts/lint`
  calls `ruff check` / `ruff format --check` with no rule overrides.
- **No pre-commit.** The `scripts/` are the only local entrypoints.
- **release-please** drives versioning + CHANGELOG from **conventional commits**;
  the "changelog entry or fail" rule is a required **PR-title check**. The
  generated changelog is an editable draft (see note), not a locked robot output.
- **Non-destructive Phase 1**: copy code in; leave both consumers as-is until a
  tagged release exists and we flip them deliberately (Phase 2).

---

## Target repo layout

```
mcp-toolsets-runtime/
├── pyproject.toml                     # single [project], one version, extras, ALL ruff/mypy config
├── uv.lock
├── README.md                          # install, module map, release process
├── CHANGELOG.md                       # release-please draft, human-editable
├── CONTRIBUTING.md                    # conventional-commit + PR-title rules
├── LICENSE                            # exists
├── .gitignore                         # exists (Python) + node_modules/, js dist
├── release-please-config.json         # 2 packages, linked-versions
├── .release-please-manifest.json
├── .github/
│   ├── CODEOWNERS
│   ├── dependabot.yml                 # pip + npm + github-actions (bumps pinned SHAs)
│   └── workflows/
│       ├── ci.yml                     # python lint+test, js build+test, PR-title check
│       ├── release-please.yml         # release PR + tag/release on main
│       ├── publish-pypi.yml           # OIDC PyPI publish on release (gated until PyPI exists)
│       └── publish-npm.yml            # npm publish of @developmentseed/mcp-view on release
├── scripts/
│   ├── lint                           # ruff check + ruff format --check + mypy src   (no rule flags)
│   ├── test                           # pytest -v
│   ├── format                         # ruff check --fix + ruff format
│   └── build-js                       # npm ci + build + typecheck in js/mcp-view
├── src/
│   ├── mcp_runtime/                   # copied verbatim (+ py.typed)
│   │   ├── __init__.py credentials.py fastmcp_output.py index.py
│   │   ├── server.py tool_result.py views.py
│   │   └── py.typed
│   ├── mcp_cli/                       # copied verbatim (+ py.typed)
│   │   ├── __init__.py main.py py.typed
│   └── mcp_agent/                     # copied verbatim (+ py.typed)
│       ├── __init__.py main.py web.py py.typed
│       └── elements/
│           └── McpView.jsx            # host-side Chainlit element, shipped as package data
├── js/
│   └── mcp-view/                      # npm: @developmentseed/mcp-view (the view-side bridge)
│       ├── package.json
│       ├── tsconfig.json
│       ├── src/
│       │   ├── host.ts                # copied verbatim from toolsets/*/ui/src/host.ts
│       │   └── index.ts               # public exports (onData/sendMessage seam)
│       └── test/ ...                  # vitest (typecheck is the primary gate)
├── tests/
│   ├── mcp_runtime/ ...               # test_contract/credentials/fastmcp_output/index/server/views
│   ├── mcp_cli/test_cli.py
│   └── mcp_agent/test_agent.py
└── docs/plans/extract-runtime-package.md
```

`py.typed` markers (PEP 561) are new — without them, consumers running mypy
against the git-installed wheel would treat the runtime as untyped.

---

## The two TS pieces (why split this way)

The bridge has two ends, and they distribute differently:

- **`host.ts` — view side, importable → npm package `@developmentseed/mcp-view`.**
  Each toolset `ui/` currently vendors an identical copy; that's the biggest JS
  drift. As a package, a toolset does
  `import { onData, sendMessage } from "@developmentseed/mcp-view"` instead of
  copying the file. Clean, high value.

- **`McpView.jsx` — host side, an *asset* not a module.** Chainlit loads it by
  path from the app root's `public/elements/`; it's never `import`ed and it uses
  Chainlit globals (`props`, `sendUserMessage`). So it can't be a normal npm
  dependency. It ships as **package data inside `mcp_agent`** (which *is* the
  Chainlit host via `web.py`), and `web.py` copies it into the app root's
  `public/elements/` on startup if absent — zero manual step for consumers.

> npm install wrinkle to decide: pip installs a package from a git repo root
> cleanly, but npm installing a **subdirectory** of a git repo is painful. So the
> JS package should be **published to a registry from the first release** rather
> than git-URL-installed. Recommended default: **GitHub Packages** (npm registry,
> auth via `GITHUB_TOKEN`, stays in the org, no new secret). Alternative: public
> npmjs. Flagged as an open choice — see `publish-npm.yml`.

---

## `pyproject.toml` (Python) — config-owned linting

```toml
[project]
name = "mcp-toolsets-runtime"
version = "0.1.0"
description = "Shared runtime for MCP Toolsets: discover LangChain tools, serve them as MCP, CLI + example agent."
readme = "README.md"
requires-python = ">=3.12,<3.14"
license = { file = "LICENSE" }
dependencies = [
    # mcp_runtime
    "mcp<2.0.0,>=1.27.2",
    "langchain-mcp-adapters<0.4.0,>=0.3.0",
    "langchain-core<2.0.0,>=1.4.6",
    "pydantic-settings<3.0.0,>=2.12.0",
    "fastapi<1.0.0,>=0.136.3",
    "typing-extensions>=4.10.0",
    # mcp_cli
    "typer<0.27.0,>=0.26.7",
    "rich<16.0.0,>=15.0.0",
]

[project.optional-dependencies]
agent = [
    "chainlit<3.0.0,>=2.11.1",
    "httpx<1.0.0,>=0.28.1",
    "langchain<2.0.0,>=1.3.7",
]

[project.scripts]
mcp-serve     = "mcp_runtime.server:main"
mcp-index     = "mcp_runtime.index:main"
mcp-cli       = "mcp_cli.main:app"
mcp-agent     = "mcp_agent.main:app"        # requires [agent]
mcp-agent-web = "mcp_agent.web:main"        # requires [agent]

[dependency-groups]
dev = [
    "pytest<10.0.0,>=9.0.3",
    "pytest-asyncio<2.0.0,>=1.4.0",
    "pytest-cov>=6.0.0",
    "ruff<0.15.0,>=0.14.11",
    "mypy<2.0.0,>=1.19.1",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/mcp_runtime", "src/mcp_cli", "src/mcp_agent"]
# Ship the Chainlit element as package data inside mcp_agent:
[tool.hatch.build.targets.wheel.force-include]
"src/mcp_agent/elements/McpView.jsx" = "mcp_agent/elements/McpView.jsx"

# --- ALL lint config lives here; scripts never pass rule flags ---
[tool.ruff]
line-length = 88
[tool.ruff.lint]
select = ["E4", "E7", "E9", "F", "S"]        # S (bandit) promoted into config
[tool.ruff.lint.per-file-ignores]
"tests/**"   = ["S"]
"scripts/**" = ["S"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--import-mode=importlib"
asyncio_mode = "auto"

[tool.mypy]
plugins = ["pydantic.mypy"]
warn_unused_ignores = true
warn_redundant_casts = true
check_untyped_defs = true
[[tool.mypy.overrides]]
module = ["langchain_mcp_adapters.*"]
ignore_missing_imports = true
```

`scripts/lint` is now purely:

```sh
#!/usr/bin/env sh
set -e
uv run ruff check           # rules come from pyproject, not this script
uv run ruff format --check
uv run mypy src
```

This is the change you asked for: promoting `S` into `[tool.ruff.lint] select`
means the CLI no longer overrides the rule set (mcp-toolsets' old
`ruff check --select S` trick is gone), and `ruff check` in the script now
enforces exactly what the config says — the editor, `format`, and CI all agree.

---

## CI — `.github/workflows/ci.yml`

Actions pinned by SHA (house style). Three jobs: Python, JS, and the PR-title gate.

```yaml
name: CI
on:
  pull_request:
  push:
    branches: [main]

jobs:
  python:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.12", "3.13"]
    steps:
      - uses: actions/checkout@<pin v7>
      - uses: astral-sh/setup-uv@<pin v9>
        with:
          python-version: ${{ matrix.python-version }}
      - run: uv sync --frozen --all-extras     # [agent] included so mcp_agent is tested
      - run: ./scripts/lint
      - run: ./scripts/test

  js:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@<pin v7>
      - uses: actions/setup-node@<pin v7>
        with:
          node-version: "22"
      - run: ./scripts/build-js                 # npm ci + tsc typecheck + build + vitest

  pr-title:
    # The "changelog entry or fail" gate.
    if: github.event_name == 'pull_request'
    runs-on: ubuntu-latest
    steps:
      - uses: amannn/action-semantic-pull-request@<pin sha>
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        with:
          types: |
            feat
            fix
            perf
            refactor
            docs
            test
            build
            ci
            chore
```

---

## Release process — release-please (Python + JS, linked)

`release-please-config.json` — two packages, one linked version so the Python
distribution and the npm bridge always bump together:

```json
{
  "$schema": "https://raw.githubusercontent.com/googleapis/release-please/main/schemas/config.json",
  "plugins": [
    { "type": "linked-versions", "groupName": "runtime",
      "components": ["mcp-toolsets-runtime", "@developmentseed/mcp-view"] }
  ],
  "packages": {
    ".": {
      "release-type": "python",
      "package-name": "mcp-toolsets-runtime",
      "changelog-path": "CHANGELOG.md",
      "bump-minor-pre-major": true,
      "bump-patch-for-minor-pre-major": true
    },
    "js/mcp-view": {
      "release-type": "node",
      "package-name": "@developmentseed/mcp-view"
    }
  }
}
```

`.release-please-manifest.json`:

```json
{ ".": "0.1.0", "js/mcp-view": "0.1.0" }
```

`release-please.yml` runs on push to `main`, uses
`googleapis/release-please-action@<pin sha>` with `contents: write` +
`pull-requests: write`, reading the config/manifest from the repo root.

How the loop works:

1. PRs merge to `main` with conventional-commit **titles** (enforced by
   `pr-title`). Repo is set to **squash-merge using the PR title** as the commit
   subject → the title is the commit release-please parses.
2. release-please maintains a rolling "release PR" that bumps the version in
   `pyproject.toml` **and** `js/mcp-view/package.json` (kept equal by
   linked-versions) and rewrites `CHANGELOG.md`.
3. Merging that release PR tags `vX.Y.Z`, cuts a GitHub Release, and the two
   `publish-*` workflows fire.

Required repo settings (need a maintainer/admin):
- **Squash merge only**, "default to PR title for squash commits" **on**.
- Branch protection on `main`: require `python`, `js`, and `pr-title` checks.

### On the "robotic changelog" concern

Fair — the raw grouped output reads mechanical. Two things soften it, and you
don't have to choose now:

- **The release PR is editable.** release-please opens a PR with a *draft*
  CHANGELOG; a human can rewrite/curate it before merging. So the workflow is
  "auto-draft → human-polish → cut," not a locked robot commit.
- Writing a real PR **body** (not just the title) lets you add a
  `Release-As:`/notes section release-please will surface, so entries can carry
  prose, not just `fix: x`.

If it still grates once you've lived with it, the fallback is a news-fragment
tool (changie) for hand-written entries — but that's more per-PR ceremony, which
cuts against "just the scripts." Recommendation: run release-please as-is,
curate the release PR when a release actually matters, revisit only if it annoys.

---

## Publish workflows (wired now)

- **`publish-pypi.yml`** — on `release: published`, `uv build` +
  `pypa/gh-action-pypi-publish@<pin>` via **OIDC trusted publishing** (no token).
  Gated (`if: false` / unapproved `pypi` environment) until the PyPI project +
  trusted publisher are registered. Until then, consumers use the git-URL pin.
- **`publish-npm.yml`** — on `release: published`, publish
  `@developmentseed/mcp-view`. Default target **GitHub Packages** (`GITHUB_TOKEN`,
  no new secret); switch the registry to npmjs if you want it fully public. This
  one is **on from the first release**, because npm can't cleanly install the JS
  subdirectory from a git URL (see the wrinkle above).

---

## Additional repo hygiene

- **CONTRIBUTING.md** — conventional-commit types + "your PR title is the
  changelog," so the `pr-title` gate isn't a surprise.
- **CODEOWNERS** — runtime changes get the right review.
- **dependabot.yml** — `pip`, `npm`, and `github-actions` (auto-bumps pinned
  action SHAs, which otherwise rot).
- **pytest-cov** in CI with a soft coverage report (optional floor later).
- **README "module map"** — what each of the three Python modules is, the
  `TOOLS` / `VIEWS` / `CREDENTIAL_HEADERS` plugin contract `mcp_runtime` expects,
  the install matrix (base vs `[agent]`), and how the npm bridge is consumed.
- (Considered and dropped per your call: pre-commit.)

---

## Phase 2 — flip the consumers (separate, later PRs; not in this repo)

Once `v0.1.0` is tagged, in **each** of `mcp-toolsets` and
`dss-agentic-ai-services`:

1. Delete `packages/mcp-runtime`, `packages/mcp-cli`, `packages/mcp-agent`.
2. Root `pyproject.toml`: drop those three from `dependencies` and
   `[tool.uv.sources]`; add
   ```toml
   dependencies = ["mcp-toolsets-runtime[agent]", ...]
   [tool.uv.sources]
   mcp-toolsets-runtime = { git = "https://github.com/developmentseed/mcp-toolsets-runtime.git", tag = "v0.1.0" }
   ```
3. Each toolset `ui/` drops its vendored `src/host.ts` and depends on
   `@developmentseed/mcp-view`; delete the repo-root `public/elements/McpView.jsx`
   (now supplied by `mcp_agent` on startup).
4. `[tool.uv.workspace] members` drops `packages/*` (keep if other local packages
   remain).
5. `uv lock`, `./scripts/lint && ./scripts/test`, agent/tool smoke test. Python
   imports are unchanged.
6. Bumping the runtime later = change the `tag` + `uv lock`. One line, both repos.

Do the two consumer migrations as independent PRs so each can be validated (and
reverted) on its own — ecmwf can lag mcp-toolsets if a runtime change needs
cluster validation first.

---

## The discipline this introduces (the real cost)

- The runtime now needs **semver + a changelog**: a breaking change to
  `build_server` / `ToolResult` / the `ui/*` wire shapes is a major bump and a
  coordinated consumer upgrade, not a same-afternoon edit in both repos.
- Treat the plugin contract (`TOOLS` / `VIEWS` / `CREDENTIAL_HEADERS`,
  `ToolResult`, the bridge protocol version) as **public API** — a short
  `CONTRACT.md` keeps changes deliberate, and linked-versions keeps the Python
  and TS ends from skewing.
- Two CI systems and two registries instead of one repo, but each is lighter and
  the back-and-forth porting stops.

## Execution order (Phase 1 checklist)

1. `src/` + `tests/` copied in, `py.typed` added; `McpView.jsx` → `mcp_agent/elements/`.
2. `js/mcp-view/` created from `host.ts` (+ `index.ts` exports, tsconfig, vitest).
3. `pyproject.toml` (config-owned ruff/mypy), `scripts/{lint,test,format,build-js}`.
4. `uv lock`; `./scripts/lint && ./scripts/test` green locally with `--all-extras`;
   `./scripts/build-js` green.
5. `ci.yml` (python matrix + js + pr-title).
6. release-please config/manifest/workflow + linked-versions; seed `CHANGELOG.md`.
7. `publish-pypi.yml` (gated), `publish-npm.yml` (live), dependabot, CODEOWNERS,
   CONTRIBUTING, README.
8. Repo settings (squash-by-title, branch protection).
9. First release PR merges → `v0.1.0` + npm publish → Phase 2 unblocked.
```
