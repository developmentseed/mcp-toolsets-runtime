# Contributing

## Your PR title is the changelog

This repo releases with [release-please](https://github.com/googleapis/release-please):
it reads the commits on `main`, groups them into `CHANGELOG.md`, and bumps the
version. Because `main` uses **squash-merge with the PR title as the commit
subject**, your **PR title** is the line that lands in the changelog.

So every PR title must be a valid [Conventional Commit](https://www.conventionalcommits.org/):

```
<type>(<optional scope>): <description>
```

CI (`pr-title`) fails the PR if the title isn't one of these types:

| Type | Use for | Version effect (pre-1.0) |
| --- | --- | --- |
| `feat` | a new capability | minor |
| `fix` | a bug fix | patch |
| `perf` | a performance improvement | patch |
| `refactor` | internal change, no behaviour change | patch |
| `docs` | docs only | none |
| `test` | tests only | none |
| `build` | build system / deps | none |
| `ci` | CI config | none |
| `chore` | anything else | none |

A **breaking change** — an incompatible change to the plugin contract
(`TOOLS` / `VIEWS` / `CREDENTIAL_HEADERS`), `ToolResult`, the `ui/*` wire
protocol, or a public signature — gets a `!`:

```
feat!: require ToolResult from every tool
```

Examples:

```
feat(runtime): stamp view _meta on tools that declare VIEWS
fix(agent): only inject declared credential headers
chore(deps): bump mcp to 1.29
```

## The release flow

1. Merge PRs with conventional titles.
2. release-please keeps a rolling **release PR** that bumps `pyproject.toml` and
   `js/mcp-view/package.json` (kept equal by the linked-versions plugin) and
   updates `CHANGELOG.md`. **This PR is editable** — curate the changelog prose
   before merging if a release warrants it.
3. Merging the release PR tags `vX.Y.Z`, cuts a GitHub Release, and publishes
   the JS package.

## Local checks

```bash
uv sync --all-extras
./scripts/lint
./scripts/test
./scripts/build-js
```

Lint/type rules live entirely in `pyproject.toml` — the scripts pass no rule
flags, so your editor, `./scripts/format`, and CI all agree.
