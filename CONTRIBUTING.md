# Contributing

## Your PR title is the changelog

This repo releases with
[release-please](https://github.com/googleapis/release-please). It reads the
commits on `main`, groups them into `CHANGELOG.md`, and bumps the version.
`main` uses squash-merge with the PR title as the commit subject, so your **PR
title** is the line that lands in the changelog.

Every PR title must be a valid
[Conventional Commit](https://www.conventionalcommits.org/):

```
<type>(<optional scope>): <description>
```

The `pr-title` check fails the PR if the type is not one of these:

| Type | Use for |
| --- | --- |
| `feat` | a new capability |
| `fix` | a bug fix |
| `perf` | a performance improvement |
| `refactor` | an internal change with no behaviour change |
| `docs` | documentation only |
| `test` | tests only |
| `build` | the build system or dependencies |
| `ci` | CI configuration |
| `chore` | anything else |

Mark a **breaking change** with a `!`:

```
feat!: require ToolResult from every tool
```

Breaking means an incompatible change to the plugin contract (`TOOLS`,
`VIEWS`, `CREDENTIAL_HEADERS`), to `ToolResult`, to the `ui/*` wire protocol,
or to a public signature.

Examples:

```
feat(runtime): stamp view _meta on tools that declare VIEWS
fix(agent): only inject declared credential headers
chore(deps): bump mcp to 1.29
```

### What the version does, pre-1.0

The package is below 1.0, and `release-please-config.json` sets both
`bump-minor-pre-major` and `bump-patch-for-minor-pre-major`. Together those
shift every bump down one place:

| Change | Bump |
| --- | --- |
| a breaking change (`!`) | minor |
| anything else that releases | patch |

So a `feat` is a **patch** here, not a minor. The two most recent releases show
it: 0.9.1 was a single `feat`, and 0.9.0 was a breaking change.

This matters to consumers, who bound the dependency at the next minor. A
change that breaks them has to carry the `!`, or release-please ships it inside
a range they already allow.

## The release flow

1. Merge PRs with conventional titles.
2. release-please keeps a rolling release PR. It bumps `pyproject.toml` and
   `js/mcp-view/package.json`, which the linked-versions plugin keeps equal,
   and updates `CHANGELOG.md`. **That PR is editable**, so curate the changelog
   before merging if a release warrants it.
3. Merging the release PR tags `vX.Y.Z` and cuts a GitHub Release. That
   publishes the wheel to PyPI and the bridge to npm.

## Local checks

```bash
uv sync --all-extras
./scripts/lint
./scripts/test
./scripts/build-js
```

Lint and type rules live in `pyproject.toml`. The scripts pass no rule flags,
so your editor, `./scripts/format` and CI all agree.
