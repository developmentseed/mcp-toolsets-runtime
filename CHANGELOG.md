# Changelog

## [0.2.1](https://github.com/developmentseed/mcp-toolsets-runtime/compare/mcp-toolsets-runtime-v0.2.0...mcp-toolsets-runtime-v0.2.1) (2026-07-31)


### Features

* **state:** record what a tool was given from session state ([#40](https://github.com/developmentseed/mcp-toolsets-runtime/issues/40)) ([946dfa1](https://github.com/developmentseed/mcp-toolsets-runtime/commit/946dfa16258b14ad35bf2c796eba2839b0dbc3d5))

## [0.2.0](https://github.com/developmentseed/mcp-toolsets-runtime/compare/mcp-toolsets-runtime-v0.1.8...mcp-toolsets-runtime-v0.2.0) (2026-07-31)


### ⚠ BREAKING CHANGES

* **state:** three defects in binding tools and wiring the state channel ([#38](https://github.com/developmentseed/mcp-toolsets-runtime/issues/38))

### Bug Fixes

* **state:** three defects in binding tools and wiring the state channel ([#38](https://github.com/developmentseed/mcp-toolsets-runtime/issues/38)) ([995a0a6](https://github.com/developmentseed/mcp-toolsets-runtime/commit/995a0a6acb3ce814eb594632841730f0a711aa2e))

## [0.1.8](https://github.com/developmentseed/mcp-toolsets-runtime/compare/mcp-toolsets-runtime-v0.1.7...mcp-toolsets-runtime-v0.1.8) (2026-07-31)


### Bug Fixes

* **mcp-agent:** describe where the BYOM key actually goes ([#36](https://github.com/developmentseed/mcp-toolsets-runtime/issues/36)) ([28b823e](https://github.com/developmentseed/mcp-toolsets-runtime/commit/28b823eb8688268f579cfa1c4f6d73e9eaa7329d))

## [0.1.7](https://github.com/developmentseed/mcp-toolsets-runtime/compare/mcp-toolsets-runtime-v0.1.6...mcp-toolsets-runtime-v0.1.7) (2026-07-31)


### Features

* **mcp-agent:** checkpoint conversations, in memory by default ([#35](https://github.com/developmentseed/mcp-toolsets-runtime/issues/35)) ([1f4abe6](https://github.com/developmentseed/mcp-toolsets-runtime/commit/1f4abe60a447685d94d2a13f6f7921f4c21659a1))
* **mcp-agent:** wire session state into the bundled agent ([#32](https://github.com/developmentseed/mcp-toolsets-runtime/issues/32)) ([1e20568](https://github.com/developmentseed/mcp-toolsets-runtime/commit/1e20568dccd0d6334229e014809b8dca15bbc020))


### Bug Fixes

* **state:** read every stored key, and bound the namespace ([#34](https://github.com/developmentseed/mcp-toolsets-runtime/issues/34)) ([9762592](https://github.com/developmentseed/mcp-toolsets-runtime/commit/976259274e13d484e715e994ef6d204bf01987de))

## [0.1.6](https://github.com/developmentseed/mcp-toolsets-runtime/compare/mcp-toolsets-runtime-v0.1.5...mcp-toolsets-runtime-v0.1.6) (2026-07-31)


### Features

* **runtime:** serve several toolsets in one local process ([#30](https://github.com/developmentseed/mcp-toolsets-runtime/issues/30)) ([4a1760d](https://github.com/developmentseed/mcp-toolsets-runtime/commit/4a1760d56432fe98735eb3c1fadd2cc6912cc963))
* **state:** keep large tool values out of the model, on any MCP server ([#17](https://github.com/developmentseed/mcp-toolsets-runtime/issues/17)) ([4cd1f56](https://github.com/developmentseed/mcp-toolsets-runtime/commit/4cd1f5623a17dd98bc02dd79c655ebe8edf9c7ad))

## [0.1.5](https://github.com/developmentseed/mcp-toolsets-runtime/compare/mcp-toolsets-runtime-v0.1.4...mcp-toolsets-runtime-v0.1.5) (2026-07-30)


### Features

* **mcp-agent:** render tool views in the side panel, not inline ([#29](https://github.com/developmentseed/mcp-toolsets-runtime/issues/29)) ([7a3c6f2](https://github.com/developmentseed/mcp-toolsets-runtime/commit/7a3c6f26a72c84acedf0798cd352524daab12d6a))


### Documentation

* install from PyPI, with badges instead of pinned versions ([#22](https://github.com/developmentseed/mcp-toolsets-runtime/issues/22)) ([9b696bc](https://github.com/developmentseed/mcp-toolsets-runtime/commit/9b696bc84810b3489846e6750245ad4990b57b58))

## [0.1.4](https://github.com/developmentseed/mcp-toolsets-runtime/compare/mcp-toolsets-runtime-v0.1.3...mcp-toolsets-runtime-v0.1.4) (2026-07-30)


### Bug Fixes

* **mcp-view:** report the package's real version to the host ([#20](https://github.com/developmentseed/mcp-toolsets-runtime/issues/20)) ([7cfabc8](https://github.com/developmentseed/mcp-toolsets-runtime/commit/7cfabc8375a45671cdad4bd1a7bdcfff3cdd71d6))

## [0.1.3](https://github.com/developmentseed/mcp-toolsets-runtime/compare/mcp-toolsets-runtime-v0.1.2...mcp-toolsets-runtime-v0.1.3) (2026-07-30)


### Features

* **publish:** release via App token, publish to PyPI and public npm ([#14](https://github.com/developmentseed/mcp-toolsets-runtime/issues/14)) ([b1da2e0](https://github.com/developmentseed/mcp-toolsets-runtime/commit/b1da2e0439d4525274b7d83a379e74a4078198d8))

## [0.1.2](https://github.com/developmentseed/mcp-toolsets-runtime/compare/mcp-toolsets-runtime-v0.1.1...mcp-toolsets-runtime-v0.1.2) (2026-07-28)


### Bug Fixes

* **release:** publish the npm bridge inline and lock versions in step ([#12](https://github.com/developmentseed/mcp-toolsets-runtime/issues/12)) ([ee28bc2](https://github.com/developmentseed/mcp-toolsets-runtime/commit/ee28bc208b4f8af0022a4796dad08d7da54570bb))

## [0.1.1](https://github.com/developmentseed/mcp-toolsets-runtime/compare/mcp-toolsets-runtime-v0.1.0...mcp-toolsets-runtime-v0.1.1) (2026-07-28)


### Features

* extract shared runtime into an installable package ([#1](https://github.com/developmentseed/mcp-toolsets-runtime/issues/1)) ([ca9ec3d](https://github.com/developmentseed/mcp-toolsets-runtime/commit/ca9ec3d37893b54dfd53409d4e4c13c7fef1b762))

## Changelog
