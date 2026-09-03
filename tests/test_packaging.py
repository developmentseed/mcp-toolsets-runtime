"""What each extra is allowed to pull in.

These read ``pyproject.toml`` rather than the installed environment on purpose:
CI syncs ``--all-extras``, so an environment check would pass no matter which
extra a dependency was declared under. The layering is only visible in the
declaration.
"""

import re
import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _project() -> dict:
    return tomllib.loads(PYPROJECT.read_text())["project"]


def _extras() -> dict[str, list[str]]:
    return _project()["optional-dependencies"]


def _names(requirements: list[str]) -> set[str]:
    """Distribution names, dropping extras, specifiers and environment markers."""
    return {
        re.split(r"[\[<>=!~;\s]", requirement, maxsplit=1)[0]
        for requirement in requirements
    }


def test_chainlit_belongs_to_web_and_nowhere_else():
    """The reason [web] exists: an API deployment installs [agent] and must not
    get a UI framework it never imports. Only ``mcp_agent.web`` imports chainlit.
    """
    extras = _extras()
    assert "chainlit" in _names(extras["web"])
    for name, requirements in extras.items():
        if name != "web":
            assert "chainlit" not in _names(requirements), (
                f"chainlit leaked into the [{name}] extra"
            )
    assert "chainlit" not in _names(_project()["dependencies"])


def test_the_extras_are_a_chain():
    """[web] -> [agent] -> [state]. Installing the outermost must bring the rest,
    so no consumer has to name two extras to get a working host.
    """
    extras = _extras()
    assert "mcp-toolsets-runtime[agent]" in extras["web"]
    assert "mcp-toolsets-runtime[state]" in extras["agent"]


def test_httpx_is_a_base_dependency():
    """``mcp_runtime.index`` imports it and ``mcp-index`` is a base entry point,
    so a tool-serving image that installs no extra still needs it. It resolved
    transitively through mcp until 0.5.0, which was luck rather than design.
    """
    assert "httpx" in _names(_project()["dependencies"])


def test_boto3_belongs_to_aws_and_nowhere_else():
    """The reason [aws] exists: only the ECS discovery backend imports boto3, so
    a cluster deployment — and every tool-serving image on either platform —
    must not install an AWS SDK to serve a directory it reaches over httpx.
    """
    extras = _extras()
    assert "boto3" in _names(extras["aws"])
    for name, requirements in extras.items():
        if name != "aws":
            assert "boto3" not in _names(requirements), (
                f"boto3 leaked into the [{name}] extra"
            )
    assert "boto3" not in _names(_project()["dependencies"])


def test_aws_stands_alone():
    """[aws] is not part of the [state] -> [agent] -> [web] chain: an index on
    ECS needs it and an agent does not, whichever platform the agent runs on.
    """
    assert not [
        requirement
        for requirement in _extras()["aws"]
        if requirement.startswith("mcp-toolsets-runtime")
    ]
