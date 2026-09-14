"""The authoring skill ships with the package and installs into a repo."""

import tomllib

import pytest
from typer.testing import CliRunner

from mcp_toolset.main import SKILL, SKILL_DIR, app

runner = CliRunner()


def _frontmatter() -> dict[str, str]:
    """The YAML block an agent reads to decide whether to load the skill.

    Parsed as TOML-ish key/value lines rather than with a YAML dependency:
    the block is two scalar fields, and this only has to catch a missing or
    malformed one.
    """
    text = SKILL.read_text(encoding="utf-8")
    assert text.startswith("---\n"), "the skill needs a frontmatter block"
    block = text.split("---\n")[1]
    fields = {}
    for line in block.splitlines():
        key, separator, value = line.partition(":")
        assert separator, f"frontmatter line is not key: value -- {line!r}"
        fields[key.strip()] = value.strip()
    return fields


def test_the_skill_ships_with_the_package():
    assert SKILL.is_file()


@pytest.mark.parametrize("field", ["name", "description"])
def test_the_frontmatter_carries(field):
    """`description` is what an agent matches on, so an empty one is inert."""
    assert _frontmatter().get(field)


def test_the_name_matches_the_directory_it_installs_to():
    assert _frontmatter()["name"] == SKILL_DIR.name


def test_the_skill_names_only_commands_that_exist():
    """It tells an agent what to run, so a stale command is worse than none."""
    text = SKILL.read_text(encoding="utf-8")
    scripts = tomllib.loads(
        (SKILL.parents[3] / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]["scripts"]
    for command in ("mcp-toolset new", "mcp-serve-local", "mcp-cli"):
        assert command in text
        assert command.split()[0] in scripts


def test_install_writes_it_where_an_agent_looks(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["skill", "--install"])
    assert result.exit_code == 0, result.output
    installed = tmp_path / SKILL_DIR / "SKILL.md"
    assert installed.read_text(encoding="utf-8") == SKILL.read_text(encoding="utf-8")


def test_install_replaces_an_older_copy(tmp_path, monkeypatch):
    """A repo re-runs this after a runtime bump, so it overwrites in place."""
    monkeypatch.chdir(tmp_path)
    stale = tmp_path / SKILL_DIR / "SKILL.md"
    stale.parent.mkdir(parents=True)
    stale.write_text("out of date", encoding="utf-8")
    assert runner.invoke(app, ["skill", "--install"]).exit_code == 0
    assert stale.read_text(encoding="utf-8") != "out of date"


def test_without_install_it_only_prints_the_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["skill"])
    assert result.exit_code == 0
    assert not (tmp_path / ".claude").exists()
