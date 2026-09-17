"""The authoring skill ships with the package and installs into a repo."""

import re
import tomllib

import pytest
from typer.testing import CliRunner

from mcp_toolset.main import SKILL, SKILL_DIR, SKILL_PATH, app

runner = CliRunner()


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A directory `--install` will accept: one with a pyproject.toml in it.

    Both installed paths are relative to the working directory, so the command
    refuses anywhere that is not a repo root. Every install test therefore
    needs a root rather than a bare tmp_path.
    """
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n", "utf-8")
    monkeypatch.chdir(tmp_path)
    return tmp_path


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
    """It tells an agent what to run, so a stale command is worse than none.

    Extracts what the skill names rather than checking a hand-kept list: a
    fixed list only proves the commands on it are real, and says nothing
    about the next one somebody adds.
    """
    text = SKILL.read_text(encoding="utf-8")
    scripts = tomllib.loads(
        (SKILL.parents[3] / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]["scripts"]
    named = set(re.findall(r"uv run ([a-z0-9-]+)", text))
    assert named, "extraction found no commands; the pattern has rotted"
    assert not named - set(scripts), f"not shipped in the wheel: {named - set(scripts)}"


def test_the_skill_names_no_repo_local_paths():
    """It installs into repos this one has never seen, so `./x` may not exist.

    The scaffold writes a package, a test, a pyproject and the deployment
    file -- never a `scripts/` directory. Naming one sends an agent at a
    path that happens to exist in the repos we know and nowhere else.
    """
    text = SKILL.read_text(encoding="utf-8")
    local = re.findall(r"^\s*\./\S+", text, re.MULTILINE)
    assert not local, f"repo-local paths a consumer may not have: {local}"


def test_install_writes_it_where_an_agent_looks(repo):
    result = runner.invoke(app, ["skill", "--install"])
    assert result.exit_code == 0, result.output
    installed = repo / SKILL_DIR / "SKILL.md"
    assert installed.read_text(encoding="utf-8") == SKILL.read_text(encoding="utf-8")


def test_install_replaces_an_older_copy(repo):
    """A repo re-runs this after a runtime bump, so it overwrites in place."""
    stale = repo / SKILL_DIR / "SKILL.md"
    stale.parent.mkdir(parents=True)
    stale.write_text("out of date", encoding="utf-8")
    assert runner.invoke(app, ["skill", "--install"]).exit_code == 0
    assert stale.read_text(encoding="utf-8") != "out of date"


def test_install_points_an_agents_file_at_the_skill(repo):
    """An agent reading AGENTS.md never looks under .claude/skills/ itself."""
    assert runner.invoke(app, ["skill", "--install"]).exit_code == 0
    assert SKILL_PATH in (repo / "AGENTS.md").read_text(encoding="utf-8")


def test_install_appends_to_an_agents_file_it_did_not_write(repo):
    """That file is the consuming repo's, so installing must not replace it."""
    theirs = repo / "AGENTS.md"
    theirs.write_text("# House rules\n\nRun the tests before pushing.\n", "utf-8")
    assert runner.invoke(app, ["skill", "--install"]).exit_code == 0
    after = theirs.read_text(encoding="utf-8")
    assert "Run the tests before pushing." in after
    assert SKILL_PATH in after


def test_installing_twice_leaves_one_pointer(repo):
    """A repo re-runs this after every bump, so the pointer must not stack."""
    for _ in range(2):
        assert runner.invoke(app, ["skill", "--install"]).exit_code == 0
    after = (repo / "AGENTS.md").read_text(encoding="utf-8")
    assert after.count(SKILL_PATH) == 1


def test_install_refuses_where_there_is_no_pyproject(tmp_path, monkeypatch):
    """Run from a subdirectory it would install where nothing reads it.

    A nested `.claude/skills/` is never loaded, while the AGENTS.md beside it
    is -- Cursor takes nested ones -- so the pointer would resolve to a skill
    no agent reads. Writing neither, and saying why, beats that.
    """
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["skill", "--install"])
    assert result.exit_code != 0
    assert "pyproject.toml" in result.output
    assert not (tmp_path / ".claude").exists()
    assert not (tmp_path / "AGENTS.md").exists()


def test_without_install_it_only_prints_the_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["skill"])
    assert result.exit_code == 0
    assert not (tmp_path / ".claude").exists()
    assert not (tmp_path / "AGENTS.md").exists()
