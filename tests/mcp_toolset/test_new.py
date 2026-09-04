import pytest

from mcp_toolset.main import scaffold


def test_scaffold_basic_toolset(tmp_path):
    written = scaffold(tmp_path, "my-toolset", with_ui=False)
    base = tmp_path / "toolsets" / "my-toolset"
    assert (base / "pyproject.toml").is_file()
    assert (base / "src" / "my_toolset" / "tools.py").is_file()
    assert (base / "tests" / "test_my_toolset.py").is_file()
    # No UI files without --with-ui.
    assert not (base / "ui").exists()
    assert all(path.is_file() for path in written)

    pyproject = (base / "pyproject.toml").read_text()
    # Depends on the extracted package, not the old workspace member.
    assert "mcp-toolsets-runtime" in pyproject
    assert "workspace = true" not in pyproject
    # The tool still imports the mcp_runtime module (name preserved).
    assert (
        "from mcp_runtime.tool_result import ToolResult"
        in (base / "src" / "my_toolset" / "tools.py").read_text()
    )


def test_scaffold_with_ui_uses_the_npm_bridge(tmp_path):
    scaffold(tmp_path, "mapped", with_ui=True)
    ui = tmp_path / "toolsets" / "mapped" / "ui"
    assert (ui / "package.json").is_file()
    assert (ui / "src" / "panel.tsx").is_file()
    # The view bridge comes from the npm package — no vendored host.ts.
    assert not (ui / "src" / "host.ts").exists()
    assert '"@developmentseed/mcp-view"' in (ui / "package.json").read_text()
    assert 'from "@developmentseed/mcp-view"' in (ui / "src" / "panel.tsx").read_text()
    # VIEWS wiring + wheel artifacts for the built bundle.
    assert (
        'VIEWS = {"example": "panel"}'
        in (
            tmp_path / "toolsets" / "mapped" / "src" / "mapped" / "tools.py"
        ).read_text()
    )
    assert (
        "artifacts" in (tmp_path / "toolsets" / "mapped" / "pyproject.toml").read_text()
    )


def test_scaffold_rejects_bad_name(tmp_path):
    with pytest.raises(ValueError, match="kebab-case"):
        scaffold(tmp_path, "Bad_Name", with_ui=False)


def test_scaffold_refuses_existing(tmp_path):
    scaffold(tmp_path, "dup", with_ui=False)
    with pytest.raises(ValueError, match="already exists"):
        scaffold(tmp_path, "dup", with_ui=False)


def deployment_files(base):
    return sorted(p.name for p in base.glob("toolset*.yaml"))


def test_scaffold_writes_the_helm_file_for_a_charts_repo(tmp_path):
    (tmp_path / "charts").mkdir()
    scaffold(tmp_path, "charted", with_ui=False)
    assert deployment_files(tmp_path / "toolsets" / "charted") == ["toolset.yaml"]


def test_scaffold_writes_the_aws_file_for_a_stack_repo(tmp_path):
    """A repo that chose AWS at bootstrap has no charts/, so a Helm values file
    would name a directory that is gone — inert until someone edits it."""
    (tmp_path / "infra").mkdir()
    scaffold(tmp_path, "stacked", with_ui=False)
    base = tmp_path / "toolsets" / "stacked"
    assert deployment_files(base) == ["toolset.aws.yaml"]
    assert "charts" not in (base / "toolset.aws.yaml").read_text()


def test_scaffold_writes_both_in_the_template_repo(tmp_path):
    (tmp_path / "charts").mkdir()
    (tmp_path / "infra").mkdir()
    scaffold(tmp_path, "both-ways", with_ui=False)
    assert deployment_files(tmp_path / "toolsets" / "both-ways") == [
        "toolset.aws.yaml",
        "toolset.yaml",
    ]


def test_scaffold_falls_back_to_the_helm_file(tmp_path):
    """Neither directory: keep what every consumer got before this existed."""
    scaffold(tmp_path, "plain", with_ui=False)
    assert deployment_files(tmp_path / "toolsets" / "plain") == ["toolset.yaml"]


def test_the_aws_file_names_the_toolset_in_its_example_path(tmp_path):
    (tmp_path / "infra").mkdir()
    scaffold(tmp_path, "stacked", with_ui=False)
    written = (tmp_path / "toolsets" / "stacked" / "toolset.aws.yaml").read_text()
    assert "/mcp-toolsets/<instance>/stacked/api-token" in written


def test_every_written_path_is_returned(tmp_path):
    """The command prints what it wrote, so a file it writes but omits from the
    list is one the user is never told about."""
    (tmp_path / "charts").mkdir()
    (tmp_path / "infra").mkdir()
    written = scaffold(tmp_path, "listed", with_ui=False)
    base = tmp_path / "toolsets" / "listed"
    assert set(deployment_files(base)) == {
        path.name for path in written if path.name.startswith("toolset")
    }
