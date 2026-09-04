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


def written_config_files(base):
    return sorted(p.name for p in base.glob("toolset*.yaml"))


def test_scaffold_writes_the_helm_file_for_a_charts_repo(tmp_path):
    (tmp_path / "charts").mkdir()
    scaffold(tmp_path, "charted", with_ui=False)
    assert written_config_files(tmp_path / "toolsets" / "charted") == ["toolset.yaml"]


def test_scaffold_writes_the_aws_file_for_a_stack_repo(tmp_path):
    """A repo that chose AWS at bootstrap has no charts/, so a Helm values file
    would name a directory that is gone — inert until someone edits it."""
    (tmp_path / "infra").mkdir()
    scaffold(tmp_path, "stacked", with_ui=False)
    base = tmp_path / "toolsets" / "stacked"
    assert written_config_files(base) == ["toolset.aws.yaml"]
    assert "charts" not in (base / "toolset.aws.yaml").read_text()


def test_scaffold_writes_both_in_the_template_repo(tmp_path):
    (tmp_path / "charts").mkdir()
    (tmp_path / "infra").mkdir()
    scaffold(tmp_path, "both-ways", with_ui=False)
    assert written_config_files(tmp_path / "toolsets" / "both-ways") == [
        "toolset.aws.yaml",
        "toolset.yaml",
    ]


def test_scaffold_falls_back_to_the_helm_file(tmp_path):
    """Neither directory: keep what every consumer got before this existed."""
    scaffold(tmp_path, "plain", with_ui=False)
    assert written_config_files(tmp_path / "toolsets" / "plain") == ["toolset.yaml"]


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
    assert set(written_config_files(base)) == {
        path.name for path in written if path.name.startswith("toolset")
    }


def declare(tmp_path, entries: str) -> None:
    """Write a root pyproject declaring what a new toolset's config files are."""
    (tmp_path / "pyproject.toml").write_text(
        f'[project]\nname = "x"\nversion = "0"\n\n'
        f"[tool.mcp-toolset]\ndeployment-config = {entries}\n"
    )


def test_a_repo_declares_its_own_deployment_files(tmp_path):
    """The file belongs to the repo that reads it: it names that repo's
    directories, so that repo says what it is and where its template lives."""
    (tmp_path / "infra" / "k8s").mkdir(parents=True)
    (tmp_path / "infra" / "k8s" / "toolset.template.yaml").write_text(
        "# Helm values for __NAME__ (see infra/k8s/charts/mcp-toolset/values.yaml).\n{}\n"
    )
    declare(
        tmp_path,
        '[{ path = "toolset.yaml", template = "infra/k8s/toolset.template.yaml" }]',
    )

    scaffold(tmp_path, "declared", with_ui=False)

    written = (tmp_path / "toolsets" / "declared" / "toolset.yaml").read_text()
    assert "Helm values for declared" in written
    assert "infra/k8s/charts" in written
    assert not (tmp_path / "toolsets" / "declared" / "toolset.aws.yaml").exists()


def test_a_declaration_wins_over_the_directories(tmp_path):
    """Otherwise a repo that moved its charts would be second-guessed by a
    marker that no longer means what it did."""
    (tmp_path / "charts").mkdir()
    (tmp_path / "custom.yaml").write_text("chosen: __NAME__\n")
    declare(tmp_path, '[{ path = "deploy.yaml", template = "custom.yaml" }]')

    scaffold(tmp_path, "picky", with_ui=False)

    base = tmp_path / "toolsets" / "picky"
    assert (base / "deploy.yaml").read_text() == "chosen: picky\n"
    assert not (base / "toolset.yaml").exists()


def test_an_empty_declaration_writes_no_deployment_file(tmp_path):
    """Declaring nothing and declaring an empty list are different answers."""
    (tmp_path / "charts").mkdir()
    declare(tmp_path, "[]")

    scaffold(tmp_path, "bare", with_ui=False)

    assert written_config_files(tmp_path / "toolsets" / "bare") == []


def test_a_missing_template_is_refused(tmp_path):
    """Loudly: the alternative is a toolset scaffolded without the one file its
    deployment reads."""
    declare(tmp_path, '[{ path = "toolset.yaml", template = "gone.yaml" }]')
    with pytest.raises(ValueError, match="template not found"):
        scaffold(tmp_path, "broken", with_ui=False)


def test_an_entry_missing_its_keys_is_refused(tmp_path):
    declare(tmp_path, '[{ path = "toolset.yaml" }]')
    with pytest.raises(ValueError, match="needs a path and a template"):
        scaffold(tmp_path, "broken", with_ui=False)


def test_a_path_escaping_the_toolset_is_refused(tmp_path):
    (tmp_path / "custom.yaml").write_text("{}\n")
    declare(tmp_path, '[{ path = "../../etc/passwd", template = "custom.yaml" }]')
    with pytest.raises(ValueError, match="stay inside the toolset"):
        scaffold(tmp_path, "escapee", with_ui=False)


def test_a_declaration_of_the_wrong_shape_is_refused(tmp_path):
    declare(tmp_path, '"toolset.yaml"')
    with pytest.raises(ValueError, match="must be a list"):
        scaffold(tmp_path, "wrong", with_ui=False)
