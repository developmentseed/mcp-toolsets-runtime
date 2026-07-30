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
