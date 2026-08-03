"""Resolving credential headers from flags and the environment, and citations.

Credentials ride the MCP transport rather than the conversation, so nothing
downstream reports a header that never got a value — which is why resolution
is tested directly here.
"""

import pytest
import typer
from langchain_core.messages import AIMessage

from mcp_agent.main import (
    answer_citations,
    credential_env_var,
    headers_from_flags,
    resolve_credentials,
)

REQUIRED = {"demo": ["x-demo-token"], "archive": ["x-archive-key"]}


def test_a_header_becomes_its_env_var():
    assert credential_env_var("x-demo-token") == "X_DEMO_TOKEN"


def test_flags_parse_to_a_lowercased_map():
    assert headers_from_flags(["X-Demo-Token=abc", "x-archive-key=def"]) == {
        "x-demo-token": "abc",
        "x-archive-key": "def",
    }


def test_a_later_flag_wins():
    assert headers_from_flags(["x-a=1", "x-a=2"]) == {"x-a": "2"}


def test_a_value_may_contain_equals_signs():
    assert headers_from_flags(["x-a=b=c"]) == {"x-a": "b=c"}


@pytest.mark.parametrize("flag", ["no-equals", "=novalue"])
def test_a_malformed_flag_is_rejected(flag):
    with pytest.raises(typer.BadParameter):
        headers_from_flags([flag])


def test_a_flag_wins_over_the_environment(monkeypatch):
    monkeypatch.setenv("X_DEMO_TOKEN", "from-env")
    resolved = resolve_credentials(REQUIRED, {"x-demo-token": "from-flag"})
    assert resolved["x-demo-token"] == "from-flag"


def test_an_advertised_header_falls_back_to_its_env_var(monkeypatch):
    monkeypatch.setenv("X_DEMO_TOKEN", "from-env")
    monkeypatch.delenv("X_ARCHIVE_KEY", raising=False)
    assert resolve_credentials(REQUIRED, {}) == {"x-demo-token": "from-env"}


def test_an_exported_var_beats_a_dotenv_one(monkeypatch):
    monkeypatch.setenv("X_DEMO_TOKEN", "exported")
    resolved = resolve_credentials(REQUIRED, {}, {"x_demo_token": "dotenv"})
    assert resolved["x-demo-token"] == "exported"


def test_dotenv_supplies_a_header_the_shell_did_not(monkeypatch):
    monkeypatch.delenv("X_DEMO_TOKEN", raising=False)
    resolved = resolve_credentials(REQUIRED, {}, {"x_demo_token": "dotenv"})
    assert resolved["x-demo-token"] == "dotenv"


def test_an_undeclared_flag_still_passes_through(monkeypatch):
    """With a direct URL nothing is advertised; the per-toolset gate forwards it."""
    monkeypatch.delenv("X_DEMO_TOKEN", raising=False)
    assert resolve_credentials(None, {"x-anything": "v"}) == {"x-anything": "v"}


def test_an_unset_header_is_simply_absent(monkeypatch):
    """Absent, not empty: an empty header would look like a supplied credential."""
    monkeypatch.delenv("X_DEMO_TOKEN", raising=False)
    monkeypatch.delenv("X_ARCHIVE_KEY", raising=False)
    assert resolve_credentials(REQUIRED, {}) == {}


def test_citations_come_out_in_first_seen_order_without_duplicates():
    message = AIMessage(
        content=[
            {"type": "text", "text": "The catalogue lists it."},
            {
                "type": "text",
                "text": "Two sources.",
                "reference": {"reference_ids": ["search_datasets", "record-7"]},
            },
            {
                "type": "text",
                "text": "One again.",
                "reference": {"reference_ids": ["record-7"]},
            },
        ]
    )
    assert answer_citations(message) == ["search_datasets", "record-7"]


def test_citations_survive_the_mistral_normalisation():
    """Mistral's reference ids ride a ``reference`` key on a text block.

    They have nowhere to go in a standard ``Citation`` — which carries url,
    title and cited_text, none of which an opaque id is — so they survive
    translation as a non-standard key rather than becoming an annotation.
    """
    message = AIMessage(
        content=[
            {"type": "text", "text": "ERA5 covers it. "},
            {
                "type": "text",
                "text": "See the catalogue.",
                "reference": {"reference_ids": ["search_datasets", "era5"]},
            },
        ]
    )
    assert answer_citations(message) == ["search_datasets", "era5"]


def test_anthropics_native_citations_are_read():
    """On ``content`` Anthropic keeps a native ``citations`` key.

    Reading ``content`` would miss it entirely; ``content_blocks`` presents it
    as a standard ``citation`` annotation. No langchain-anthropic here, so the
    already-standardised blocks stand in for what its translator produces.
    """
    message = AIMessage(
        content=[
            {
                "type": "text",
                "text": "ERA5 covers it.",
                "annotations": [
                    {
                        "type": "citation",
                        "title": "ERA5 docs",
                        "cited_text": "hourly reanalysis",
                    }
                ],
            }
        ]
    )
    assert answer_citations(message) == ["ERA5 docs"]


def test_a_citation_with_only_an_excerpt_still_counts():
    message = AIMessage(
        content=[
            {
                "type": "text",
                "text": "ERA5 covers it.",
                "annotations": [{"type": "citation", "cited_text": "hourly"}],
            }
        ]
    )
    assert answer_citations(message) == ["hourly"]


def test_standard_citation_annotations_are_read():
    message = AIMessage(
        content=[
            {
                "type": "text",
                "text": "ERA5 covers it.",
                "annotations": [
                    {"type": "citation", "title": "ERA5 land", "url": "http://x"},
                    {"type": "citation", "id": "doc-2"},
                ],
            }
        ]
    )
    assert answer_citations(message) == ["ERA5 land", "doc-2"]


def test_a_reference_id_is_not_repeated_across_shapes():
    message = AIMessage(
        content=[
            {"type": "text", "text": "a", "reference": {"reference_ids": ["era5"]}},
            {
                "type": "text",
                "text": "b",
                "annotations": [{"type": "citation", "id": "era5"}],
            },
        ]
    )
    assert answer_citations(message) == ["era5"]


def test_plain_text_content_carries_no_citations():
    assert answer_citations(AIMessage(content="just prose")) == []
    assert answer_citations(AIMessage(content=[{"type": "text", "text": "hi"}])) == []


def test_malformed_reference_blocks_are_ignored():
    message = AIMessage(
        content=[
            {
                "type": "text",
                "text": "a",
                "reference": {"reference_ids": [None, "", 7, "ok"]},
            },
            {"type": "text", "text": "b", "reference": {}},
            {"type": "text", "text": "c", "annotations": [{"type": "citation"}]},
        ]
    )
    assert answer_citations(message) == ["ok"]
