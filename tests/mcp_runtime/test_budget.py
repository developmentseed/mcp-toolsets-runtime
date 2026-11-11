import json
import logging
import warnings
from typing import NotRequired

import pytest
from langchain_core.tools import tool
from mcp.server.mcpserver import MCPServer
from pydantic import BaseModel

from mcp_runtime import budget
from mcp_runtime.bridge import to_mcp_tool
from mcp_runtime.budget import (
    StubMessageWarning,
    assert_result_fits,
    check_definitions,
    definitions_size,
    result_size,
)
from mcp_runtime.tool_result import ToolResult


class BlobResult(ToolResult):
    blob: NotRequired[str]


@tool
async def blob(size: int) -> BlobResult:
    """Return a blob of the given size."""
    return BlobResult(message=f"A blob of {size} characters.", blob="x" * size)


def test_result_size_counts_the_text_and_the_structured_copy():
    result = {"message": "hi", "items": [1, 2]}
    text = json.dumps(result, indent=2)
    structured = json.dumps(result, separators=(",", ":"))
    assert result_size(result) == len(text) + len(structured)


async def test_a_result_over_budget_is_logged(monkeypatch, caplog):
    monkeypatch.setattr(budget, "RESULT_BUDGET", 1_000)
    converted = to_mcp_tool(blob)

    with caplog.at_level(logging.WARNING, logger="mcp_runtime.budget"):
        await converted.run({"size": 100}, None)
        assert not caplog.records
        await converted.run({"size": 1_000}, None)

    [record] = caplog.records
    assert "'blob'" in record.getMessage()
    assert "MCP_RESULT_BUDGET" in record.getMessage()


def test_definitions_over_budget_are_logged(monkeypatch, caplog):
    tools = [to_mcp_tool(blob)]
    monkeypatch.setattr(budget, "DEFINITIONS_BUDGET", definitions_size(tools) - 1)

    with caplog.at_level(logging.WARNING, logger="mcp_runtime.budget"):
        check_definitions("demo", tools)

    [record] = caplog.records
    assert "'demo'" in record.getMessage()


async def test_definitions_size_is_what_tools_list_sends():
    tools = [to_mcp_tool(blob)]
    listed = await MCPServer(name="demo", tools=tools).list_tools()
    sent = sum(
        len(item.model_dump_json(by_alias=True, exclude_none=True)) for item in listed
    )
    assert definitions_size(tools) == sent


def test_the_docstring_is_sent_once():
    converted = to_mcp_tool(blob)
    assert converted.description == "Return a blob of the given size."
    assert "description" not in converted.parameters


def test_a_multiline_docstring_is_sent_once():
    @tool
    async def search(query: str) -> ToolResult:
        """Search the catalogue.

        Use for a dataset the user names.
        """
        return ToolResult(message=query)

    converted = to_mcp_tool(search)
    assert "Use for a dataset" in converted.description
    assert "description" not in converted.parameters


def test_a_hand_written_args_schema_does_not_repeat_it_either():
    class SearchArgs(BaseModel):
        """What to search for."""

        query: str

    @tool(args_schema=SearchArgs)
    async def search(query: str) -> ToolResult:
        """Search the catalogue."""
        return ToolResult(message=query)

    assert "description" not in to_mcp_tool(search).parameters


def test_assert_result_fits_fails_over_budget():
    with pytest.raises(AssertionError, match="over the budget of 10"):
        assert_result_fits({"message": "A long enough message, " * 4}, budget=10)


def test_assert_result_fits_warns_on_a_stub_message():
    with pytest.warns(StubMessageWarning, match="records"):
        assert_result_fits(
            {
                "message": "Found 6 records.",
                "records": [{"id": n, "name": f"record {n}"} for n in range(6)],
            }
        )


@pytest.mark.parametrize(
    "result",
    [
        {"message": "Done."},
        {"message": "Hello, dev!", "name": "dev"},
        {"message": "Found 6 records: " + ", ".join(["a record"] * 10), "records": []},
        {"error": "bad_query", "detail": "No such dataset."},
    ],
)
def test_assert_result_fits_is_quiet_otherwise(result):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert_result_fits(result)
