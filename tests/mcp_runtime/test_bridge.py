"""The extended to_mcp_tool: derived output schemas and the ToolResult gate."""

from types import SimpleNamespace
from typing import Any, NotRequired, TypedDict

import pytest
from langchain_core.tools import tool
from mcp.types import ElicitResult, InputRequiredResult
from pydantic import BaseModel

from mcp_runtime.bridge import to_mcp_tool
from mcp_runtime.questions import AskUser, answers
from mcp_runtime.tool_result import ToolError, ToolResult


class ProbeResult(ToolResult):
    items: NotRequired[list[dict[str, Any]]]


@tool
async def probe(query: str) -> ProbeResult | ToolError:
    """Return items for a query."""
    if query == "boom":
        return ToolError(error="bad_query", detail="boom")
    if query == "empty":
        return ProbeResult(message="Nothing found.")
    return ProbeResult(message=f"Found 1 for {query!r}.", items=[{"id": query}])


@tool
def message_only(text: str) -> ToolResult:
    """Echo the text (sync tool)."""
    return ToolResult(message=text)


@tool
async def pick_year() -> ToolResult:
    """Pick a year, asking the user which."""
    if (replies := answers()) is None:
        raise AskUser(year=("Which year?", {"type": "object"}))
    year = replies["year"].content
    return ToolResult(message=f"picked {year['year']}" if year else "no year")


async def run_structured(converted, arguments: dict[str, Any], context=None) -> Any:
    """The structuredContent the server would emit for a call.

    ``context`` is the per-request context the SDK passes the tool. Only its
    ``input_responses`` are read, so a test that needs one builds that much.
    """
    result = await converted.run(arguments, context, convert_result=True)
    assert result.structured_content is not None, "no structured content was produced"
    return result.structured_content


def test_output_schema_is_object_rooted_union():
    schema = to_mcp_tool(probe).output_schema
    assert schema is not None
    assert schema["type"] == "object"
    arms = [ref["$ref"].removeprefix("#/$defs/") for ref in schema["anyOf"]]
    assert set(arms) == {"ProbeResult", "ToolError"}
    result_arm = schema["$defs"]["ProbeResult"]
    assert result_arm["properties"]["message"]["type"] == "string"
    assert "message" in result_arm["required"]
    assert schema["$defs"]["ToolError"]["required"] == ["error", "detail"]


def test_single_arm_schema_is_inlined():
    schema = to_mcp_tool(message_only).output_schema
    assert schema is not None
    assert "$ref" not in schema
    assert schema["type"] == "object"
    assert "message" in schema["required"]


async def test_structured_content_is_the_returned_dict():
    converted = to_mcp_tool(probe)
    structured = await run_structured(converted, {"query": "era5"})
    assert structured == {"message": "Found 1 for 'era5'.", "items": [{"id": "era5"}]}


async def test_absent_optional_keys_are_not_null_padded():
    structured = await run_structured(to_mcp_tool(probe), {"query": "empty"})
    assert structured == {"message": "Nothing found."}


async def test_error_returns_conform_to_the_schema():
    structured = await run_structured(to_mcp_tool(probe), {"query": "boom"})
    assert structured == {"error": "bad_query", "detail": "boom"}


async def test_sync_tool_supported():
    structured = await run_structured(to_mcp_tool(message_only), {"text": "hi"})
    assert structured == {"message": "hi"}


async def test_undeclared_keys_are_dropped_from_structured_content():
    @tool
    async def sneaky(text: str) -> ToolResult:
        """Echo the text."""
        result = ToolResult(message=text)
        result["extra"] = "undeclared"  # type: ignore[typeddict-unknown-key]
        return result

    structured = await run_structured(to_mcp_tool(sneaky), {"text": "hi"})
    assert structured == {"message": "hi"}


def test_basemodel_with_required_message_accepted():
    class ModelResult(BaseModel):
        message: str
        count: int = 0

    @tool
    async def modeled(text: str) -> ModelResult:
        """Echo the text."""
        return ModelResult(message=text)

    schema = to_mcp_tool(modeled).output_schema
    assert schema is not None
    assert "message" in schema["required"]


@pytest.mark.parametrize(
    ("annotation", "body"),
    [
        (str, "text"),
        (dict[str, Any], {"message": "hi"}),
        (list[str], ["hi"]),
        (None, "text"),
    ],
    ids=["str", "dict-str-any", "list", "missing"],
)
def test_non_contract_annotations_rejected(annotation, body):
    async def loose(text: str):  # noqa: ANN202 - annotation applied below
        """Echo the text."""
        return body

    loose.__annotations__["return"] = annotation
    if annotation is None:
        del loose.__annotations__["return"]
    with pytest.raises(RuntimeError, match="loose"):
        to_mcp_tool(tool(loose))


def test_typeddict_without_message_rejected():
    class NoMessage(TypedDict):
        data: str

    @tool
    async def messageless(text: str) -> NoMessage:
        """Echo the text."""
        return NoMessage(data=text)

    with pytest.raises(RuntimeError, match="required str 'message'"):
        to_mcp_tool(messageless)


def test_non_string_message_rejected():
    class NumericMessage(TypedDict):
        message: int

    @tool
    async def numeric(text: str) -> NumericMessage:
        """Echo the text length."""
        return NumericMessage(message=len(text))

    with pytest.raises(RuntimeError, match="required str 'message'"):
        to_mcp_tool(numeric)


def test_union_with_non_dict_arm_rejected():
    @tool
    async def mixed(text: str) -> ToolResult | str:
        """Echo the text."""
        return text

    with pytest.raises(RuntimeError, match="mixed"):
        to_mcp_tool(mixed)


async def test_an_undeclared_argument_is_refused_not_dropped():
    """Pydantic's default is to drop an argument the model did not declare.

    The caller is never told, which is the dangerous part: a parameter the model
    meant to send simply is not there, and the tool runs as though it had never
    been asked for.
    """
    converted = to_mcp_tool(probe)

    with pytest.raises(Exception, match="query_typo"):
        await converted.run({"query": "hello", "query_typo": "hello"}, None)


async def test_a_declared_argument_still_gets_through():
    converted = to_mcp_tool(probe)

    assert await run_structured(converted, {"query": "hello"}) == {
        "message": "Found 1 for 'hello'.",
        "items": [{"id": "hello"}],
    }


def test_the_input_schema_says_so_too():
    """Upstream publishes no extra policy at all, so enforcing one on its own
    would refuse calls the advertised schema allowed. The rule has to be in the
    schema for a client to apply it, or to know why it was refused."""
    assert to_mcp_tool(probe).parameters["additionalProperties"] is False


async def test_a_tool_that_asks_returns_the_question():
    result = await to_mcp_tool(pick_year).run({}, None, convert_result=True)
    assert isinstance(result, InputRequiredResult)
    assert result.input_requests is not None
    assert result.input_requests["year"].params.message == "Which year?"


async def test_the_answers_reach_the_tool_on_the_retry():
    answered = SimpleNamespace(
        input_responses={"year": ElicitResult(action="accept", content={"year": 2024})}
    )
    declined = SimpleNamespace(input_responses={"year": ElicitResult(action="decline")})
    converted = to_mcp_tool(pick_year)
    assert (await run_structured(converted, {}, answered))["message"] == "picked 2024"
    assert (await run_structured(converted, {}, declined))["message"] == "no year"


def test_the_sdk_internals_we_build_on_keep_their_shape():
    """`to_mcp_tool` builds on `mcp.server.mcpserver.utilities.func_metadata`,
    which is internal to the SDK, so a minor release can change it. This names
    exactly what we rely on, so such a release fails here first and clearly."""
    from mcp.server.mcpserver.tools import Tool as MCPTool
    from mcp.server.mcpserver.utilities.func_metadata import (
        ArgModelBase,
        FuncMetadata,
    )

    assert issubclass(ArgModelBase, BaseModel)
    assert {"arg_model", "output_model", "output_schema", "wrap_output"} <= set(
        FuncMetadata.model_fields
    )
    assert {
        "fn",
        "name",
        "description",
        "parameters",
        "fn_metadata",
        "is_async",
        "context_kwarg",
    } <= set(MCPTool.model_fields)
