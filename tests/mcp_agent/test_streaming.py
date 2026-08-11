"""``stream_turn``: one turn delivered as it happens.

``StreamingScriptedModel`` lives here for other suites to import, the way
``tests.mcp_state.test_injection`` is shared. It exists because
``GenericFakeChatModel`` — the stub the rest of these tests use — drops
``tool_calls`` on its streaming path: the moment ``messages`` is among the
stream modes LangGraph streams the model, so an agent driven by that stub emits
no tool call, runs no tool, produces no artifact, and a test asserting on
receipts passes while exercising nothing at all. Hence
``test_a_tool_actually_ran``, which is the guard against that whole class of
false green.
"""

import json
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver

from mcp_agent.main import run_turn, with_session_state
from mcp_agent.streaming import (
    AnswerChunk,
    ToolFinished,
    ToolStarted,
    TurnFinished,
    stream_turn,
)
from mcp_runtime.declarations import CONSUMES_META_KEY, PRODUCES_META_KEY
from mcp_runtime.kinds import GEOJSON_AREA_OF_INTEREST

AOI = {"type": "FeatureCollection", "features": [{"id": "polygon"}]}
STATE_KEY = "dataset-search/geometry"


class StreamingScriptedModel(BaseChatModel):
    """Replays a script, streaming it the way a real provider does.

    Tool calls go out as real ``tool_call_chunks`` so a tool actually runs, and
    text goes out word by word so token order is observable.
    """

    script: list[BaseMessage]
    index: int = 0

    @property
    def _llm_type(self) -> str:
        return "streaming-scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        """The non-streaming path, so the same script can drive ``run_turn`` —
        which is how ``stream_turn`` is checked against the sibling it claims to
        be. A real provider serves both; a stub that served only one would make
        that comparison impossible to write."""
        message = self.script[self.index]
        self.index += 1
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        message = self.script[self.index]
        self.index += 1
        if calls := (getattr(message, "tool_calls", None) or []):
            for position, call in enumerate(calls):
                yield ChatGenerationChunk(
                    message=AIMessageChunk(
                        content="",
                        tool_call_chunks=[
                            {
                                "name": call["name"],
                                "args": json.dumps(call.get("args") or {}),
                                "id": call.get("id"),
                                "index": position,
                                "type": "tool_call_chunk",
                            }
                        ],
                    )
                )
            return
        if isinstance(message.content, list):
            # Structured content (reference blocks and the like) arrives whole:
            # splitting it would change its shape, not just its timing.
            yield ChatGenerationChunk(message=AIMessageChunk(content=message.content))
            return
        # Split so the chunks concatenate back to exactly the original text:
        # a trailing separator on the last word would show up in the answer.
        words = str(message.content).split(" ")
        for position, word in enumerate(words):
            suffix = " " if position < len(words) - 1 else ""
            yield ChatGenerationChunk(message=AIMessageChunk(content=word + suffix))


def _publisher() -> StructuredTool:
    """Publishes a geometry into state, tagged with its kind."""

    async def call() -> tuple[str, dict[str, Any]]:
        return "found 3", {
            "structured_content": {"message": "found 3", "geometry": AOI}
        }

    return StructuredTool(
        name="search",
        description="search",
        args_schema={"type": "object", "properties": {}},
        coroutine=call,
        response_format="content_and_artifact",
        metadata={
            "_meta": {
                PRODUCES_META_KEY: [
                    {
                        "stateKey": STATE_KEY,
                        "field": "geometry",
                        "kind": GEOJSON_AREA_OF_INTEREST,
                    }
                ]
            }
        },
    )


def _consumer() -> StructuredTool:
    """Takes that geometry on a parameter the model never sees."""

    async def call(**arguments: Any) -> tuple[str, dict[str, Any]]:
        return "clipped", {"structured_content": {"ok": True}}

    return StructuredTool(
        name="clip",
        description="clip",
        args_schema={"type": "object", "properties": {"aoi": {"type": "object"}}},
        coroutine=call,
        response_format="content_and_artifact",
        metadata={
            "_meta": {
                CONSUMES_META_KEY: [
                    {
                        "parameter": "aoi",
                        "kind": GEOJSON_AREA_OF_INTEREST,
                        "required": True,
                        "modelGeneratable": False,
                    }
                ]
            }
        },
    )


def _tool_call(name: str, call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": {}, "id": call_id, "type": "tool_call"}],
    )


ANSWER = "Clipped chirps to your area."


def _agent(script: list[BaseMessage] | None = None) -> Any:
    if script is None:
        script = [
            _tool_call("search", "c1"),
            _tool_call("clip", "c2"),
            AIMessage(content=ANSWER),
        ]
    agent, _ = with_session_state(
        StreamingScriptedModel(script=script),
        [_publisher(), _consumer()],
        InMemorySaver(),
    )
    return agent


async def _collect(agent: Any, text: str = "clip chirps", thread: str = "t1") -> list:
    return [event async for event in stream_turn(agent, text, thread)]


async def test_a_tool_actually_ran():
    """The guard on every other assertion here. With a stub that drops
    ``tool_calls`` when streaming, no tool runs and the receipt tests below pass
    against an empty stream."""
    events = await _collect(_agent())

    started = [event for event in events if isinstance(event, ToolStarted)]
    finished = [event for event in events if isinstance(event, ToolFinished)]
    assert [event.name for event in started] == ["search", "clip"]
    assert [event.name for event in finished] == ["search", "clip"]
    assert [event.id for event in started] == ["c1", "c2"]


async def test_tokens_carry_the_answer_and_not_the_tool_output():
    """Tool results reach the message channel too, and they have content. A loop
    that yields every chunk with text prints tool output into the answer."""
    events = await _collect(_agent())

    streamed = "".join(event.text for event in events if isinstance(event, AnswerChunk))
    assert streamed.strip() == ANSWER
    assert "found 3" not in streamed
    assert "clipped" not in streamed
    assert "state updated" not in streamed


async def test_tokens_arrive_before_the_turn_finishes():
    """Otherwise this is `run_turn` with extra steps."""
    events = await _collect(_agent())

    kinds = [type(event).__name__ for event in events]
    assert kinds[-1] == "TurnFinished"
    assert kinds.index("AnswerChunk") < len(kinds) - 1
    assert kinds.index("ToolFinished") < kinds.index("AnswerChunk")


async def test_a_declared_fill_is_reported_on_the_tool_that_received_it():
    events = await _collect(_agent())

    clip = next(
        event
        for event in events
        if isinstance(event, ToolFinished) and event.name == "clip"
    )
    assert clip.received == {
        "aoi": {
            "key": STATE_KEY,
            "via": "declaration",
            "kind": GEOJSON_AREA_OF_INTEREST,
            "tool": "search",
        }
    }


def test_the_model_never_saw_the_filled_parameter():
    """Not a streaming claim as such, but the reason `received` has to exist:
    the argument is absent from the call, so the tool step would otherwise show
    a clip that ran against nothing."""
    call = _tool_call("clip", "c2")
    assert "aoi" not in (call.tool_calls[0]["args"] or {})


async def test_what_a_tool_published_is_reported_too():
    events = await _collect(_agent())

    search = next(
        event
        for event in events
        if isinstance(event, ToolFinished) and event.name == "search"
    )
    assert search.published == {"geometry": STATE_KEY}
    assert search.received == {}


async def test_the_artifact_is_passed_through_for_view_props():
    events = await _collect(_agent())

    search = next(
        event
        for event in events
        if isinstance(event, ToolFinished) and event.name == "search"
    )
    assert search.artifact["structured_content"] == {"message": "found 3"}


async def test_the_final_state_is_read_back_not_accumulated():
    """``tool_state`` reaches the update channel only on the updates that changed
    it — the second tool call carries none — so the turn's state cannot be taken
    from the last update."""
    events = await _collect(_agent())

    finished = events[-1]
    assert isinstance(finished, TurnFinished)
    assert finished.result.sidecar is not None
    assert STATE_KEY in finished.result.sidecar
    assert finished.result.sidecar[STATE_KEY]["value"] == AOI


async def test_it_ends_where_run_turn_would_have():
    """The sibling claim, asserted rather than asserted-about: the same script
    through both paths produces the same result."""
    streamed = await _collect(_agent())
    invoked = await run_turn(_agent(), "clip chirps", "t1")

    result = streamed[-1].result
    assert result.answer == invoked.answer == ANSWER
    assert result.citations == invoked.citations
    assert [type(m).__name__ for m in result.new_messages] == [
        type(m).__name__ for m in invoked.new_messages
    ]
    assert (result.sidecar or {}).keys() == (invoked.sidecar or {}).keys()


async def test_citations_reach_the_end():
    """Structured content survives the stream intact, so the ids are still there
    to be read. The reference rides a text block, which is where Mistral puts it
    — see ``test_citations_survive_the_mistral_normalisation``."""
    script = [
        AIMessage(
            content=[
                {"type": "text", "text": "Two datasets match. "},
                {
                    "type": "text",
                    "text": "See the catalogue.",
                    "reference": {"reference_ids": ["chirps", "era5"]},
                },
            ]
        )
    ]
    events = await _collect(_agent(script))

    assert events[-1].result.citations == ["chirps", "era5"]
    assert events[-1].result.answer == "Two datasets match. See the catalogue."


async def test_a_second_turn_continues_the_thread():
    """Only the new message is sent; the transcript lives in the checkpointer."""
    agent = _agent(
        [AIMessage(content="First answer."), AIMessage(content="Second answer.")]
    )

    first = await _collect(agent, "one", "t1")
    second = await _collect(agent, "two", "t1")

    assert first[-1].result.answer == "First answer."
    assert second[-1].result.answer == "Second answer."
    # Four: two human turns and two replies. `new_messages` is only the latest.
    assert len(second[-1].result.history) == 4
    assert [m.text for m in second[-1].result.new_messages] == ["Second answer."]


async def test_a_turn_with_no_tools_yields_only_text():
    events = await _collect(_agent([AIMessage(content="No tools needed.")]))

    assert not [event for event in events if isinstance(event, ToolStarted)]
    assert events[-1].result.answer == "No tools needed."
    assert (
        "".join(
            event.text for event in events if isinstance(event, AnswerChunk)
        ).strip()
        == "No tools needed."
    )
