"""``interrupt_gate`` and the turns around it: stopping, answering, cancelling.

Over a real graph and a real checkpointer throughout, because every rule in
:mod:`mcp_agent.interrupts` is a fact about LangGraph rather than about this
code — a replayed cached write, an interrupt dropped by new input — and a stub
would agree with whatever the code assumed.
"""

from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import ValidationError

from mcp_agent.interrupt_gate import (
    TOOL_NAME,
    CHOICE,
    INPUT_REQUIRED,
    NOT_ANSWERED,
    Option,
    answer_text,
    argument_errors,
    make_interrupt_gate,
    options_of,
    response_from_reply,
    response_schema,
)
from mcp_agent.interrupts import ResumeMismatch
from mcp_agent.main import InterruptGateSettings, run_turn, with_session_state
from mcp_agent.streaming import ToolFinished, TurnFinished, stream_turn
from tests.mcp_agent.test_agent_state import _record_create_agent
from tests.mcp_agent.test_streaming import (
    STATE_KEY,
    StreamingScriptedModel,
    _publisher,
)

OPTIONS = [
    {"value": "ESP.2_1", "label": "Cordoba, Spain"},
    {"value": "ARG.6_1", "label": "Cordoba, Argentina"},
]


def _calls(*calls: tuple[str, str, dict[str, Any]]) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {"name": name, "args": args, "id": call_id, "type": "tool_call"}
            for name, call_id, args in calls
        ],
    )


def _ask(call_id: str = "q1", **args: Any) -> tuple[str, str, dict[str, Any]]:
    return (
        TOOL_NAME,
        call_id,
        {"question": "Which Cordoba?", "options": OPTIONS, **args},
    )


def _agent(*script: BaseMessage, runs: dict[str, int] | None = None) -> Any:
    """The state-wired agent with ``interrupt_gate``, and a publisher that counts."""
    publisher = _publisher()
    original = publisher.coroutine
    assert original is not None

    async def counted(q: str = "") -> Any:
        if runs is not None:
            runs["search"] = runs.get("search", 0) + 1
        return await original(q)

    publisher.coroutine = counted
    model = StreamingScriptedModel(script=list(script))
    agent = with_session_state(model, [publisher], InMemorySaver())
    agent.scripted = model  # to count model calls
    return agent


async def _turn(agent: Any, text: str | None, **kwargs: Any) -> list:
    return [event async for event in stream_turn(agent, text, "t1", **kwargs)]


def _result(events: list) -> Any:
    assert isinstance(events[-1], TurnFinished)
    return events[-1].result


def _chose(value: str) -> dict[str, Any]:
    return {"status": "resolved", "payload": {CHOICE: value}}


# --- the tool --------------------------------------------------------------


def test_the_arguments_are_checked_before_anything_is_asked():
    one = [Option(value="a", label="A")]
    same = [Option(value="a", label="A"), Option(value="a", label="B")]
    assert argument_errors("Which?", one) == ["give 2 to 10 options, not 1"]
    five = [Option(value=str(n), label=str(n)) for n in range(5)]
    assert argument_errors("Which?", five) == []
    assert argument_errors("Which?", five, max_options=4) == [
        "give 2 to 4 options, not 5"
    ]
    assert argument_errors("Which?", one, min_options=1) == []
    assert argument_errors(" ", [*one, Option(value="b", label="B")]) == [
        "question is empty"
    ]
    assert argument_errors("Which?", same) == ["two options have the same value"]


def test_one_choice_is_a_titled_one_of():
    options = [Option.model_validate(option) for option in OPTIONS]
    schema = response_schema(options)
    assert schema["required"] == [CHOICE]
    assert schema["properties"][CHOICE]["oneOf"][0] == {
        "const": "ESP.2_1",
        "title": "Cordoba, Spain",
    }
    assert options_of(schema) == (
        [("ESP.2_1", "Cordoba, Spain"), ("ARG.6_1", "Cordoba, Argentina")],
        False,
    )


def test_several_choices_are_an_array_of_any_of():
    options = [Option.model_validate(option) for option in OPTIONS]
    schema = response_schema(options, multiple=True)
    assert schema["properties"][CHOICE]["type"] == "array"
    assert schema["properties"][CHOICE]["items"]["anyOf"][1]["const"] == "ARG.6_1"
    assert options_of(schema)[1] is True


def test_the_answer_reads_as_the_label_and_the_value():
    options = [Option.model_validate(option) for option in OPTIONS]
    assert (
        answer_text(_chose("ARG.6_1"), options)
        == "User chose: Cordoba, Argentina (value: ARG.6_1)"
    )
    both = {"status": "resolved", "payload": {CHOICE: ["ESP.2_1", "ARG.6_1"]}}
    assert answer_text(both, options).count("value:") == 2


@pytest.mark.parametrize(
    "response",
    [
        {"status": "cancelled"},
        {"status": "resolved", "payload": {CHOICE: "not an option"}},
        {"status": "resolved"},
        "a bare string",
    ],
)
def test_anything_but_a_valid_choice_is_not_an_answer(response):
    options = [Option.model_validate(option) for option in OPTIONS]
    assert answer_text(response, options) == NOT_ANSWERED


def test_a_typed_reply_picks_options_by_number():
    options = [Option.model_validate(option) for option in OPTIONS]
    one, several = response_schema(options), response_schema(options, multiple=True)

    assert response_from_reply("2", one) == _chose("ARG.6_1")
    assert response_from_reply(" ", one) == {"status": "cancelled"}
    assert response_from_reply("2, 1 2", several) == {
        "status": "resolved",
        "payload": {CHOICE: ["ARG.6_1", "ESP.2_1"]},
    }
    # Not a reply: asked again rather than guessed at.
    for reply, schema in [("3", one), ("1 2", one), ("Seville", one), ("0", several)]:
        assert response_from_reply(reply, schema) is None


async def test_a_broken_call_returns_a_sentence_and_does_not_stop_the_run():
    agent = _agent(
        _calls((TOOL_NAME, "q1", {"question": "Which?", "options": OPTIONS[:1]})),
        AIMessage(content="fine, I will guess"),
    )
    result = _result(await _turn(agent, "find cordoba"))

    assert result.interrupts == []
    assert "give 2 to 10 options" in result.new_messages[1].text
    assert result.answer == "fine, I will guess"


# --- stopping and answering -------------------------------------------------


async def test_the_turn_stops_on_the_question():
    agent = _agent(_calls(_ask()), AIMessage(content="never reached"))
    result = _result(await _turn(agent, "find cordoba"))

    assert agent.scripted.index == 1  # the model was not called again
    assert result.answer == ""
    [asked] = result.interrupts
    assert asked.value["reason"] == INPUT_REQUIRED
    assert asked.value["message"] == "Which Cordoba?"
    # The call it stopped on, so a client can draw the question beside it.
    assert asked.value["toolCallId"] == "q1"
    assert CHOICE in asked.value["responseSchema"]["properties"]


async def test_the_answer_is_the_result_of_the_call_that_asked():
    agent = _agent(_calls(_ask()), AIMessage(content="Cordoba, Argentina it is."))
    [asked] = _result(await _turn(agent, "find cordoba")).interrupts

    events = await _turn(agent, None, resume={asked.id: _chose("ARG.6_1")})
    result = _result(events)

    [finished] = [event for event in events if isinstance(event, ToolFinished)]
    assert finished.id == "q1"
    assert finished.content == "User chose: Cordoba, Argentina (value: ARG.6_1)"
    assert result.answer == "Cordoba, Argentina it is."
    assert result.interrupts == []
    # An answer adds no user message: the thread is the question, the tool
    # result, and the reply.
    assert [m.type for m in result.history] == ["human", "ai", "tool", "ai"]
    assert [m.type for m in result.new_messages] == ["tool", "ai"]


async def test_a_cancelled_question_tells_the_model_so():
    agent = _agent(_calls(_ask()), AIMessage(content="ok, never mind"))
    [asked] = _result(await _turn(agent, "find cordoba")).interrupts

    result = _result(
        await _turn(agent, None, resume={asked.id: {"status": "cancelled"}})
    )

    assert result.new_messages[0].text == NOT_ANSWERED
    assert result.answer == "ok, never mind"


async def test_a_tool_that_finished_beside_the_question_is_not_run_again():
    """Its result is a pending write: kept, not re-run, and not re-announced."""
    runs: dict[str, int] = {}
    agent = _agent(
        _calls(("search", "s1", {"q": "cordoba"}), _ask()),
        AIMessage(content="done"),
        runs=runs,
    )
    first = await _turn(agent, "find cordoba")
    [asked] = _result(first).interrupts
    second = await _turn(agent, None, resume={asked.id: _chose("ESP.2_1")})

    assert runs["search"] == 1
    finished = [
        event.id for event in [*first, *second] if isinstance(event, ToolFinished)
    ]
    assert finished == ["s1", "q1"]
    assert STATE_KEY in (_result(second).sidecar or {})


async def test_two_questions_in_one_message_are_answered_together():
    agent = _agent(
        _calls(_ask("q1"), _ask("q2", question="And which year?")),
        AIMessage(content="both answered"),
    )
    asked = _result(await _turn(agent, "find cordoba")).interrupts
    assert len(asked) == 2

    with pytest.raises(ResumeMismatch, match="missing"):
        await _turn(agent, None, resume={asked[0].id: _chose("ESP.2_1")})

    result = _result(
        await _turn(
            agent,
            None,
            resume={item.id: _chose("ESP.2_1") for item in asked},
        )
    )
    assert result.answer == "both answered"


async def test_a_resume_must_name_the_open_interrupts():
    agent = _agent(_calls(_ask()), AIMessage(content="x"))

    with pytest.raises(ResumeMismatch, match="no open interrupt"):
        await _turn(agent, None, resume={"nope": _chose("ESP.2_1")})

    await _turn(agent, "find cordoba")
    with pytest.raises(ResumeMismatch, match="not an open interrupt"):
        await _turn(agent, None, resume={"nope": _chose("ESP.2_1")})
    with pytest.raises(ResumeMismatch, match="waiting for an answer"):
        await _turn(agent, None)


# --- a new message while a question is open ---------------------------------


async def test_a_new_message_on_a_paused_thread_is_refused():
    """LangGraph would take it and drop the question, leaving a tool call with
    no result — which the next model call is refused for."""
    agent = _agent(_calls(_ask()), AIMessage(content="never reached"))
    [asked] = _result(await _turn(agent, "find cordoba")).interrupts

    with pytest.raises(ResumeMismatch, match="waiting for an answer"):
        await _turn(agent, "actually, Seville")

    # Nothing changed: the question is still open, and cancelling it is how
    # the person moves on.
    snapshot = await agent.aget_state({"configurable": {"thread_id": "t1"}})
    assert [item.id for item in snapshot.interrupts] == [asked.id]


async def test_a_turn_answers_or_asks_not_both():
    agent = _agent(_calls(_ask()), AIMessage(content="x"))
    [asked] = _result(await _turn(agent, "find cordoba")).interrupts

    with pytest.raises(ValueError, match="not both"):
        await _turn(agent, "and Seville", resume={asked.id: _chose("ESP.2_1")})


# --- run_turn keeps the same rules ------------------------------------------


async def test_run_turn_stops_answers_and_refuses_the_same_way():
    agent = _agent(
        _calls(_ask()),
        AIMessage(content="Argentina."),
    )
    stopped = await run_turn(agent, "find cordoba", "t1")
    [asked] = stopped.interrupts
    assert asked.value["toolCallId"] == "q1"
    with pytest.raises(ResumeMismatch):
        await run_turn(agent, "Seville instead", "t1")

    answered = await run_turn(agent, None, "t1", resume={asked.id: _chose("ARG.6_1")})
    assert answered.interrupts == []
    assert answered.answer == "Argentina."
    assert [m.type for m in answered.new_messages] == ["tool", "ai"]


# --- wiring -----------------------------------------------------------------


def test_the_tool_is_added_only_where_a_run_can_pause(monkeypatch):
    recorded = _record_create_agent(monkeypatch)
    with_session_state("model", [], InMemorySaver())
    assert TOOL_NAME in recorded["tools"]

    with_session_state("model", [])
    assert TOOL_NAME not in recorded["tools"]

    with_session_state("model", [], InMemorySaver(), interrupt_gate=False)
    assert TOOL_NAME not in recorded["tools"]


def test_a_host_passing_its_own_interrupt_gate_keeps_it(monkeypatch):
    recorded = _record_create_agent(monkeypatch)
    own = make_interrupt_gate()
    with_session_state("model", [], InMemorySaver(), extra_tools=[own])
    assert recorded["tools"].count(TOOL_NAME) == 1


def test_interrupt_gate_is_on_unless_switched_off(monkeypatch):
    monkeypatch.delenv("MCP_AGENT_INTERRUPT_GATE", raising=False)
    assert InterruptGateSettings(_env_file=None).mcp_agent_interrupt_gate is True
    monkeypatch.setenv("MCP_AGENT_INTERRUPT_GATE", "0")
    assert InterruptGateSettings(_env_file=None).mcp_agent_interrupt_gate is False


def test_the_option_limits_come_from_the_environment(monkeypatch):
    settings = InterruptGateSettings(_env_file=None)
    assert (
        settings.mcp_agent_interrupt_gate_min_options,
        settings.mcp_agent_interrupt_gate_max_options,
    ) == (2, 10)

    monkeypatch.setenv("MCP_AGENT_INTERRUPT_GATE_MIN_OPTIONS", "3")
    monkeypatch.setenv("MCP_AGENT_INTERRUPT_GATE_MAX_OPTIONS", "4")
    recorded = _record_create_agent(monkeypatch)
    with_session_state("model", [], InMemorySaver())
    [gate] = [tool for tool in recorded["built"] if tool.name == TOOL_NAME]
    assert "between 3 and 4 options" in gate.description

    # A maximum below the minimum is a broken setting, not a silent swap.
    monkeypatch.setenv("MCP_AGENT_INTERRUPT_GATE_MAX_OPTIONS", "2")
    with pytest.raises(ValidationError):
        InterruptGateSettings(_env_file=None)


def test_the_terminal_asks_again_until_the_reply_is_an_option(monkeypatch):
    from mcp_agent import main
    from mcp_agent.interrupts import PendingInterrupt

    options = [Option.model_validate(option) for option in OPTIONS]
    asked = PendingInterrupt(
        id="i1",
        value={"message": "Which Cordoba?", "responseSchema": response_schema(options)},
    )
    replies = iter(["Seville", "3", "2"])
    monkeypatch.setattr(main.console, "input", lambda prompt: next(replies))

    assert main.ask_in_terminal(asked) == _chose("ARG.6_1")


def test_the_terminal_cancels_what_it_cannot_draw(monkeypatch):
    from mcp_agent import main
    from mcp_agent.interrupts import PendingInterrupt

    def no_input(prompt: str) -> str:
        raise EOFError

    monkeypatch.setattr(main.console, "input", no_input)
    assert main.ask_in_terminal(PendingInterrupt("i1", "raw")) == {
        "status": "cancelled"
    }
    options = [Option.model_validate(option) for option in OPTIONS]
    schema = {"responseSchema": response_schema(options)}
    assert main.ask_in_terminal(PendingInterrupt("i2", schema)) == {
        "status": "cancelled"
    }


def test_the_default_prompt_tells_the_model_to_ask_only_when_it_can(monkeypatch):
    from mcp_agent.interrupt_gate import INTERRUPT_GATE_PROMPT
    from mcp_agent.main import SYSTEM_PROMPT

    recorded = _record_create_agent(monkeypatch)
    with_session_state("model", [], InMemorySaver())
    assert recorded["system_prompt"] == f"{SYSTEM_PROMPT}\n\n{INTERRUPT_GATE_PROMPT}"

    with_session_state("model", [], InMemorySaver(), interrupt_gate=False)
    assert recorded["system_prompt"] == SYSTEM_PROMPT

    with_session_state("model", [])
    assert recorded["system_prompt"] == SYSTEM_PROMPT

    # A host's own prompt is used verbatim; appending is the host's call.
    with_session_state("model", [], InMemorySaver(), system_prompt="be terse")
    assert recorded["system_prompt"] == "be terse"


def test_the_description_states_the_limits_in_force():
    assert "between 2 and 10 options" in make_interrupt_gate().description
    assert "between 2 and 6 options" in make_interrupt_gate(max_options=6).description
    assert "between 1 and 6 options" in make_interrupt_gate(1, 6).description
