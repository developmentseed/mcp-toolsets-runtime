"""``interrupt``: the agent asks the user a question, and the run stops.

The agent calls the tool with a question and its options. The tool calls
LangGraph ``interrupt()``, which ends the run. The next run carries the user's
answer, and that answer is the result of the same tool call.

The interrupt value has the AG-UI field names (``reason``, ``message``,
``toolCallId``, ``responseSchema``), so :mod:`mcp_agent_api.events` reads it
into an ``ag_ui.core.Interrupt`` as it is. It is a plain dict here because
``mcp_agent`` does not depend on ``ag_ui``, which the ``[api]`` extra installs.

This module is only the tool. The rules for stopping a turn and answering it
are in :mod:`mcp_agent.interrupts`.
"""

from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Literal, Union

from langchain_core.tools import BaseTool, InjectedToolCallId, StructuredTool
from langgraph.types import interrupt
from pydantic import BaseModel, Field, create_model

from mcp_agent.interrupts import CANCELLED, RESOLVED

#: The tool's name, as the model and every client see it.
TOOL_NAME = "interrupt"

#: AG-UI's core ``reason`` for "the run needs input from the user". A core
#: value rather than a framework-scoped one, so a generic AG-UI client knows it.
INPUT_REQUIRED = "input_required"

#: How many options a question may offer, by default. A host sets other limits
#: with ``MCP_AGENT_INTERRUPT_GATE_MIN_OPTIONS`` and
#: ``MCP_AGENT_INTERRUPT_GATE_MAX_OPTIONS``, or by building the tool with
#: :func:`make_interrupt_gate`.
MIN_OPTIONS = 2
MAX_OPTIONS = 10

#: The ``responseSchema`` property that holds the answer: one value, or a list
#: of values where the question allows several. Part of the wire format that
#: clients read and send, so it is named once here, not a setting.
CHOICE = "choice"

#: What the model reads when the user did not answer.
NOT_ANSWERED = "User did not answer the question."


def description(min_options: int = MIN_OPTIONS, max_options: int = MAX_OPTIONS) -> str:
    """The tool description the model reads, for the limits in force."""
    return (
        f"Asks the user to choose between {min_options} and {max_options} "
        "options, and waits for the answer. The run stops until the user "
        "chooses, and the choice is the result of this tool. It is for a "
        "choice only the user can make: an ambiguous request, several results "
        "to choose from, or a step that is costly or hard to undo. It is not "
        "for a choice the agent can make itself. One call asks one question. "
        "The client shows the question and the options, so the message that "
        "calls this tool must contain no text of its own."
    )


#: The system-prompt fragment for a model that has the tool. The description
#: alone is not enough: a model that has options to offer tends to write them
#: into its answer instead. Appended by :func:`~mcp_agent.main.build_agent` and
#: :func:`~mcp_agent.main.with_session_state` to their default prompts; a host
#: with its own prompt appends it the same way, and only where the tool is.
INTERRUPT_GATE_PROMPT = """\
Asking the user (required):

If your reply would ask the user to choose — which item, which of several \
results, which reading of the request — do not write that reply. Call the \
interrupt tool instead, with the options. The user answers the tool itself; a \
list of options with a question in plain text leaves the user to type the \
answer.

When you call interrupt, write no text at all in the same message. Put the \
question in the question argument and the choices in the options argument, \
never in your text. The tool shows them to the user; text beside the call \
shows them a second time.

Example: a search finds several matches and the request does not say which one \
to use. Call interrupt(question="Which one should I use?", \
options=[{"value": "<id>", "label": "<name>"}, ...]), with no text beside the \
call. Do not reply with a numbered list and "Which one?".\
"""


class Option(BaseModel):
    """One answer the user can choose."""

    value: str = Field(
        description="The value returned when the user chooses this option: an "
        "id, a code, or the answer itself."
    )
    label: str = Field(description="The text the user reads for this option.")


class InterruptGateInput(BaseModel):
    """The arguments the model writes. Types only; the rules are checked in the
    tool, so a broken call comes back as a sentence the model can act on rather
    than a schema error."""

    question: str = Field(description="The question, as the user reads it.")
    options: list[Option] = Field(
        description="The options, each with a distinct value."
    )
    multiple: bool = Field(
        default=False,
        description="Whether the user can choose more than one option.",
    )
    #: Filled by the graph, never by the model: the interrupt names the call it
    #: stopped on, so a client can put the question beside that call.
    tool_call_id: Annotated[str, InjectedToolCallId]


def argument_errors(
    question: str,
    options: Sequence[Option],
    min_options: int = MIN_OPTIONS,
    max_options: int = MAX_OPTIONS,
) -> list[str]:
    """Every rule the arguments break, as sentences. Empty when they are good."""
    errors = []
    if not question.strip():
        errors.append("question is empty")
    if not min_options <= len(options) <= max_options:
        errors.append(
            f"give {min_options} to {max_options} options, not {len(options)}"
        )
    values = [option.value for option in options]
    if len(set(values)) != len(values):
        errors.append("two options have the same value")
    if any(not option.value.strip() or not option.label.strip() for option in options):
        errors.append("every option needs a value and a label")
    return errors


def answer_model(options: Sequence[Option], multiple: bool = False) -> type[BaseModel]:
    """A model of the answer: :data:`CHOICE` is one of the option values.

    Each option is a ``Literal`` titled with its label, so the option list and
    the schema clients receive cannot drift apart.
    """
    # Built at runtime, over as many options as the question has, so the types
    # are `Any` to a type checker.
    one_of: Any = Union[  # noqa: UP007
        tuple(
            Annotated[Literal[option.value], Field(title=option.label)]
            for option in options
        )
    ]
    answer: Any = (
        Annotated[
            list[one_of], Field(min_length=1, json_schema_extra={"uniqueItems": True})
        ]
        if multiple
        else one_of
    )
    return create_model("Answer", **{CHOICE: (answer, ...)})  # type: ignore[call-overload]


def _titled(member: Mapping[str, Any]) -> dict[str, Any]:
    """One generated union member as a titled constant."""
    return {"const": member["const"], "title": member["title"]}


def response_schema(
    options: Sequence[Option], multiple: bool = False
) -> dict[str, Any]:
    """The JSON Schema an answer must satisfy, from :func:`answer_model`.

    MCP elicitation's enum shape (SEP-1330), under :data:`CHOICE` either way: a
    titled ``oneOf`` for one choice, an array of ``anyOf`` items for several.
    The titles are what a client puts on its options, so a client needs nothing
    but this schema to draw them. Pydantic writes a union as ``anyOf`` and
    names every model it generates, so the single choice is turned into
    ``oneOf`` and the generated names are left out.
    """
    answer = answer_model(options, multiple).model_json_schema()["properties"][CHOICE]
    answer.pop("title", None)
    if multiple:
        answer["items"]["anyOf"] = [_titled(item) for item in answer["items"]["anyOf"]]
    else:
        answer = {
            "type": "string",
            "oneOf": [_titled(item) for item in answer["anyOf"]],
        }
    return {
        "type": "object",
        "properties": {CHOICE: answer},
        "required": [CHOICE],
    }


def options_of(schema: Mapping[str, Any]) -> tuple[list[tuple[str, str]], bool]:
    """``([(value, label)], multiple)`` read back out of a response schema.

    For a client drawing the options, and for a terminal numbering them: the
    schema is the one thing every surface has, so reading it here keeps them
    from each parsing it their own way.
    """
    answer = (schema.get("properties") or {}).get(CHOICE) or {}
    multiple = answer.get("type") == "array"
    titled = (
        (answer.get("items") or {}).get("anyOf") if multiple else answer.get("oneOf")
    ) or []
    return [
        (str(item.get("const")), str(item.get("title", item.get("const"))))
        for item in titled
    ], multiple


def response_from_reply(reply: str, schema: Mapping[str, Any]) -> dict[str, Any] | None:
    """A typed reply to a numbered list of options, as a resume response.

    For a surface with no option controls — a terminal, a chat box. ``"2"``
    picks the second option, ``"1, 3"`` two of them where the question allows
    several, and an empty reply cancels. ``None`` for anything else, so the
    caller can ask again rather than send a guess.
    """
    options, multiple = options_of(schema)
    text = reply.strip()
    if not text:
        return {"status": CANCELLED}
    try:
        numbers = [int(part) for part in text.replace(",", " ").split()]
    except ValueError:
        return None
    if not numbers or not all(1 <= number <= len(options) for number in numbers):
        return None
    if not multiple and len(numbers) != 1:
        return None
    values = list(dict.fromkeys(options[number - 1][0] for number in numbers))
    return {
        "status": RESOLVED,
        "payload": {CHOICE: values if multiple else values[0]},
    }


def answer_text(response: Any, options: Sequence[Option]) -> str:
    """What the model reads for one resume response.

    ``response`` is what the resume carried for this interrupt: AG-UI's
    ``{"status": "resolved", "payload": {...}}`` or ``{"status": "cancelled"}``.
    Anything else — a payload that names no option, or no status at all — is
    reported as not answered rather than guessed at: the route validates
    payloads against the schema, so this only happens when a host of its own
    resumed with something else.
    """
    if not isinstance(response, Mapping) or response.get("status") != RESOLVED:
        return NOT_ANSWERED
    payload = response.get("payload")
    if not isinstance(payload, Mapping):
        return NOT_ANSWERED
    chosen = payload.get(CHOICE)
    picked: list[Any] = chosen if isinstance(chosen, list) else [chosen]
    labels = {option.value: option.label for option in options}
    if not picked or any(value not in labels for value in picked):
        return NOT_ANSWERED
    return "User chose: " + ", ".join(
        f"{labels[value]} (value: {value})" for value in picked
    )


def make_interrupt_gate(
    min_options: int = MIN_OPTIONS, max_options: int = MAX_OPTIONS
) -> BaseTool:
    """The ``interrupt`` tool, ready for ``extra_tools``.

    ``min_options`` and ``max_options`` are how many options one question may
    offer; the tool description tells the model the same numbers. A host that
    wants other limits sets them in the environment (see
    :class:`~mcp_agent.main.InterruptGateSettings`), or builds the tool here
    and passes it in ``extra_tools``, which the runtime then keeps rather than
    adding its own.

    It needs a checkpointer: ``interrupt()`` keeps the paused run there, and
    raises without one. :func:`~mcp_agent.main.with_session_state` and
    :func:`~mcp_agent.main.build_agent` add it only when there is one.
    """

    async def ask(
        question: str,
        options: list[Option],
        tool_call_id: Annotated[str, InjectedToolCallId],
        multiple: bool = False,
    ) -> str:
        # Coerced here because a tool called directly, rather than through the
        # graph, receives the dicts the model wrote.
        options = [Option.model_validate(option) for option in options]
        if errors := argument_errors(question, options, min_options, max_options):
            # Before interrupt(), not after: a question the client cannot draw
            # must never stop the run.
            return (
                f"{TOOL_NAME} was not sent: {'; '.join(errors)}. Fix it and call again."
            )
        response = interrupt(
            {
                "reason": INPUT_REQUIRED,
                "message": question,
                "toolCallId": tool_call_id,
                "responseSchema": response_schema(options, multiple),
            }
        )
        return answer_text(response, options)

    return StructuredTool.from_function(
        coroutine=ask,
        name=TOOL_NAME,
        description=description(min_options, max_options),
        args_schema=InterruptGateInput,
    )
