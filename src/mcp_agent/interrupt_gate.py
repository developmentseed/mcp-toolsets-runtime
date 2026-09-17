"""``interrupt``: the agent asks the person a question, and the run stops.

The tool is called ``interrupt``, a verb, because that is what it does to the
run. This module is ``interrupt_gate``: the run stops at the gate and goes on
only when the person answers. The agent calls the tool with a question and its
options. The tool raises a LangGraph ``interrupt()``, so the run ends there
with nothing more for the model to do; the person's answer comes back on the
next run as the **result of that same tool call**.

That is the whole design, and each part of it is there because the other way
round went wrong elsewhere. A choice sent back as a new user message is plain
text with no link to what it answers, so the model has to work out which
question it was; a run that carries on after asking lets the model call more
tools or answer for the person; and options kept in shared state can be
overwritten by the next tool that sets some. Here the run is over the moment
the question is asked, the answer is addressed to the call that asked it, and
one call holds one question.

The interrupt value is shaped for AG-UI (``reason``, ``message``,
``toolCallId``, ``responseSchema``) so :mod:`mcp_agent_api.events` passes it on
as-is, and the schema uses MCP elicitation's titled-enum shape so the same
schema would serve an MCP client too.

This module is only the tool. Stopping a turn, answering it, and refusing a
message while a question is open are LangGraph-general and live in
:mod:`mcp_agent.interrupts`.
"""

from collections.abc import Mapping, Sequence
from typing import Annotated, Any

from langchain_core.tools import BaseTool, InjectedToolCallId, StructuredTool
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from mcp_agent.interrupts import CANCELLED, RESOLVED

#: The tool's name, as the model and every client see it.
TOOL_NAME = "interrupt"

#: AG-UI's core ``reason`` for "the run needs input from a person". A core value
#: rather than a framework-scoped one, so a generic AG-UI client knows it.
INPUT_REQUIRED = "input_required"

#: How many options a question may offer, by default. Two, because one option
#: is not a choice. The upper bound only keeps a row of buttons readable; a
#: host with other needs passes ``max_options`` to :func:`make_interrupt_gate`.
MIN_OPTIONS = 2
MAX_OPTIONS = 10

#: The ``responseSchema`` property that holds the answer: one value, or a list
#: of values where the question allows several. Part of the wire format that
#: clients read and send, so it is named once here, not a setting.
CHOICE = "choice"

#: What the model reads when the person did not answer.
NOT_ANSWERED = "User did not answer the question."


def description(max_options: int = MAX_OPTIONS) -> str:
    """The tool description the model reads, for the limits in force.

    Third person — what the tool does and what it is for — rather than orders
    to the model; the orders are in :data:`INTERRUPT_GATE_PROMPT`.
    """
    return (
        f"Asks the user to choose between {MIN_OPTIONS} and {max_options} "
        "options, and waits for the answer. The user sees one button for each "
        "option. The run stops until the user chooses, and the choice is the "
        "result of this tool. It is for a point where the reply would otherwise "
        "end with a list of options and a question: an ambiguous request with "
        "several reasonable readings, several results to choose from, or a step "
        "that is costly or hard to undo. It is not for a choice the agent can "
        "make itself, a free and reversible step, or a choice that a tool's own "
        "view already shows. One call asks one question. The tool itself shows "
        "the question and the options to the user. For that reason, the message "
        "that calls this tool must contain no text: a question or a list of "
        "options written beside the call is shown to the user a second time."
    )


#: The system-prompt fragment for a model that has the tool. The description
#: alone was not enough in practice: a model that found three datasets listed
#: them in its answer and asked which one, which the user can only answer by
#: typing. Appended by :func:`~mcp_agent.main.build_agent` and
#: :func:`~mcp_agent.main.with_session_state` to their default prompts; a host
#: with its own prompt appends it the same way, and only where the tool is.
INTERRUPT_GATE_PROMPT = """\
Asking the user (required):

If your reply would ask the user to choose — which item, which of several \
results, which reading of the request — do not write that reply. Call the \
interrupt tool instead, with the options. The user answers by clicking a \
button; a list of options with a question in plain text cannot be answered \
that way.

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
    """One answer the person can pick."""

    value: str = Field(
        description="The value returned when the user chooses this option: an "
        "id, a code, or the answer itself."
    )
    label: str = Field(description="The text on the button.")


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
    question: str, options: Sequence[Option], max_options: int = MAX_OPTIONS
) -> list[str]:
    """Every rule the arguments break, as sentences. Empty when they are good."""
    errors = []
    if not question.strip():
        errors.append("question is empty")
    if not MIN_OPTIONS <= len(options) <= max_options:
        errors.append(
            f"give {MIN_OPTIONS} to {max_options} options, not {len(options)}"
        )
    values = [option.value for option in options]
    if len(set(values)) != len(values):
        errors.append("two options have the same value")
    if any(not option.value.strip() or not option.label.strip() for option in options):
        errors.append("every option needs a value and a label")
    return errors


def response_schema(
    options: Sequence[Option], multiple: bool = False
) -> dict[str, Any]:
    """The JSON Schema an answer must satisfy, built from the options.

    MCP elicitation's enum shape (SEP-1330), under :data:`CHOICE` either way: a
    titled ``oneOf`` for one choice, an array of ``anyOf`` items for several.
    The titles are what a client puts on its buttons, so a client needs nothing
    but this schema to draw them.
    """
    titled = [{"const": option.value, "title": option.label} for option in options]
    answer: dict[str, Any] = (
        {
            "type": "array",
            "items": {"anyOf": titled},
            "minItems": 1,
            "uniqueItems": True,
        }
        if multiple
        else {"type": "string", "oneOf": titled}
    )
    return {
        "type": "object",
        "properties": {CHOICE: answer},
        "required": [CHOICE],
    }


def options_of(schema: Mapping[str, Any]) -> tuple[list[tuple[str, str]], bool]:
    """``([(value, label)], multiple)`` read back out of a response schema.

    For a client drawing the buttons, and for a terminal numbering them: the
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

    For a surface with no buttons — a terminal, a chat box. ``"2"`` picks the
    second option, ``"1, 3"`` two of them where the question allows several,
    and an empty reply cancels. ``None`` for anything else, so the caller can
    ask again rather than send a guess.
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


def make_interrupt_gate(max_options: int = MAX_OPTIONS) -> BaseTool:
    """The ``interrupt`` tool, ready for ``extra_tools``.

    ``max_options`` is the most options one question may offer; the tool
    description tells the model the same number. A host that wants another
    limit builds the tool here and passes it in ``extra_tools``, and the runtime
    keeps it rather than adding its own.

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
        if errors := argument_errors(question, options, max_options):
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
        description=description(max_options),
        args_schema=InterruptGateInput,
    )
