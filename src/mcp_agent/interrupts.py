"""A turn that stops for a person, and the turn that picks it up again.

A tool that calls LangGraph's ``interrupt()`` ends the run with the thread
paused on that tool call. What is paused is the *graph*, not any one host, so
the rules for going on live here, where :func:`~mcp_agent.main.run_turn` and
:func:`~mcp_agent.streaming.stream_turn` both use them:

- **An answer is a resume, not a message.** ``Command(resume={id: response})``
  runs the paused tool call again and ``interrupt()`` returns the response, so
  the answer is that call's result.
- **Every open interrupt is answered at once.** AG-UI does not support a
  partial resume, and neither does this: an answer set that leaves one out, or
  names one that is not open, is refused with :class:`ResumeMismatch`.
- **A paused thread takes an answer, not a message.** Sending input to a paused
  thread makes LangGraph drop the interrupt silently and leave the tool call
  with no result, which a provider then refuses on the next model call. So a
  message on a paused thread is refused with :class:`ResumeMismatch`; a person
  who wants to move on cancels the question first.

A tool call that finished beside the one that paused is **not run again** on
resume: its result is a pending write, and the update carrying it is replayed
marked ``__metadata__: {"cached": True}`` — see :func:`is_replayed`.

**An MCP tool can ask too.** A toolset tool that needs input mid-call returns
an elicitation, and ``langchain.mcp`` pauses the run on it with a value of its
own shape. :func:`question` puts that value in the ``interrupt`` tool's terms,
so every surface asks it the same way, and :func:`answer` turns the response
back into the shape the paused call waits for.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from langchain.mcp.elicitation import ELICITATION_INTERRUPT_TYPE
from langgraph.types import Command

#: AG-UI's core ``reason`` for "the run needs input from the user". A core
#: value rather than a framework-scoped one, so a generic AG-UI client knows it.
INPUT_REQUIRED = "input_required"

#: The status values a resume response carries, as AG-UI's ``ResumeEntry``.
RESOLVED = "resolved"
CANCELLED = "cancelled"


@dataclass
class PendingInterrupt:
    """One open question: LangGraph's interrupt id and what the tool raised.

    ``value`` is the :func:`question` the tool asks: what the ``interrupt``
    tool passed to ``interrupt()``, an MCP tool's elicitation in that same
    shape, or anything else as it is, for a host to make what it will of it.
    """

    id: str
    value: Any


class ResumeMismatch(ValueError):
    """A turn that does not fit what the thread is waiting for.

    A resume that does not answer exactly the open interrupts, or a message
    sent while some are open. Either way the thread has moved on from what the
    caller believed, and reading it again is the fix.
    """


def _elicitation(value: Any) -> list[Mapping[str, Any]] | None:
    """The requests of an MCP elicitation interrupt, or ``None`` for another value."""
    if isinstance(value, Mapping) and value.get("type") == ELICITATION_INTERRUPT_TYPE:
        return list(value["requests"])
    return None


def question(value: Any) -> Any:
    """An interrupt's value as the question a client puts to the user.

    The ``interrupt`` tool's value already is one, as is anything not
    recognised. An MCP elicitation becomes the same shape: its message, its
    requested schema as ``responseSchema``, and the tool that asked under
    ``metadata``. A round asking several things at once nests each request's
    schema under its key. A ``url`` request has no schema; its address goes in
    the message.

    It carries no ``toolCallId``: the value names the tool, not the call.
    """
    if (requests := _elicitation(value)) is None:
        return value
    lines = [
        f"{request['message']} {request['url']}"
        if request["mode"] == "url"
        else request["message"]
        for request in requests
    ]
    forms = [request for request in requests if request["mode"] == "form"]
    if not forms:
        schema = None
    elif len(requests) == 1:
        schema = forms[0]["requested_schema"]
    else:
        schema = {
            "type": "object",
            "properties": {r["key"]: r["requested_schema"] for r in forms},
            "required": [r["key"] for r in forms],
        }
    asked = {
        "reason": INPUT_REQUIRED,
        "message": "\n".join(lines),
        "metadata": {"tool": value["tool_name"]},
    }
    return {**asked, "responseSchema": schema} if schema else asked


def answer(value: Any, response: Mapping[str, Any]) -> Any:
    """What a paused call resumes with, given the user's response.

    The ``interrupt`` tool reads the response as it is. An MCP elicitation
    waits for ``{"responses": {key: {action, content}}}``. A resolved response
    accepts each request with its part of the payload; accepting a ``url``
    request means the user agreed to open it. A cancelled response, which is
    the user skipping the question, declines them all: MCP's ``cancel`` is a
    question dismissed without a choice, and skipping is a choice. A form
    request the payload has no part for is declined too, since the server
    rejects an accept with no content.
    """
    if (requests := _elicitation(value)) is None:
        return response
    declined = {"action": "decline"}
    if response.get("status") != RESOLVED:
        return {"responses": {r["key"]: declined for r in requests}}
    payload = response.get("payload") or {}

    def accepted(request: Mapping[str, Any]) -> dict[str, Any]:
        if request["mode"] != "form":
            return {"action": "accept"}
        content = payload if len(requests) == 1 else payload.get(request["key"])
        return declined if content is None else {"action": "accept", "content": content}

    return {"responses": {r["key"]: accepted(r) for r in requests}}


def pending(interrupts: Iterable[Any] | None) -> list[PendingInterrupt]:
    """LangGraph ``Interrupt`` objects as :class:`PendingInterrupt`, in order.

    Takes what either place hands back: a state snapshot's ``interrupts`` and
    the ``__interrupt__`` key of an ``ainvoke`` result are the same objects.
    Each value is the :func:`question` it asks.
    """
    return [
        PendingInterrupt(id=str(item.id), value=question(item.value))
        for item in interrupts or ()
    ]


def check_resume(
    open_interrupts: Sequence[PendingInterrupt], resume: Mapping[str, Any]
) -> None:
    """Refuse a resume that is not one response per open interrupt.

    Both directions are refused and named: an id that is not open is a client
    answering a question that is gone (already answered, or from another
    thread), and an open id with no response is a partial resume.
    """
    if not open_interrupts:
        raise ResumeMismatch("the thread has no open interrupt to resume")
    open_ids = {item.id for item in open_interrupts}
    if unknown := sorted(set(resume) - open_ids):
        raise ResumeMismatch(f"not an open interrupt: {', '.join(unknown)}")
    if missing := sorted(open_ids - set(resume)):
        raise ResumeMismatch(
            f"every open interrupt needs a response; missing: {', '.join(missing)}"
        )


def turn_input(
    snapshot: Any, message: Any | None, resume: Mapping[str, Any] | None
) -> Any:
    """What a turn sends to the graph, given the thread as it stands.

    ``message`` is the turn's new ``HumanMessage``; ``resume`` maps an
    interrupt id to its response. A turn carries one or the other:

    - a message, on a thread that is not paused, is the message;
    - a resume that answers every open interrupt is ``Command(resume=...)``.

    Raises :class:`ResumeMismatch` for a message on a paused thread and for a
    resume that does not match the open interrupts, and ``ValueError`` for a
    turn with both or neither.
    """
    if message is not None and resume is not None:
        raise ValueError("a turn carries a message or a resume, not both")
    raw = list(getattr(snapshot, "interrupts", None) or ())
    open_interrupts = pending(raw)
    if resume is not None:
        check_resume(open_interrupts, resume)
        values = {str(item.id): item.value for item in raw}
        return Command(
            resume={id: answer(values[id], response) for id, response in resume.items()}
        )
    if open_interrupts:
        raise ResumeMismatch(
            "the thread is waiting for an answer to "
            + ", ".join(item.id for item in open_interrupts)
            + "; resume it, or cancel, before sending a message"
        )
    if message is None:
        raise ValueError("a turn needs a message or a resume")
    return {"messages": [message]}


def is_replayed(update: Mapping[str, Any]) -> bool:
    """Whether a ``stream_mode="updates"`` payload replays a cached write.

    On resume, a task that finished before the pause is not run again, but its
    update is sent again with this marker. Reading it twice would announce the
    same tool result twice.
    """
    metadata = update.get("__metadata__")
    return isinstance(metadata, Mapping) and bool(metadata.get("cached"))
