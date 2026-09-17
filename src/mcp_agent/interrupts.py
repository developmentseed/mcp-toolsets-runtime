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
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from langgraph.types import Command

#: The status values a resume response carries, as AG-UI's ``ResumeEntry``.
RESOLVED = "resolved"
CANCELLED = "cancelled"


@dataclass
class PendingInterrupt:
    """One open question: LangGraph's interrupt id and what the tool raised.

    ``value`` is whatever the tool passed to ``interrupt()``. For ``ask_user``
    that is already shaped for AG-UI; another tool's value is passed on as it
    is, and a host decides what to make of it.
    """

    id: str
    value: Any


class ResumeMismatch(ValueError):
    """A turn that does not fit what the thread is waiting for.

    A resume that does not answer exactly the open interrupts, or a message
    sent while some are open. Either way the thread has moved on from what the
    caller believed, and reading it again is the fix.
    """


def pending(interrupts: Iterable[Any] | None) -> list[PendingInterrupt]:
    """LangGraph ``Interrupt`` objects as :class:`PendingInterrupt`, in order.

    Takes what either place hands back: a state snapshot's ``interrupts`` and
    the ``__interrupt__`` key of an ``ainvoke`` result are the same objects.
    """
    return [
        PendingInterrupt(id=str(item.id), value=item.value) for item in interrupts or ()
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
    open_interrupts = pending(getattr(snapshot, "interrupts", None))
    if resume is not None:
        check_resume(open_interrupts, resume)
        return Command(resume=dict(resume))
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
