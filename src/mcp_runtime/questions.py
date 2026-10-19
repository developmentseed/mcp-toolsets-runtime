"""A toolset tool asks the user a question mid-call.

MCP lets a tool stop and ask: it returns an ``InputRequiredResult`` naming
what it needs, and the client calls the tool again with the answers. A toolset
tool is a LangChain tool whose return annotation is its output schema, so it
asks by raising :class:`AskUser` instead, and the bridge turns that into the
result the client expects.

The answers come back on the retry through a context variable the bridge fills
from the request's ``input_responses``, read with :func:`answers`. It is a
context variable for the reason the credentials are: the SDK hands its
per-request context to the handler, and a tool body several frames down has no
way to reach an argument.
"""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, cast

from mcp.types import (
    ElicitRequest,
    ElicitRequestFormParams,
    ElicitResult,
    InputRequiredResult,
)

#: The user's answers to the question this call asked last time, or ``None``
#: when it has not asked yet.
_answers: ContextVar[Mapping[str, ElicitResult] | None] = ContextVar(
    "mcp_runtime_answers", default=None
)


class AskUser(Exception):
    """Raise from a tool to stop the call and ask the user.

    Each keyword is one request: its key, and ``(message, schema)``, where the
    schema is the JSON schema of the answer. The call ends here; the client
    puts the question to the user and calls the tool again, from the start,
    with :func:`answers` filled in. So do the asking before any work that
    should happen once.
    """

    def __init__(self, **requests: tuple[str, dict[str, Any]]) -> None:
        super().__init__(", ".join(message for message, _ in requests.values()))
        self.requests = requests

    def result(self) -> InputRequiredResult:
        """The question as the MCP result the bridge returns for it."""
        return InputRequiredResult(
            input_requests={
                key: ElicitRequest(
                    params=ElicitRequestFormParams(
                        message=message, requested_schema=schema
                    )
                )
                for key, (message, schema) in self.requests.items()
            }
        )


def answers() -> Mapping[str, ElicitResult] | None:
    """The user's answers, by request key, or ``None`` on a call that has not asked.

    Each answer's ``action`` is ``accept`` with the ``content`` the schema
    asked for, or ``decline`` with none. A declined request is still an
    answer: the call has asked, and asking again would ask forever.
    """
    return _answers.get()


@contextmanager
def answers_context(responses: Mapping[str, Any] | None) -> Iterator[None]:
    """Run as if the user had answered ``responses`` (the bridge, and tests).

    Lets toolset tests exercise a tool's second round without a server:

        with answers_context({"year": ElicitResult(action="accept", content={...})}):
            my_tool.invoke({...})
    """
    token = _answers.set(cast(Mapping[str, ElicitResult], responses))
    try:
        yield
    finally:
        _answers.reset(token)
