"""A toolset tool asks the user a question mid-call.

MCP lets a tool stop and ask: it returns an ``InputRequiredResult`` naming
what it needs, and the client calls the tool again with the answer. A toolset
tool does both with one call, :func:`ask`, the way the agent's own tools call
LangGraph's ``interrupt()``: the first time through it ends the call with the
question, and on the retry it returns the answer.

The answer reaches the retry through a context variable the bridge fills from
the request's ``input_responses``. It is a context variable for the reason the
credentials are: the SDK hands its per-request context to the handler, and a
tool body several frames down has no way to reach an argument.
"""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, TypeVar, overload

from mcp.types import (
    ElicitRequest,
    ElicitRequestFormParams,
    ElicitResult,
    InputRequiredResult,
)
from pydantic import BaseModel

Answer = TypeVar("Answer", bound=BaseModel)

#: The one request a call makes, as the SDK keys it.
_KEY = "answer"

#: The user's answer to the question this call asked last time, or ``None``
#: when it has not asked yet.
_answer: ContextVar[ElicitResult | None] = ContextVar(
    "mcp_runtime_answer", default=None
)


class Asked(Exception):
    """A call ended by :func:`ask`: the question, for the bridge to return.

    A test sees it with ``pytest.raises(Asked)``; a tool never catches it.
    """

    def __init__(self, message: str, schema: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.message = message
        self.schema = dict(schema)

    def result(self) -> InputRequiredResult:
        return InputRequiredResult(
            input_requests={
                _KEY: ElicitRequest(
                    params=ElicitRequestFormParams(
                        message=self.message, requested_schema=self.schema
                    )
                )
            }
        )


@overload
def ask(message: str, answer: type[Answer]) -> Answer | None: ...
@overload
def ask(message: str, answer: Mapping[str, Any]) -> dict[str, Any] | None: ...
def ask(message: str, answer: Any) -> Any:
    """Put a question to the user; the answer, or ``None`` if they declined.

    ``answer`` says what to ask for: a pydantic model, whose fields are the
    questions and whose instance is the answer, or the same as a JSON schema,
    answered as a dict. Fields are strings, numbers, booleans and enums, which
    is what an MCP client can draw; ``Field`` constraints and descriptions
    reach the client too.

    The first time through, the call ends here and the client puts the
    question to the user. The client then calls the tool again from the start,
    and this time ``ask`` returns what they gave. So ask before any work that
    should happen once, and ask once per call: several things go in one model,
    not in a sequence of asks.
    """
    model: type[BaseModel] | None = (
        answer if isinstance(answer, type) and issubclass(answer, BaseModel) else None
    )
    reply = _answer.get()
    if reply is None:
        raise Asked(message, model.model_json_schema() if model else answer)
    if reply.action != "accept":
        return None
    content = dict(reply.content or {})
    return model.model_validate(content) if model else content


@contextmanager
def answered(content: Mapping[str, Any] | BaseModel | None) -> Iterator[None]:
    """Run as if the user had answered ``content``, or declined with ``None``.

    The bridge uses it for the retry; a toolset test uses it to reach the
    second half of a tool without a server:

        with answered({"year": 2024}):
            my_tool.invoke({...})
    """
    if isinstance(content, BaseModel):
        content = content.model_dump(mode="json")
    reply = (
        ElicitResult(action="decline")
        if content is None
        else ElicitResult(action="accept", content=dict(content))
    )
    token = _answer.set(reply)
    try:
        yield
    finally:
        _answer.reset(token)


@contextmanager
def replies(responses: Mapping[str, Any] | None) -> Iterator[None]:
    """Run with the SDK's ``input_responses`` as the answer (the bridge)."""
    reply = responses.get(_KEY) if responses else None
    token = _answer.set(reply if isinstance(reply, ElicitResult) else None)
    try:
        yield
    finally:
        _answer.reset(token)
