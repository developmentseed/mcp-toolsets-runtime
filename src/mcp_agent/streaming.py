"""One chat turn, delivered as it happens.

:func:`~mcp_agent.main.run_turn` is built on ``ainvoke``: everything it returns
is derived from a finished result, so a surface that wants to show a token
before the turn ends can reuse none of it. :func:`stream_turn` is the sibling
that reads ``agent.astream`` and yields each part as it arrives.

It lives here rather than in a host or a route handler because the parts are
awkward to recover and easy to get subtly wrong:

- **Tool results arrive on the token channel too.** ``stream_mode="messages"``
  carries every message the graph produces, tool ones included, so a loop that
  treats each chunk with content as answer text prints tool output into the
  answer. Tokens are ``AIMessageChunk`` from the model node and nothing else.
- **Receipts ride ``ToolMessage.artifact``**, not content, and reach the caller
  only on ``stream_mode="updates"``.
- **``tool_state`` appears on an update only when it changed, and names only
  what that node wrote** — not the merged state. Taking it straight off an
  update shows the newest key as though it were the only one, and the turn's
  final state is on no update at all. :class:`StateChanged` carries the
  running total; the final state is read back from the checkpointer.

Both stream modes come from one ``astream`` call. What a host does with the
events is its own business: :mod:`mcp_agent.host` renders them for Chainlit, and
an HTTP surface can map them onto its own wire format.
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, cast

from langchain_core.messages import AIMessageChunk, BaseMessage, HumanMessage

from mcp_agent.main import TurnResult, answer_citations
from mcp_state import CAPTURED_ARTIFACT_KEY, receipts_of
from mcp_state.receipts import Receipt
from mcp_state.state import TOOL_STATE_KEY, StateEntry

#: The graph node a token comes from. Anything else on the message channel is
#: some other node's output — a tool result, most often — and not answer text.
MODEL_NODE = "model"


@dataclass
class AnswerChunk:
    """A piece of the model's answer, as it was generated."""

    text: str


@dataclass
class ToolStarted:
    """The model asked for a tool, with the arguments it wrote.

    ``arguments`` is the parsed, complete argument dict rather than a stream of
    JSON fragments: it is emitted once the model node finishes, which is the
    first point the arguments are known to be whole. A parameter filled from
    state is absent here — that is the point of it — and shows up in the
    matching :class:`ToolFinished`'s ``received``.
    """

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolFinished:
    """A tool returned, with both directions of its session-state traffic.

    ``received`` maps a parameter to the receipt for the value state supplied,
    covering both paths: ``via="declaration"`` for a parameter the model never
    saw, ``via="handle"`` for one it pointed at with ``@state:<key>``.
    ``published`` maps a field of the tool's own return to the key it was stored
    under. ``artifact`` is the whole tool artifact, which a host needs to build
    view props.
    """

    id: str
    name: str
    content: str
    artifact: Any = None
    received: dict[str, Receipt] = field(default_factory=dict)
    published: dict[str, str] = field(default_factory=dict)


@dataclass
class StateChanged:
    """Session state after a tool wrote to it — the whole of it, accumulated.

    The update channel carries only what the node that ran wrote, not the merged
    state, so a consumer that took ``tool_state`` straight off an update would
    see the newest key and believe it was the only one. This carries the running
    total instead, which is what a surface rendering "what is in state" needs,
    and what makes an earlier value's shape available to describe a later tool's
    receipt.
    """

    state: dict[str, StateEntry]


@dataclass
class TurnFinished:
    """The turn is over. Carries exactly what :func:`run_turn` would have.

    Always the last event, so a caller wanting one answer can consume the
    stream and keep only this.
    """

    result: TurnResult


#: Anything :func:`stream_turn` yields.
TurnEvent = AnswerChunk | ToolStarted | ToolFinished | StateChanged | TurnFinished


def _published(artifact: Any) -> dict[str, str]:
    """``{field: state key}`` for what a tool's return put into state."""
    if not isinstance(artifact, dict):
        return {}
    captured = artifact.get(CAPTURED_ARTIFACT_KEY)
    return captured if isinstance(captured, dict) else {}


def _tool_calls(message: BaseMessage) -> list[ToolStarted]:
    return [
        ToolStarted(
            id=str(call.get("id") or ""),
            name=str(call.get("name") or ""),
            arguments=dict(call.get("args") or {}),
        )
        for call in getattr(message, "tool_calls", None) or []
    ]


def _tool_result(message: BaseMessage) -> ToolFinished:
    artifact = getattr(message, "artifact", None)
    return ToolFinished(
        id=str(getattr(message, "tool_call_id", "") or ""),
        name=str(getattr(message, "name", "") or ""),
        content=str(message.content),
        artifact=artifact,
        received=receipts_of(artifact),
        published=_published(artifact),
    )


def _is_token(mode: str, payload: Any) -> str | None:
    """The answer text in a message-channel payload, or ``None``.

    A tool result reaches this channel as well as the update channel, and it
    carries content — so the node and the chunk type both have to agree before
    a chunk counts as answer text.
    """
    if mode != "messages" or not isinstance(payload, tuple) or len(payload) != 2:
        return None
    chunk, metadata = payload
    if not isinstance(chunk, AIMessageChunk):
        return None
    if (metadata or {}).get("langgraph_node") != MODEL_NODE:
        return None
    text = str(getattr(chunk, "text", "") or "")
    return text or None


async def stream_turn(
    agent: Any,
    text: str,
    thread_id: str,
    config: dict[str, Any] | None = None,
) -> AsyncIterator[TurnEvent]:
    """Run one chat turn on ``thread_id``, yielding each part as it arrives.

    The same contract as :func:`~mcp_agent.main.run_turn` — only the new message
    is sent, and the thread's transcript and ``tool_state`` live in the
    checkpointer — but delivered incrementally, ending with a
    :class:`TurnFinished` carrying the identical :class:`TurnResult`.

    ``config`` is the runnable config, for a host attaching per-turn callbacks or
    metadata; ``thread_id`` is merged into its ``configurable`` and wins over any
    set there.
    """
    merged = dict(config or {})
    merged["configurable"] = {
        **(merged.get("configurable") or {}),
        "thread_id": thread_id,
    }

    # The thread is read before the turn only to know where this turn's messages
    # begin, exactly as run_turn does; it is cheap next to the model call.
    before = await agent.aget_state(cast(Any, merged))
    seen = len((getattr(before, "values", None) or {}).get("messages") or [])

    last_ai: BaseMessage | None = None
    running: dict[str, StateEntry] = {}
    async for mode, payload in agent.astream(
        cast(Any, {"messages": [HumanMessage(text)]}),
        cast(Any, merged),
        stream_mode=["updates", "messages"],
    ):
        if (token := _is_token(mode, payload)) is not None:
            yield AnswerChunk(token)
            continue
        if mode != "updates" or not isinstance(payload, dict):
            continue
        for update in payload.values():
            if not isinstance(update, dict):
                continue
            for message in update.get("messages") or []:
                if getattr(message, "type", None) == "tool":
                    yield _tool_result(message)
                    continue
                for started in _tool_calls(message):
                    yield started
                if getattr(message, "type", None) == "ai":
                    last_ai = message
            # An update carries only what its node wrote, so this is a running
            # total rather than a replacement — two tools publishing different
            # keys arrive as two updates naming one key each.
            if isinstance(written := update.get(TOOL_STATE_KEY), dict):
                running.update(written)
                yield StateChanged(dict(running))

    # Read the thread back rather than accumulating: `tool_state` reaches the
    # update channel only on the turns that changed it, so the final state is
    # not recoverable from the last update alone.
    after = await agent.aget_state(cast(Any, merged))
    values = getattr(after, "values", None) or {}
    history: list[BaseMessage] = values.get("messages") or []
    sidecar: dict[str, StateEntry] | None = values.get(TOOL_STATE_KEY)
    yield TurnFinished(
        TurnResult(
            history=history,
            # +1 skips the HumanMessage just added: "new" means the agent's
            # replies.
            new_messages=history[seen + 1 :],
            # ``.text`` flattens list-structured content, which ``str(.content)``
            # would render as a Python repr.
            answer=str(last_ai.text) if last_ai is not None else "",
            sidecar=sidecar,
            citations=answer_citations(last_ai) if last_ai is not None else [],
        )
    )
