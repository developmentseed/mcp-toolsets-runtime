"""What a tool was handed from session state, and where it came from.

The other direction from capture. :mod:`mcp_state.middleware` tells the model
what a tool *wrote* to ``tool_state``; this records what a tool *read* out of
it, so a value can be traced from the tool that published it to the tool that
consumed it without leaving the transcript.

A declared parameter is removed from the schema the model sees and filled at
call time (:mod:`mcp_state.injection`), which means nothing else records that
it was supplied at all: the tool call carries no such argument, and the return
says nothing about it. Two things follow from that, and both are what a receipt
is for.

Where several stored values share a kind, resolution takes the most recent.
Without a receipt the model cannot tell which one it was given, so it can
neither correct a wrong pick nor describe the result accurately. And a host
rendering the call — ``mcp_agent.web``'s tool steps, say — shows arguments that
are missing the one value that decided the output.

Receipts ride on the tool message's artifact under
:data:`INJECTED_ARTIFACT_KEY`, beside the ``captured_state`` map capture leaves
there, and :class:`~mcp_state.middleware.StateCaptureMiddleware` turns the
declared ones into a ``[state used: …]`` line next to ``[state updated: …]``.

Handle-supplied parameters (:mod:`mcp_state.handles`) are recorded but not
turned into a line: the model wrote ``@state:<key>`` itself, so the key is
already in the tool call arguments.
"""

from collections.abc import Mapping
from typing import Any, NotRequired, TypedDict, cast

from mcp_state.state import StateEntry

#: Artifact key under which a tool message records ``{parameter: Receipt}`` for
#: every parameter session state supplied. Read it with :func:`receipts_of`
#: rather than by hand.
INJECTED_ARTIFACT_KEY = "injected_state"

#: The client filled this parameter by matching the kind its server declared.
#: The model never saw the parameter.
BY_DECLARATION = "declaration"

#: The model pointed this parameter at a stored value with ``@state:<key>``.
BY_HANDLE = "handle"


class Receipt(TypedDict):
    """Where one filled parameter's value came from."""

    #: The ``tool_state`` key the value was read from.
    key: str
    #: :data:`BY_DECLARATION` or :data:`BY_HANDLE`.
    via: str
    kind: NotRequired[str | None]
    #: The tool that published the value, as recorded on its state entry.
    tool: NotRequired[str | None]


def receipt_for(key: str, entry: StateEntry, via: str) -> Receipt:
    """A receipt for the entry stored under ``key``."""
    return Receipt(key=key, via=via, kind=entry.get("kind"), tool=entry.get("tool"))


def receipts_of(artifact: Any) -> dict[str, Receipt]:
    """Every receipt on a tool message artifact, keyed by parameter.

    Empty for a message that received nothing from state, and for anything
    that is not an artifact — so a caller can read it off every tool result
    without branching.
    """
    if not isinstance(artifact, dict):
        return {}
    found = artifact.get(INJECTED_ARTIFACT_KEY)
    if not isinstance(found, dict):
        return {}
    return {
        parameter: cast("Receipt", receipt)
        for parameter, receipt in found.items()
        if isinstance(receipt, dict) and receipt.get("key")
    }


def supplied(
    receipts: Mapping[str, Receipt], arguments: Mapping[str, Any]
) -> dict[str, Receipt]:
    """The receipts for parameters ``arguments`` does not already account for.

    What a host showing a tool call needs: a declared parameter is absent from
    the arguments the model produced, so the call reads as though it ran
    without it. A handle is already there as the ``@state:<key>`` string the
    model wrote, and needs no second telling.
    """
    return {
        parameter: receipt
        for parameter, receipt in sorted(receipts.items())
        if parameter not in arguments
    }


def describe_receipt(parameter: str, receipt: Receipt) -> str:
    """One receipt as ``aoi ← dataset-search/geometry, published by search``."""
    origin = f"{parameter} ← {receipt['key']}"
    if tool := receipt.get("tool"):
        return f"{origin}, published by {tool}"
    return origin


def breadcrumb(receipts: Mapping[str, Receipt]) -> str | None:
    """The ``[state used: …]`` note, or ``None`` when there is nothing to say.

    Only declared fills are named. A handle is already in the tool call
    arguments the model wrote, so repeating it here would spend tokens on
    something the transcript records anyway.
    """
    declared = [
        describe_receipt(parameter, receipt)
        for parameter, receipt in sorted(receipts.items())
        if receipt.get("via") == BY_DECLARATION
    ]
    if not declared:
        return None
    return f"[state used: {'; '.join(declared)}]"
