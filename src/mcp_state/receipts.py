"""What a tool was handed from session state, and where it came from.

The other direction from capture. :mod:`mcp_state.middleware` tells the model
what a tool *wrote* to ``tool_state``; this records what a tool *read* out of
it, so a value can be traced from the tool that published it to the tool that
consumed it without leaving the transcript.

A value reaches a tool because the model wrote ``@state:<key>`` as an argument
and :mod:`mcp_state.handles` substituted it on the way out. The key is
therefore already in the transcript — but the transcript says nothing about
what was behind it, and by the time a host renders the call the argument is
just a string. A receipt adds the two facts a reader needs: the entry that key
held, and the tool that published it.

Receipts ride on the tool message's artifact under
:data:`INJECTED_ARTIFACT_KEY`, beside the ``captured_state`` map capture leaves
there. Nothing echoes them back to the model: it wrote the handle itself, so
repeating it would spend tokens on something it already knows.
"""

from typing import Any, NotRequired, TypedDict, cast

from mcp_state.state import StateEntry

#: Artifact key under which a tool message records ``{parameter: Receipt}`` for
#: every parameter session state supplied. Read it with :func:`receipts_of`
#: rather than by hand.
INJECTED_ARTIFACT_KEY = "injected_state"


class Receipt(TypedDict):
    """Where one substituted parameter's value came from."""

    #: The ``tool_state`` key the value was read from.
    key: str
    #: The tool that published the value, as recorded on its state entry.
    tool: NotRequired[str | None]


def receipt_for(key: str, entry: StateEntry) -> Receipt:
    """A receipt for the entry stored under ``key``."""
    return Receipt(key=key, tool=entry.get("tool"))


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


def describe_receipt(parameter: str, receipt: Receipt) -> str:
    """One receipt as ``aoi ← <key>, published by <tool>``."""
    origin = f"{parameter} ← {receipt['key']}"
    if tool := receipt.get("tool"):
        return f"{origin}, published by {tool}"
    return origin
