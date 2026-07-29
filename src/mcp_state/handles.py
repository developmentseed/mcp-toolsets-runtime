"""Pass session state to tools that declare nothing, by reference.

The general path. :mod:`mcp_state.injection` needs a server-side ``Kind`` tag
to know which parameter to fill; this needs nothing at all, and so works
against any MCP server — including ones that will never adopt anything of
ours.

The trick is to stop trying to work out *which* parameter wants state. That
question cannot be answered from an unmodified tool: the JSON Schema of a
parameter holding a large object is almost always ``{"type": "object"}``,
which matches every object ever. So instead of deciding for the model, give it
a way to say so:

1. Every parameter that could hold a bulk value — schema type ``object`` or
   ``array`` — gains a second accepted form: the string ``@state:<key>``.
2. The model learns which keys exist from the ``[state updated: …]``
   breadcrumb the capture middleware already writes.
3. :func:`dereference` swaps a handle for the stored value on the way to the
   server, so the tool receives the real thing and never knows.

The model spends about ten tokens naming a key instead of thousands
reproducing a value, and the payload still never enters the transcript. What
it does not buy is the guarantee a declaration gives: the parameter is still
in the model's schema, so a model determined to inline a geometry can. That is
the price of requiring nothing of the server.

Choosing *which* stored value is the model's job here, and it is better placed
than any heuristic — it has the conversation, and "the area the user just drew"
is not something recency can be relied on to know.
"""

from typing import Any

from mcp_state.detect import describe
from mcp_state.state import StateEntry

#: Prefix marking an argument as a reference into ``tool_state``.
HANDLE_PREFIX = "@state:"

#: JSON Schema types whose values are big enough to be worth passing by
#: reference. A string or a number is cheaper to generate than to name.
BULK_TYPES = frozenset({"object", "array"})


def handle_for(key: str) -> str:
    """The handle a model writes to pass the value stored under ``key``."""
    return f"{HANDLE_PREFIX}{key}"


def is_handle(value: Any) -> bool:
    """Whether an argument is a state reference rather than a literal value."""
    return isinstance(value, str) and value.startswith(HANDLE_PREFIX)


def handle_key(value: str) -> str:
    """The ``tool_state`` key a handle refers to."""
    return value[len(HANDLE_PREFIX) :]


def _is_bulk(schema: Any) -> bool:
    """Whether a parameter's schema could hold a value worth passing by name.

    Accepts both ``"type": "object"`` and the list form JSON Schema allows.
    A parameter with no stated type is treated as bulk: unconstrained means it
    could be anything, including something large.
    """
    if not isinstance(schema, dict):
        return False
    if any(key in schema for key in ("$ref", "anyOf", "oneOf", "allOf")):
        return True
    declared = schema.get("type")
    if declared is None:
        return True
    if isinstance(declared, list):
        return bool(BULK_TYPES.intersection(declared))
    return declared in BULK_TYPES


def _with_handle_branch(schema: dict[str, Any]) -> dict[str, Any]:
    """One parameter's schema, also accepting a handle string."""
    handle_branch = {
        "type": "string",
        "pattern": f"^{HANDLE_PREFIX}",
        "description": (
            "A session-state reference, e.g. "
            f"{handle_for('dataset-search/geometry')} — the key from a "
            "[state updated: …] note. The value is substituted before the "
            "tool runs, so prefer this over repeating a large value."
        ),
    }
    original = {key: value for key, value in schema.items() if key != "description"}
    branched: dict[str, Any] = {"anyOf": [original, handle_branch]}
    if description := schema.get("description"):
        branched["description"] = description
    return branched


def offer_handles(args_schema: Any, skip: frozenset[str] = frozenset()) -> Any:
    """The schema with every bulk parameter also accepting ``@state:<key>``.

    Pure and shallow: only the entries under ``properties`` change, so
    ``$defs``, ``required`` and everything else survive byte-for-byte.
    ``skip`` names parameters an injected declaration already handles — those
    are about to be removed from the schema entirely, so offering a handle for
    them would only confuse the model.
    """
    if not isinstance(args_schema, dict):
        return args_schema
    properties = args_schema.get("properties")
    if not isinstance(properties, dict):
        return args_schema
    updated = {
        name: (
            _with_handle_branch(schema)
            if name not in skip and _is_bulk(schema)
            else schema
        )
        for name, schema in properties.items()
    }
    if updated == properties:
        return args_schema
    return {**args_schema, "properties": updated}


def dereference(
    arguments: dict[str, Any], tool_state: dict[str, StateEntry] | None
) -> dict[str, Any]:
    """Replace every ``@state:<key>`` argument with the value it names.

    A handle naming a key that does not exist is left as the literal string.
    The tool then rejects it against its own schema, which reports a real
    error to the model — better than silently passing ``None`` and having the
    tool fail somewhere less legible.
    """
    state = tool_state or {}
    resolved = {}
    for name, value in arguments.items():
        if is_handle(value):
            entry = state.get(handle_key(value))
            if entry is not None:
                resolved[name] = entry.get("value")
                continue
        resolved[name] = value
    return resolved


def available(tool_state: dict[str, StateEntry] | None) -> list[str]:
    """One line per stored value, naming its handle, kind and shape.

    For a host that wants to put what is in state in front of the model
    directly rather than relying on the capture breadcrumbs.
    """
    return [
        f"{handle_for(key)} — {entry.get('kind') or 'untyped'}, "
        f"{describe(entry.get('value'))}, from {entry.get('tool') or 'unknown'}"
        for key, entry in sorted(
            (tool_state or {}).items(),
            key=lambda item: item[1].get("seq", 0),
            reverse=True,
        )
    ]


def offers_handles(args_schema: Any, skip: frozenset[str] = frozenset()) -> bool:
    """Whether any parameter would gain a handle branch.

    Lets a caller skip wrapping a tool with nothing bulk to point at state.
    """
    return offer_handles(args_schema, skip) is not args_schema
