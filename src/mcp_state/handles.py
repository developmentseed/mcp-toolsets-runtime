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

1. Every parameter that could hold a structured value — schema type ``object``
   or ``array`` — gains a second accepted form: the string ``@state:<key>``.
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
from mcp_state.receipts import Receipt, receipt_for
from mcp_state.state import StateEntry, authored

#: Prefix marking an argument as a reference into ``tool_state``.
HANDLE_PREFIX = "@state:"

#: JSON Schema types worth passing by reference. A string or a number is
#: cheaper for a model to emit than to name.
STRUCTURED_TYPES = frozenset({"object", "array"})


def handle_for(key: str) -> str:
    """The handle a model writes to pass the value stored under ``key``."""
    return f"{HANDLE_PREFIX}{key}"


def is_handle(value: Any) -> bool:
    """Whether an argument is a state reference rather than a literal value."""
    return isinstance(value, str) and value.startswith(HANDLE_PREFIX)


def handle_key(value: str) -> str:
    """The ``tool_state`` key a handle refers to."""
    return value[len(HANDLE_PREFIX) :]


def _could_be_structured(schema: Any) -> bool:
    """Whether a parameter's schema admits a value worth passing by name.

    A test on the declared *type*, never on a value's size: this runs at
    connect, when nothing has been produced to measure. (Capture is the
    opposite — it holds the value, so it weighs it.)

    It errs towards yes. A parameter with no stated type, or one behind a
    ``$ref`` or a combinator, could hold anything. A false yes costs a few
    schema tokens on a parameter nobody ever points at state; a false no
    silently withdraws the mechanism from one that needed it.
    """
    if not isinstance(schema, dict):
        return False
    if any(key in schema for key in ("$ref", "anyOf", "oneOf", "allOf")):
        return True
    declared = schema.get("type")
    if declared is None:
        return True
    if isinstance(declared, list):
        return bool(STRUCTURED_TYPES.intersection(declared))
    return declared in STRUCTURED_TYPES


def _handle_branch() -> dict[str, Any]:
    """The schema arm accepting a ``@state:<key>`` reference."""
    return {
        "type": "string",
        "pattern": f"^{HANDLE_PREFIX}",
        "description": (
            "A session-state reference, e.g. "
            f"{handle_for('dataset-search/search_datasets/area_of_interest')} — the key from "
            "a "
            "[state updated: …] note. The value is substituted before the "
            "tool runs, so prefer this over repeating a large value."
        ),
    }


def _with_handle_branch(schema: dict[str, Any]) -> dict[str, Any]:
    """One parameter's schema, also accepting a handle string."""
    original = {key: value for key, value in schema.items() if key != "description"}
    branched: dict[str, Any] = {"anyOf": [original, _handle_branch()]}
    if description := schema.get("description"):
        branched["description"] = description
    return branched


def handle_only(schema: dict[str, Any]) -> dict[str, Any]:
    """One parameter's schema, accepting *nothing but* a handle string.

    For a parameter its server tagged :class:`~mcp_runtime.declarations.
    NotAuthored`. Dropping the original arm is the enforcement: a model cannot
    write a literal into a parameter whose only accepted form is a string
    matching ``^@state:``, so the value it passes is necessarily one some tool
    already produced.

    The parameter's own description is kept — it is the sentence that says why
    the constraint is there, and the only part of this a model reads as prose.
    """
    branch = _handle_branch()
    if description := schema.get("description"):
        branch["description"] = f"{description} {branch['description']}"
    return branch


def offer_handles(
    args_schema: Any,
    skip: frozenset[str] = frozenset(),
    only: frozenset[str] = frozenset(),
) -> Any:
    """The schema with every structured parameter also accepting ``@state:<key>``.

    Pure and shallow: only the entries under ``properties`` change, so
    ``$defs``, ``required`` and everything else survive byte-for-byte.

    ``skip`` names parameters an injected declaration already handles — those
    are about to be removed from the schema entirely, so offering a handle for
    them would only confuse the model. ``only`` names parameters that must take
    a handle and nothing else (:func:`handle_only`); it wins over ``skip``, and
    over the structured-type test, because the constraint is the server's
    stated intent rather than an inference from the schema.
    """
    if not isinstance(args_schema, dict):
        return args_schema
    properties = args_schema.get("properties")
    if not isinstance(properties, dict):
        return args_schema

    def rewritten(name: str, schema: Any) -> Any:
        if not isinstance(schema, dict):
            return schema
        if name in only:
            return handle_only(schema)
        if name not in skip and _could_be_structured(schema):
            return _with_handle_branch(schema)
        return schema

    updated = {name: rewritten(name, schema) for name, schema in properties.items()}
    if updated == properties:
        return args_schema
    return {**args_schema, "properties": updated}


def dereference_with_receipts(
    arguments: dict[str, Any], tool_state: dict[str, StateEntry] | None
) -> tuple[dict[str, Any], dict[str, Receipt]]:
    """:func:`dereference`, also reporting which arguments it resolved.

    The receipts (:mod:`mcp_state.receipts`) say where each substituted value
    came from, so a host can show the call as it actually ran rather than as
    the model wrote it.
    """
    state = tool_state or {}
    resolved: dict[str, Any] = {}
    receipts: dict[str, Receipt] = {}
    for name, value in arguments.items():
        if is_handle(value):
            key = handle_key(value)
            entry = state.get(key)
            if entry is not None:
                resolved[name] = entry.get("value")
                receipts[name] = receipt_for(key, entry)
                continue
        resolved[name] = value
    return resolved, receipts


def dereference(
    arguments: dict[str, Any], tool_state: dict[str, StateEntry] | None
) -> dict[str, Any]:
    """Replace every ``@state:<key>`` argument with the value it names.

    A handle naming a key that does not exist is left as the literal string,
    rather than passing ``None`` to a tool that would fail somewhere less
    legible. Substitution reports nothing itself: :func:`unresolved` is what
    finds a handle this left behind, and :mod:`mcp_state.injection` runs it
    before the call goes out.
    """
    return dereference_with_receipts(arguments, tool_state)[0]


def unresolved(value: Any, path: str = "") -> list[tuple[str, str]]:
    """Every ``@state:`` reference still present, as ``(path, key)`` pairs.

    :func:`dereference` substitutes a handle only where one is a whole
    argument. Two things survive it, and both reach the server as the literal
    string:

    - a handle written *inside* an argument — ``request.area`` on a tool taking
      an opaque ``dict[str, Any]``, which is what a model does when the field
      that wants the value is nested;
    - a handle naming a key that is not in state.

    Neither is caught by the tool's own schema when that schema is permissive,
    so the failure surfaces from the remote service as an error quoting our own
    prefix back. This finds them before the call goes out.

    Paths are dotted, with list indices in brackets: ``request.area``,
    ``features[0].geometry``.
    """
    if is_handle(value):
        return [(path, handle_key(value))]
    if isinstance(value, dict):
        return [
            found
            for name, item in value.items()
            for found in unresolved(item, f"{path}.{name}" if path else str(name))
        ]
    if isinstance(value, list):
        return [
            found
            for index, item in enumerate(value)
            for found in unresolved(item, f"{path}[{index}]")
        ]
    return []


def unresolved_message(
    tool_name: str,
    found: list[tuple[str, str]],
    tool_state: dict[str, StateEntry] | None,
    not_authored: frozenset[str] = frozenset(),
) -> str:
    """What to tell the model about handles that could not be substituted.

    Names the path rather than the parameter, because on a tool taking an
    opaque dict the parameter is the whole request and the path is the only
    thing that says which field is wrong. Each line says which of the two
    failures it is: a key nobody published is the model's to correct by running
    another tool, while a nested handle is one the mechanism cannot serve at
    all.

    ``not_authored`` changes the advice, not the diagnosis. Reading the value
    with ``inspect_state`` and writing the field by hand is the right answer
    for an ordinary opaque field, and the wrong one for a tool holding a
    parameter its server said a model must not write: there the same value has
    a parameter of its own, and "write it yourself" is an instruction to carry
    it around the constraint. Observed, not anticipated — a model refused for a
    handle nested in an opaque request read this, fetched the value, and wrote
    it in.
    """
    state = tool_state or {}
    lines = [
        f"  {path or 'the argument'}: {handle_for(key)} — "
        + (
            "a handle is substituted only where it is a whole argument, never "
            "inside one, so this would have reached the tool as text."
            if key in state
            else "no such key in session state."
        )
        for path, key in found
    ]
    listing = available(state)
    if not listing:
        closing = "Nothing has been published to session state yet."
    elif not_authored:
        named = ", ".join(f"{name!r}" for name in sorted(not_authored))
        closing = (
            f"Do not read the value and write it in — {tool_name} takes it as "
            f"{named}, which is the parameter to pass the handle to. If nothing "
            "holds it yet, call the tool that produces it first."
        )
    else:
        closing = (
            "Read a value with inspect_state and write the field yourself, or "
            "call the tool that produces it first."
        )
    return "\n".join(
        [f"{tool_name} was not called. Unresolved session-state references:"]
        + lines
        + [closing]
        + [f"  {line}" for line in listing]
    )


def available(tool_state: dict[str, StateEntry] | None) -> list[str]:
    """One line per stored value: its handle, shape, publisher and provenance.

    For a host that wants to put what is in state in front of the model
    directly rather than relying on the capture breadcrumbs, and what a refusal
    lists so the model can correct itself.

    A value whose producing call had model-authored arguments says which — the
    model is choosing between these, and "this one rests on something you wrote
    rather than something a tool found" is the fact most likely to change that
    choice.

    Only those are named, and that is a judgement about *this* surface rather
    than about the record. Every line here costs context, the unremarkable case
    is a parameter filled from state, and the model has just seen the call
    anyway. A surface without those constraints should show the whole of
    ``entry["inputs"]`` — a host panel is read long after the call scrolled
    away, and half a record leaves the chain unwalkable from the one place
    built to display it.
    """
    return [
        f"{handle_for(key)} — {describe(entry.get('value'))}, "
        f"from {entry.get('tool') or 'unknown'}"
        + (
            f" (you wrote: {', '.join(written)})"
            if (written := authored(entry))
            else ""
        )
        for key, entry in sorted(
            (tool_state or {}).items(),
            key=lambda item: item[1].get("seq", 0),
            reverse=True,
        )
    ]


def offers_handles(
    args_schema: Any,
    skip: frozenset[str] = frozenset(),
    only: frozenset[str] = frozenset(),
) -> bool:
    """Whether any parameter would gain a handle branch.

    Lets a caller skip wrapping a tool with nothing to point at state.
    """
    return offer_handles(args_schema, skip, only) is not args_schema
