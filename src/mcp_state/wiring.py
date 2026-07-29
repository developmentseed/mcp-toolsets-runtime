"""Check that every declared parameter has something that can satisfy it.

A ``Kind``-tagged parameter names the kind it takes. If nothing connected
*publishes* that kind, the declaration can never be satisfied. Whether that
matters depends on what the tag said:

- ``model_generatable`` (the default) is fine — the parameter stays in the
  model's schema and is filled the way any plain MCP client would fill it.
- ``model_generatable=False`` on a *required* parameter is fatal: every call
  raises before it reaches the server, so the tool should not be offered.
- ``model_generatable=False`` on an *optional* one leaves it empty forever,
  and the tool falls back to whatever default it chose.

All three are worth reporting — a mistyped kind string or a toolset deployed
without the one that feeds it is invisible until a user happens to trigger it.

Runs at connect, the first point holding every connected server's declarations
— and holding them for the servers actually running rather than the ones a
manifest expects. A server checking its own tools cannot do it: the producer of
a kind usually lives in another toolset.

The check is on the wiring, not on membership of :mod:`mcp_runtime.kinds`, so
a kind this package has never heard of is fine and a mistyped one still shows
up as unsatisfiable.
"""

from dataclasses import dataclass

from langchain_core.tools import BaseTool

from mcp_state.injection import (
    declarations_for,
    model_generatable,
    satisfiable,
    wants,
)
from mcp_state.middleware import published_kinds


@dataclass(frozen=True)
class Unsatisfiable:
    """One declared parameter nothing connected publishes the kind for."""

    tool: str
    parameter: str
    #: The kind the parameter asked for.
    wants: str
    #: Whether the tool's own schema requires the parameter.
    required: bool
    #: Whether the tool said a model may produce the value instead.
    model_generatable: bool

    @property
    def fatal(self) -> bool:
        """Whether this makes the tool impossible to call."""
        return self.required and not self.model_generatable

    def __str__(self) -> str:
        if self.model_generatable:
            outcome = "the model is asked for it"
        elif self.required:
            outcome = "the tool cannot be called"
        else:
            outcome = "it stays empty"
        return f"{self.tool}.{self.parameter} wants {self.wants} — {outcome}"


def unsatisfiable(tools: list[BaseTool]) -> list[Unsatisfiable]:
    """Every declared parameter nothing in ``tools`` publishes the kind for.

    Empty means the wiring is sound: each declared parameter has at least one
    publisher of the right kind among the connected servers. Ordered by tool
    then parameter, so the output is stable enough to diff between deployments.
    """
    published = published_kinds(tools)
    found = [
        Unsatisfiable(
            tool=tool.name,
            parameter=declaration["parameter"],
            wants=wants(declaration) or "nothing (malformed declaration)",
            required=bool(declaration.get("required", True)),
            model_generatable=model_generatable(declaration),
        )
        for tool in tools
        for declaration in declarations_for(tool)
        if not satisfiable(declaration, published)
    ]
    return sorted(found, key=lambda item: (item.tool, item.parameter))


def partition_usable(
    tools: list[BaseTool],
) -> tuple[list[BaseTool], list[Unsatisfiable]]:
    """Split tools into the ones a model can actually call, and why the rest can't.

    A tool with a required parameter that nothing publishes and a model may not
    invent is dead: every call raises before it reaches the server. Advertising
    it to the model only buys failed turns and a confusing error, so the usual
    handling is to leave it out of the agent and report what was left out::

        agent_tools, withheld = partition_usable(bind_all_injected(tools))

    Only a *fatal* declaration withholds a tool — the other outcomes leave it
    callable. An optional one is simply omitted from the call, so the tool
    picks its own default; a model-generatable one stays in the schema.

    This is about whether a *publisher is connected*, never about whether a
    value has been published yet. A tool whose producer is connected but has
    not run stays available — the model is meant to call it, be told to run the
    producer first, and try again.
    """
    withheld = [item for item in unsatisfiable(tools) if item.fatal]
    blocked = {item.tool for item in withheld}
    return [tool for tool in tools if tool.name not in blocked], withheld


def raise_unsatisfiable(tools: list[BaseTool], *, fatal_only: bool = True) -> None:
    """Refuse to start when the state wiring cannot work.

    ``fatal_only`` (the default) reports only declarations that leave a tool
    uncallable. Pass ``False`` to treat any unsatisfiable declaration as an
    error, including ones that degrade to a tool default or to the model.
    """
    found = [item for item in unsatisfiable(tools) if item.fatal or not fatal_only]
    if not found:
        return
    listed = "\n  ".join(str(item) for item in found)
    raise RuntimeError(
        "no connected tool publishes what these declared parameters need — "
        "check the kind strings, or connect the toolset that produces them:"
        f"\n  {listed}"
    )
