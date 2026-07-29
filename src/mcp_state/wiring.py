"""Check that every injected parameter has something that can satisfy it.

An ``Injected`` parameter names a kind (or an explicit key). If nothing
connected *publishes* that kind, the declaration can never be satisfied: a
required one raises on first use, an optional one silently stays empty
forever. Both are wiring bugs — a typo in a kind string, a toolset deployed
without the one that feeds it — and both are invisible until a user happens
to trigger the tool.

**This cannot be a build-time gate, and it is worth being precise about why.**
Consuming a kind another toolset produces is the whole point, so a server
validating its own tools at ``build_server`` time has no way to distinguish
"nobody produces this" from "the producer lives elsewhere" — the information
simply is not there. The earliest point with complete *and accurate*
information is the client, once connected: it holds every server's
declarations, and it holds them for the servers actually running rather than
the ones some manifest expects.

So this runs at connect. Call :func:`unsatisfiable` after loading tools and
decide what it means for your host — log it, surface it in a health route, or
:func:`raise_unsatisfiable` to refuse to start. A deployment that
intentionally connects a consumer without its producer is legitimate, so
nothing here fails by default.

A useful side effect: because the check is on the *wiring* rather than on
membership of :mod:`mcp_runtime.kinds`, a mistyped or unregistered kind shows
up here as unsatisfiable. That is what lets the vocabulary stay open — a
consumer repo can mint its own kinds without this package knowing them, and
still have a typo caught.
"""

from dataclasses import dataclass

from langchain_core.tools import BaseTool

from mcp_state.injection import declarations_for, satisfiable, wanted
from mcp_state.middleware import produced


@dataclass(frozen=True)
class Unsatisfiable:
    """One injected parameter nothing connected can fill."""

    tool: str
    parameter: str
    #: The kind, or ``key:<stateKey>`` when the declaration named one exactly.
    wants: str
    #: Required parameters raise on first use; optional ones stay empty.
    required: bool
    #: Declared ``model_fallback``, so the model is asked for it instead and
    #: the tool stays callable.
    model_fallback: bool

    @property
    def fatal(self) -> bool:
        """Whether this makes the tool impossible to call."""
        return self.required and not self.model_fallback

    def __str__(self) -> str:
        if self.model_fallback:
            outcome = "the model is asked for it"
        elif self.required:
            outcome = "the tool cannot be called"
        else:
            outcome = "it stays empty"
        return f"{self.tool}.{self.parameter} wants {self.wants} — {outcome}"


def unsatisfiable(tools: list[BaseTool]) -> list[Unsatisfiable]:
    """Every injected declaration nothing in ``tools`` can satisfy.

    Empty means the wiring is sound: each injected parameter has at least one
    publisher of the right kind (or of the exact key it asked for) among the
    connected servers. Ordered by tool then parameter, so the output is stable
    enough to diff between deployments.
    """
    published = produced(tools)
    found = [
        Unsatisfiable(
            tool=tool.name,
            parameter=declaration["parameter"],
            wants=wanted(declaration) or "nothing (malformed declaration)",
            required=bool(declaration.get("required", True)),
            model_fallback=bool(declaration.get("modelFallback")),
        )
        for tool in tools
        for declaration in declarations_for(tool)
        if not satisfiable(declaration, published)
    ]
    return sorted(found, key=lambda item: (item.tool, item.parameter))


def usable(tools: list[BaseTool]) -> tuple[list[BaseTool], list[Unsatisfiable]]:
    """Split tools into the ones a model can actually call, and why the rest can't.

    A tool with a *required* injected parameter nothing publishes is dead: every
    call raises before it reaches the server. Advertising it to the model only
    buys failed turns and a confusing error, so the usual handling is to leave
    it out of the agent and report what was left out::

        agent_tools, withheld = usable(bind_all_injected(tools))

    Only a declaration that is *required* and has no ``model_fallback``
    withholds a tool — the two other outcomes leave it callable. An optional
    one is simply omitted from the call, so the tool picks its own default; a
    fallback one stays in the schema for the model to fill.

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
        "no connected tool publishes what these injected parameters need — "
        "check the kind strings, or connect the toolset that produces them:"
        f"\n  {listed}"
    )
