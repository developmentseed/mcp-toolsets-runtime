"""Session state for MCP toolsets: capture, inspection, injection.

Some values a tool returns or takes are bulk data the model has no business
handling — a clip geometry, an item collection, a raster footprint. This
package moves them from the tool that produced one to the tool that needs it,
through agent state, without them passing through the model.

It works against **any** MCP server. A server that says nothing about itself
still has its large returns captured (by size, labelled by recognising the
value's own shape), and its structured parameters still gain a ``@state:<key>``
handle the model can point at a stored value with. Nothing needs to be
declared, installed or configured for that path.

Declaring is an accelerator. A server built on :mod:`mcp_runtime` tags a
parameter with :class:`mcp_runtime.declarations.Kind`, and the parameter leaves
the model's schema entirely: the client matches the kind and fills it, so the
model neither sees the value nor spends a token choosing it.

Five moving parts, one namespace:

- :mod:`mcp_state.state` — the ``tool_state`` dict on graph state, keyed by
  ``<toolset>/<field>``, values wrapped in a :class:`~mcp_state.state.StateEntry`.
- :mod:`mcp_state.middleware` — captures values out of tool returns into
  ``tool_state``, keeping bulky payloads out of the transcript.
- :mod:`mcp_state.detect` — recognises what a captured value is from its own
  shape, so an undeclared value is still labelled.
- :mod:`mcp_state.handles` — the ``@state:<key>`` reference a model passes in
  place of a value.
- :mod:`mcp_state.injection` — binds tools so declared parameters are filled
  by the client and handles are resolved before the call.

``inspect_state`` (:mod:`mcp_state.inspect`) is the dual of all of it: the
model pulling a value by key, having learned the key from a breadcrumb, and
paying tokens for it. Same namespace, opposite direction.

Nothing here depends on a particular chat UI — :mod:`mcp_agent` is one host
that uses it, not the only possible one. Install with the ``[state]`` extra.

**Trust assumption: every connected server is trusted.**

Declarations are honoured unconditionally, and participating is unilateral and
free, so "does not follow the spec" is not a boundary. A hostile server can
tag a parameter with a kind and be handed the matching value from
``tool_state`` on its next call, with no model or user in the loop; or declare
that it publishes a kind, and have its return written into ``tool_state``
where another server's tool consumes it — poisoning an input that, by design,
nothing in the transcript shows.

Undeclared capture widens this a little: a large value from any connected
server can be reached by a handle, so a server need not declare anything to
get its output in front of another tool. The model has to name it, which the
transcript records, but that is visibility rather than control.

That is fine while every server behind the index is yours, which is the only
configuration this is built for today. The moment an index aggregates
third-party servers, the missing control is **per-connection**, not per-key:
filter which server names may take part at all, once, where ``publications``
and ``bind_all_injected`` are applied. Doing it there changes no call site
here.

See ``docs/SESSION-STATE.md`` for the flows this implies.
"""

from mcp_state.detect import describe, detect_kind
from mcp_state.handles import (
    HANDLE_PREFIX,
    available,
    dereference,
    handle_for,
    is_handle,
    offer_handles,
)
from mcp_state.injection import bind_all_injected, bind_injected
from mcp_state.inspect import make_inspect_state, read_state_key
from mcp_state.middleware import (
    DEFAULT_CAPTURE_BYTES,
    StateCaptureMiddleware,
    publications,
    published_kinds,
    state_keys,
)
from mcp_state.state import (
    TOOL_STATE_KEY,
    AgentState,
    StateEntry,
    entries_of_kind,
    merge_tool_state,
)
from mcp_state.wiring import (
    Unsatisfiable,
    partition_usable,
    raise_unsatisfiable,
    unsatisfiable,
)

__all__ = [
    "DEFAULT_CAPTURE_BYTES",
    "HANDLE_PREFIX",
    "TOOL_STATE_KEY",
    "AgentState",
    "StateCaptureMiddleware",
    "StateEntry",
    "Unsatisfiable",
    "available",
    "bind_all_injected",
    "bind_injected",
    "dereference",
    "describe",
    "detect_kind",
    "entries_of_kind",
    "handle_for",
    "is_handle",
    "make_inspect_state",
    "merge_tool_state",
    "offer_handles",
    "partition_usable",
    "publications",
    "published_kinds",
    "raise_unsatisfiable",
    "read_state_key",
    "state_keys",
    "unsatisfiable",
]
