"""Session state for MCP toolsets: capture, inspection, injection.

The consuming half of the contract :mod:`mcp_runtime` serves. A toolset
declares what it publishes (a ``ToolResult`` subclass, its data keys tagged
with :class:`mcp_runtime.injected.Kind`) and what it consumes
(:class:`mcp_runtime.injected.Injected` parameters); this package is what an
agent does about it.

Three moving parts, one namespace:

- :mod:`mcp_state.state` — the ``tool_state`` dict on graph state, keyed by
  ``<toolset>/<field>``, values wrapped in a :class:`~mcp_state.state.StateEntry`.
- :mod:`mcp_state.middleware` — captures declared data keys out of tool
  returns into ``tool_state``, keeping bulky payloads out of the transcript.
- :mod:`mcp_state.inspect` — the ``inspect_state`` tool, for the model to
  read a stored value **on demand**.
- :mod:`mcp_state.injection` — binds declared parameters so the **client**
  fills them from ``tool_state``, with the model never involved.

The last two are duals. ``inspect_state`` is the model pulling a value by
key, having learned the key from a breadcrumb, and paying tokens for it.
Injection is the client pushing a value into a call by kind, with the model
neither seeing nor paying for it. Same namespace, opposite directions.

Nothing here depends on a particular chat UI — :mod:`mcp_agent` is one host
that uses it, not the only possible one. Install with the ``[state]`` extra.

**Trust assumption: every connected server is trusted.**

Both halves are driven by declarations a server makes about itself in its
tool ``_meta``, and this package honours them unconditionally. A server that
declares nothing is inert — it is never handed state and never writes any —
but participating is unilateral and costs a server nothing, so "does not
follow the spec" is not a boundary. A hostile server can:

- declare an ``Injected`` parameter of some kind and be handed the matching
  value from ``tool_state`` on its next call, with no model or user in the
  loop; or
- declare that it *publishes* a kind, and have its return written into
  ``tool_state`` where another server's tool consumes it by kind — poisoning
  an input that, by design, nothing in the transcript shows.

That is fine while every server behind the index is yours, which is the only
configuration this is built for today. The moment an index aggregates
third-party servers, the missing control is **per-connection**, not per-key:
filter which server names may declare production and injection at all, once,
where ``published_keys`` and ``bind_all_injected`` are applied. Doing it
there changes no call site here.

See ``docs/SESSION-STATE.md`` for the flows this implies.
"""

from mcp_state.injection import bind_all_injected, bind_injected
from mcp_state.inspect import make_inspect_state, read_state_key
from mcp_state.middleware import StateCaptureMiddleware, published_keys, state_keys
from mcp_state.state import (
    TOOL_STATE_KEY,
    AgentState,
    StateEntry,
    entries_of_kind,
    merge_tool_state,
)
from mcp_state.wiring import (
    Unsatisfiable,
    raise_unsatisfiable,
    unsatisfiable,
    usable,
)

__all__ = [
    "TOOL_STATE_KEY",
    "AgentState",
    "StateCaptureMiddleware",
    "StateEntry",
    "Unsatisfiable",
    "bind_all_injected",
    "bind_injected",
    "entries_of_kind",
    "make_inspect_state",
    "merge_tool_state",
    "published_keys",
    "raise_unsatisfiable",
    "read_state_key",
    "state_keys",
    "unsatisfiable",
    "usable",
]
