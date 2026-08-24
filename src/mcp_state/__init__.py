"""Session state for MCP toolsets: capture, inspection, injection.

Some values a tool returns or takes are too large for the model to be handling — a clip geometry, an item collection, a raster footprint. This
package moves them from the tool that produced one to the tool that needs it,
through agent state, without them passing through the model.

It works against **any** MCP server. A server that says nothing about itself
still has its large returns captured by size, and its structured parameters
still gain a ``@state:<key>`` handle the model can point at a stored value
with. Nothing needs to be declared, installed or configured for that path.

**The model chooses which stored value to use, always.** It has the
conversation, and "the area the user just drew" is not something a heuristic
can be relied on to know. What this package does is make the choice cheap to
express, make what is stored legible, and refuse a call that got it wrong in a
way the model can act on. A server that wants a parameter it can trust tags it
:class:`mcp_runtime.declarations.NotAuthored`, and the parameter is narrowed
until a handle is the only thing it accepts.

Six moving parts, one namespace:

- :mod:`mcp_state.state` — the ``tool_state`` dict on graph state, keyed by
  ``<toolset>/<tool>/<field>``, values wrapped in a
  :class:`~mcp_state.state.StateEntry`.
- :mod:`mcp_state.middleware` — captures values out of tool returns into
  ``tool_state``, keeping large payloads out of the transcript, and records
  what the call that produced each value was given.
- :mod:`mcp_state.detect` — describes a stored value's shape for the listing
  the model chooses from. Shape only: what a value *means* is carried by the
  name its tool stored it under.
- :mod:`mcp_state.handles` — the ``@state:<key>`` reference a model passes in
  place of a value.
- :mod:`mcp_state.injection` — binds tools so handles are resolved before the
  call and a narrowed parameter cannot be written by the model.
- :mod:`mcp_state.receipts` — what a tool was handed from ``tool_state`` and
  which tool published it, so a host can show a call as it ran.
- :mod:`mcp_state.prompt` — the system-prompt fragment that explains all of
  the above to the model; a host appends it to its own instructions.

``inspect_state`` (:mod:`mcp_state.inspect`) is the dual of all of it: the
model pulling a value by key, having learned the key from a breadcrumb, and
paying tokens for it. Same namespace, opposite direction.

Nothing here depends on a particular chat UI — :mod:`mcp_agent` is one host
that uses it, not the only possible one. Install with the ``[state]`` extra.

**Trust assumption: every connected server is trusted.**

Declarations are honoured unconditionally, and participating is unilateral and
free, so "does not follow the spec" is not a boundary. A hostile server can
declare data keys it does not have, and have its return written into
``tool_state`` under a name chosen to be mistaken for another toolset's — the
model picks values by name, so a plausible name is the attack.

Undeclared capture widens this: a large value from any connected server can be
reached by a handle, so a server need not declare anything to get its output in
front of another tool. The model has to name it, which the transcript records,
but that is visibility rather than control.

That is fine while every server behind the index is yours, which is the only
configuration this is built for today. The moment an index aggregates
third-party servers, the missing control is **per-connection**, not per-key:
filter which server names may take part at all, once, where ``publications``
and ``bind_all_injected`` are applied. Doing it there changes no call site
here.

See ``docs/SESSION-STATE.md`` for the flows this implies.
"""

from mcp_state.detect import describe
from mcp_state.handles import (
    HANDLE_PREFIX,
    available,
    dereference,
    dereference_with_receipts,
    handle_for,
    is_handle,
    offer_handles,
    unresolved,
    unresolved_message,
)
from mcp_state.injection import StateRefusal, bind_all_injected, bind_injected
from mcp_state.inspect import make_inspect_state, read_state_key
from mcp_state.middleware import (
    CAPTURED_ARTIFACT_KEY,
    DEFAULT_CAPTURE_BYTES,
    SERVER_METADATA_KEY,
    StateCaptureMiddleware,
    call_inputs,
    owners,
    publications,
    restore_structured,
    state_keys,
    with_server_name,
)
from mcp_state.prompt import SESSION_STATE_PROMPT
from mcp_state.receipts import (
    INJECTED_ARTIFACT_KEY,
    Receipt,
    describe_receipt,
    receipts_of,
)
from mcp_state.state import (
    MODEL_AUTHORED,
    TOOL_STATE_KEY,
    AgentState,
    StateEntry,
    authored,
    merge_tool_state,
)

__all__ = [
    "CAPTURED_ARTIFACT_KEY",
    "DEFAULT_CAPTURE_BYTES",
    "HANDLE_PREFIX",
    "INJECTED_ARTIFACT_KEY",
    "MODEL_AUTHORED",
    "SERVER_METADATA_KEY",
    "SESSION_STATE_PROMPT",
    "TOOL_STATE_KEY",
    "AgentState",
    "Receipt",
    "StateCaptureMiddleware",
    "StateRefusal",
    "StateEntry",
    "authored",
    "available",
    "bind_all_injected",
    "bind_injected",
    "call_inputs",
    "dereference",
    "dereference_with_receipts",
    "describe",
    "describe_receipt",
    "handle_for",
    "is_handle",
    "make_inspect_state",
    "merge_tool_state",
    "offer_handles",
    "owners",
    "publications",
    "read_state_key",
    "receipts_of",
    "restore_structured",
    "state_keys",
    "unresolved",
    "unresolved_message",
    "with_server_name",
]
