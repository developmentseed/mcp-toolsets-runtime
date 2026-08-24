"""The system-prompt fragment that tells a model how session state works.

Everything else in this package is host-side machinery the model never sees.
The model's half of the contract is two surfaces that do reach it: the
``[state updated: …]`` breadcrumbs capture writes, and the ``inspect_state``
tool. This fragment explains them once, in prompt form, so the model drives
them deliberately instead of inferring them from tool descriptions alone — and
asks it to carry the provenance they record into its answers.

It is host-agnostic and self-contained. A host that replaces the bundled
prompt appends it to its own instructions::

    system_prompt = MY_PROMPT + "\\n\\n" + SESSION_STATE_PROMPT

which is exactly how :data:`mcp_agent.main.SYSTEM_PROMPT` is built. Append it
only when the machinery is actually wired in (capture middleware, bound tools,
``inspect_state`` — see ``docs/CONSUMING.md``): a prompt that describes notes
which never appear is worse than no prompt.
"""

SESSION_STATE_PROMPT = """\
How this system moves tool data:

Large tool values — geometries, item collections, data arrays — do not pass \
through this conversation. The host keeps them in session state, and you move \
them between tools by naming them.

- A "[state updated: <key> — …]" note in a tool result means the value was \
stored under that key. The value itself is not in the transcript.
- To pass a stored value to a tool, write "@state:<key>" as the whole \
argument. The host substitutes the stored value before the tool runs. Prefer \
a handle over copying a large value into a call. A handle only works as a \
whole argument, never as a fragment inside one, and only on a parameter whose \
schema accepts it — elsewhere it is just a string, and will be taken as one.
- A key reads "<toolset>/<tool>/<field>", so it says which call produced the \
value. Choose between stored values on that and on what the tool you are \
calling asks for, not on which was written most recently.
- Some parameters accept nothing but a handle. Their description says the \
value must already exist; write "@state:<key>", never a value of your own. If \
nothing suitable has been stored yet, call the tool that produces it first.
- Call inspect_state with a bare key from a "[state updated: …]" note — not \
an "@state:" handle — to read or search a stored value when you need its \
content.

Provenance: when you present a result, name the tool that produced the data \
behind it and say where that data was reused — for example "clipped with the \
area of interest that search_datasets returned". If you passed a handle that \
turns out not to be the value the user meant, say so and repeat the call with \
the intended "@state:<key>"."""
