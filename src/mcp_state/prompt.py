"""The system-prompt fragment that tells a model how session state works.

Everything else in this package is host-side machinery the model never sees.
The model's half of the contract is three surfaces that do reach it: the
``[state updated: …]`` breadcrumbs capture writes, the listing a refusal puts
in front of it, and the ``inspect_state`` tool — including the part of it that
says more than one turn wrote a key, which a model has no other way to notice.
This fragment explains them once, in prompt form, so the model drives them
deliberately instead of inferring them from tool descriptions alone — and asks
it to carry the provenance they record into its answers.

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
- A listing may add "(you wrote: <parameter>)". That means the call which \
produced the value was given an argument you wrote rather than one a tool \
supplied, so the value rests on it. Prefer a value that does not say this \
where you have the choice, and say so when you present a result that depends \
on one.
- Some parameters accept nothing but a handle. Their description says the \
value must already exist; write "@state:<key>", never a value of your own. If \
nothing suitable has been stored yet, call the tool that produces it first.
- Call inspect_state with a bare key from a "[state updated: …]" note — not \
an "@state:" handle — to read or search a stored value when you need its \
content.
- A key holds one value, so a later call to the same tool replaces what was \
there. A "[state updated: …]" note saying a write "replaces what <key> held \
at turn <n>", a read saying that several turns wrote the key, or a listing \
marked "written in N turns", all mean the same thing: an earlier turn may hold \
a different value from the one in front of you. Watch for it especially when \
you are about to pass "@state:<key>" to a tool — a handle always resolves to \
the current value. When the question is about an earlier one — \
comparing, or checking what a previous answer used — call inspect_state with \
turn=<n>, counting the user's questions from 1. Reading it is the only way to \
find out whether it differs; the note does not claim it does. If a turn is no \
longer retained, say so; do not answer from the current value as though it \
were the earlier one.

Provenance: when you present a result, name the tool that produced the data \
behind it and say where that data was reused — for example "clipped with the \
area of interest that search_datasets returned". Where a result rests on a \
value you wrote rather than one a tool produced, say so plainly; do not \
recite provenance that has no bearing on whether an answer can be relied on. \
If you passed a handle that turns out not to be the value the user meant, say \
so and repeat the call with the intended "@state:<key>"."""
