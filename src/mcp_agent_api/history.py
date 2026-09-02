"""Where the turn derivation used to live.

It is now :mod:`mcp_agent.history`, one layer down, because the model needs the
same answer the routes do — "this key, as it stood at that turn" — and
``mcp_agent`` is the lowest package both can reach. Two implementations of what
a turn is would be one too many.

Re-exported here rather than moved outright: the routes read the same names,
and so may anyone who found them.
"""

from mcp_agent.history import HUMAN, History, Turn, turns_of

__all__ = ["HUMAN", "History", "Turn", "turns_of"]
