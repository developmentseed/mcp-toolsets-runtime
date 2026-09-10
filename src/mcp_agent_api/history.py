"""A thread's past turns, under the names the routes read them by.

The derivation itself is :mod:`mcp_agent.history`, one layer down, because the
model needs the same answer the routes do — "this key, as it stood at that
turn" — and ``mcp_agent`` is the lowest package both can reach. Two
implementations of what a turn is would be one too many.
"""

from mcp_agent.history import HUMAN, History, Turn, turns_of

__all__ = ["HUMAN", "History", "Turn", "turns_of"]
