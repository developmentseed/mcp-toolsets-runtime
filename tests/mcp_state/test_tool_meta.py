from mcp_state import tool_meta


def test_tool_meta_is_read_wherever_the_client_put_it():
    """`langchain.mcp` nests an MCP tool's `_meta` under its own `mcp`
    namespace; a tool built here carries it flat. Every reader of a server-side
    declaration goes through `tool_meta`, and a declaration that is not found
    does not fail — it just stops applying, which is why this is pinned."""

    class Stub:
        def __init__(self, metadata):
            self.metadata = metadata

    nested = Stub({"mcp": {"tool": {"_meta": {"k": 1}}, "server": {"name": "x"}}})
    flat = Stub({"_meta": {"k": 2}})

    assert tool_meta(nested) == {"k": 1}
    assert tool_meta(flat) == {"k": 2}
    assert tool_meta(Stub({"mcp": {"server": {"name": "x"}}})) == {}
    assert tool_meta(Stub(None)) == {}
