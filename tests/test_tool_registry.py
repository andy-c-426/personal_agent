from personal_agent.tools.registry import ToolRegistry, Tool


def test_register_and_get_tool():
    def my_func(query: str) -> str:
        """Search for things."""
        return f"found: {query}"

    registry = ToolRegistry()
    registry.register(Tool(
        name="my_search",
        description="Search for things",
        function=my_func,
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"}
            },
            "required": ["query"],
        },
    ))
    assert registry.get("my_search") is not None
    result = registry.execute("my_search", query="test")
    assert result == "found: test"


def test_list_tool_schemas():
    def foo(x: str) -> str:
        return x

    registry = ToolRegistry()
    registry.register(Tool(
        name="foo",
        description="Do foo",
        function=foo,
        parameters={"type": "object", "properties": {"x": {"type": "string"}}},
    ))
    schemas = registry.schemas()
    assert len(schemas) == 1
    assert schemas[0]["function"]["name"] == "foo"


def test_execute_unknown_tool():
    registry = ToolRegistry()
    result = registry.execute("nonexistent")
    assert "not found" in result.lower()


def test_execute_with_error():
    def bad_func(**kwargs):
        raise ValueError("boom")

    registry = ToolRegistry()
    registry.register(Tool(
        name="bad",
        description="Bad tool",
        function=bad_func,
        parameters={"type": "object", "properties": {}},
    ))
    result = registry.execute("bad")
    assert "error" in result.lower()
