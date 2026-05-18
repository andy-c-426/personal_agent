import json
from unittest.mock import MagicMock
from personal_agent.core.agent import Agent
from personal_agent.core.conversation import Conversation
from personal_agent.tools.registry import ToolRegistry, Tool
from personal_agent.core.memory import MemoryManager


class FakeConfig:
    deepseek_model = "deepseek-chat"
    deepseek_api_key = "sk-test"
    deepseek_base_url = "https://api.deepseek.com"
    max_tool_iterations = 10
    max_same_tool_calls = 3
    context_threshold_ratio = 0.8
    recent_message_count = 20


def _make_tool_call_mock(tc_id, func_name, func_args):
    """Create a MagicMock that mimics an OpenAI tool_call with string attributes."""
    func_mock = MagicMock()
    func_mock.name = func_name
    func_mock.arguments = func_args
    tc_mock = MagicMock()
    tc_mock.id = tc_id
    tc_mock.function = func_mock
    return tc_mock


def test_agent_responds_with_text():
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(
            message=MagicMock(content="Hello! How can I help?", tool_calls=None),
            finish_reason="stop",
        )]
    )

    conv = Conversation()
    registry = ToolRegistry()
    memory = MemoryManager(FakeConfig())

    agent = Agent(FakeConfig(), mock_client, registry, memory_manager=memory)
    response, _ = agent.run("Hi", conv)

    assert response == "Hello! How can I help?"


def test_agent_calls_tool_and_loops():
    call1 = MagicMock(
        choices=[MagicMock(
            message=MagicMock(
                content=None,
                tool_calls=[_make_tool_call_mock("call_1", "mock_tool", '{"query":"test"}')],
            ),
            finish_reason="tool_calls",
        )]
    )
    call2 = MagicMock(
        choices=[MagicMock(
            message=MagicMock(content="Found results for test", tool_calls=None),
            finish_reason="stop",
        )]
    )

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [call1, call2]

    conv = Conversation()
    registry = ToolRegistry()
    registry.register(Tool(
        name="mock_tool",
        description="Mock tool",
        function=lambda query: f"mock result for {query}",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
    ))
    memory = MemoryManager(FakeConfig())
    memory._model_context = 1000

    agent = Agent(FakeConfig(), mock_client, registry, memory_manager=memory)
    response, tool_calls = agent.run("Search for test", conv)

    assert "Found" in response
    assert len(tool_calls) == 1


def test_agent_stops_on_same_tool_repeat():
    call1 = MagicMock(
        choices=[MagicMock(
            message=MagicMock(
                content=None,
                tool_calls=[_make_tool_call_mock("call_1", "stuck_tool", "{}")],
            ),
            finish_reason="tool_calls",
        )]
    )

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [call1, call1, call1, call1]

    conv = Conversation()
    registry = ToolRegistry()
    registry.register(Tool(
        name="stuck_tool",
        description="Always returns same thing",
        function=lambda: "same result",
        parameters={"type": "object", "properties": {}},
    ))
    memory = MemoryManager(FakeConfig())
    memory._model_context = 1000

    agent = Agent(FakeConfig(), mock_client, registry, memory_manager=memory)
    response, tool_calls = agent.run("test", conv)

    assert len(tool_calls) <= 3


def test_agent_passes_kb_doc_count():
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(
            message=MagicMock(content="I know about your docs", tool_calls=None),
            finish_reason="stop",
        )]
    )

    conv = Conversation()
    registry = ToolRegistry()
    memory = MemoryManager(FakeConfig())

    agent = Agent(FakeConfig(), mock_client, registry, kb_doc_count=42, memory_manager=memory)
    response, _ = agent.run("How many docs?", conv)

    call_args = mock_client.chat.completions.create.call_args
    messages = call_args[1]["messages"]
    system_msg = messages[0]["content"]
    assert "42" in system_msg
