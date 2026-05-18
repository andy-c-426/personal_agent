from unittest.mock import MagicMock
from personal_agent.core.conversation import Conversation
from personal_agent.core.memory import MemoryManager


class FakeConfig:
    context_threshold_ratio = 0.8
    recent_message_count = 5
    deepseek_model = "deepseek-chat"
    deepseek_api_key = "sk-test"
    deepseek_base_url = "https://api.deepseek.com"


class FakeClient:
    def __init__(self):
        self.chat = MagicMock()
        self.chat.completions = MagicMock()


def test_should_compress_under_threshold():
    conv = Conversation()
    conv.add_message("user", "short msg")
    mgr = MemoryManager(FakeConfig())
    assert not mgr.should_compress(conv, 65536)


def test_should_compress_over_threshold():
    conv = Conversation()
    long_text = "hello world " * 500
    for _ in range(5):
        conv.add_message("user", long_text)
    mgr = MemoryManager(FakeConfig())
    assert not mgr.should_compress(conv, model_context=10000)
    assert mgr.should_compress(conv, model_context=1000)


def test_maybe_compress_trims_then_compresses():
    conv = Conversation()
    long_text = "hello world " * 200
    for i in range(30):
        conv.add_message("user", f"{i}: {long_text}")
    conv.add_message("assistant", "final answer")

    mock_client = FakeClient()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Compressed summary of the conversation"))]
    )

    mgr = MemoryManager(FakeConfig(), client=mock_client)
    result = mgr.maybe_compress(conv, model_context=2000)

    assert result is not None
    assert result == "Compressed summary of the conversation"
    assert conv.summary == "Compressed summary of the conversation"
    assert len(conv.messages) <= FakeConfig.recent_message_count
    assert mock_client.chat.completions.create.called


def test_maybe_compress_no_client():
    conv = Conversation()
    long_text = "hello world " * 200
    for _ in range(30):
        conv.add_message("user", long_text)
    mgr = MemoryManager(FakeConfig(), client=None)
    result = mgr.maybe_compress(conv, model_context=1000)
    assert result is None


def test_maybe_compress_preserves_system_messages():
    conv = Conversation()
    conv.add_message("system", "You are a helpful assistant")
    long_text = "hello world " * 200
    for i in range(30):
        conv.add_message("user", f"{i}: {long_text}")
    conv.add_message("assistant", "final answer")

    mock_client = FakeClient()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Summary of the discussion"))]
    )

    mgr = MemoryManager(FakeConfig(), client=mock_client)
    result = mgr.maybe_compress(conv, model_context=2000)

    assert result is not None
    # System message should be preserved
    roles = [m.role for m in conv.messages]
    assert "system" in roles
    assert roles[0] == "system"


def test_maybe_compress_accumulates_summary():
    conv = Conversation()
    conv.summary = "First part: user asked about Python basics"
    long_text = "hello world " * 200
    for i in range(30):
        conv.add_message("user", f"{i}: {long_text}")
    conv.add_message("assistant", "final answer")

    mock_client = FakeClient()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Updated summary including Python discussion"))]
    )

    mgr = MemoryManager(FakeConfig(), client=mock_client)
    mgr.maybe_compress(conv, model_context=2000)

    # Verify the previous summary was included in the prompt
    call_args = mock_client.chat.completions.create.call_args
    user_content = call_args[1]["messages"][1]["content"]
    assert "First part" in user_content or "Previous conversation summary" in user_content


def test_maybe_compress_handles_api_error():
    conv = Conversation()
    long_text = "hello world " * 200
    for _ in range(30):
        conv.add_message("user", long_text)

    mock_client = FakeClient()
    mock_client.chat.completions.create.side_effect = Exception("API error")

    mgr = MemoryManager(FakeConfig(), client=mock_client)
    result = mgr.maybe_compress(conv, model_context=1000)
    assert result is None


def test_build_system_prompt():
    conv = Conversation()
    conv.summary = "Prior discussion about Python"
    mgr = MemoryManager(FakeConfig())
    prompt = mgr.build_system_prompt(12, conv)
    assert "12" in prompt
    assert "Prior discussion about Python" in prompt
    assert "kb_search" in prompt
    assert "web_search" in prompt


def test_model_context_zero_handled():
    conv = Conversation()
    long_text = "hello world " * 500
    for _ in range(50):
        conv.add_message("user", long_text)
    mgr = MemoryManager(FakeConfig())
    # model_context=0 should NOT silently become 65536 via or-operator bug
    assert mgr.should_compress(conv, model_context=0)
