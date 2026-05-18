import json
from personal_agent.core.conversation import Conversation, Message


def test_add_message():
    conv = Conversation()
    conv.add_message("user", "hello")
    assert len(conv.messages) == 1
    assert conv.messages[0].role == "user"
    assert conv.messages[0].content == "hello"


def test_to_dicts():
    conv = Conversation()
    conv.add_message("system", "You are helpful")
    conv.add_message("user", "hi")
    conv.add_message("assistant", "Hello!")
    dicts = conv.to_dicts()
    assert len(dicts) == 3
    assert dicts[0] == {"role": "system", "content": "You are helpful"}
    assert dicts[2] == {"role": "assistant", "content": "Hello!"}


def test_message_with_tool_calls():
    conv = Conversation()
    conv.add_message("assistant", "Let me search", tool_calls=[{"id": "1", "type": "function", "function": {"name": "kb_search", "arguments": '{"query":"test"}'}}])
    conv.add_message("tool", "result 1", tool_call_id="1")
    dicts = conv.to_dicts()
    assert dicts[0]["tool_calls"][0]["id"] == "1"
    assert dicts[1]["role"] == "tool"
    assert dicts[1]["tool_call_id"] == "1"


def test_trim_to_fit_recent():
    conv = Conversation()
    for i in range(50):
        conv.add_message("user", f"message {i}")
    conv.trim_to_fit(20)
    assert len(conv.messages) == 20
    assert conv.messages[0].content != "message 0"


def test_to_json():
    conv = Conversation()
    conv.add_message("user", "hello")
    data = conv.to_json()
    parsed = json.loads(data)
    assert parsed["messages"][0]["role"] == "user"


def test_from_json():
    data = json.dumps({"messages": [{"role": "user", "content": "test"}]})
    conv = Conversation.from_json(data)
    assert len(conv.messages) == 1


def test_message_token_count():
    msg = Message(role="user", content="hello world")
    assert msg.token_count > 0


def test_total_tokens():
    conv = Conversation()
    conv.add_message("user", "hello world")
    conv.add_message("assistant", "hi there")
    assert conv.total_tokens > 3
