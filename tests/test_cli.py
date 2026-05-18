from unittest.mock import MagicMock, patch
from personal_agent.cli.app import _handle_slash_command, _setup_tools
from personal_agent.kb.retrieval import KBMetadata
from personal_agent.tools.registry import ToolRegistry


class FakeConfig:
    deepseek_model = "deepseek-chat"
    deepseek_api_key = "sk-test"
    deepseek_base_url = "https://api.deepseek.com"
    agent_dir = MagicMock()
    chroma_dir = MagicMock()
    kb_dir = MagicMock()


def test_handle_slash_help():
    retriever = MagicMock(spec=KBMetadata)
    result = _handle_slash_command("help", "", retriever, FakeConfig())
    assert result is True  # Continue


def test_handle_slash_quit():
    retriever = MagicMock(spec=KBMetadata)
    result = _handle_slash_command("quit", "", retriever, FakeConfig())
    assert result is False  # Exit


def test_handle_slash_exit():
    retriever = MagicMock(spec=KBMetadata)
    result = _handle_slash_command("exit", "", retriever, FakeConfig())
    assert result is False


def test_handle_slash_unknown():
    retriever = MagicMock(spec=KBMetadata)
    result = _handle_slash_command("unknown", "", retriever, FakeConfig())
    assert result is True  # Continue, shows error


def test_setup_tools():
    retriever = MagicMock(spec=KBMetadata)
    tavily_mock = MagicMock()
    registry = _setup_tools(retriever, tavily_mock)
    assert isinstance(registry, ToolRegistry)
    schemas = registry.schemas()
    assert len(schemas) == 5
    names = {s["function"]["name"] for s in schemas}
    assert names == {"kb_search", "web_search", "kb_ingest", "kb_list", "kb_remove"}
