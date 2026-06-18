from unittest.mock import MagicMock
from personal_agent.cli.app import _handle_slash_command, _setup_tools, AppContext
from personal_agent.kb.retrieval import KBMetadata
from personal_agent.tools.registry import ToolRegistry


class FakeConfig:
    deepseek_model = "deepseek-chat"
    deepseek_api_key = "sk-test"
    deepseek_base_url = "https://api.deepseek.com"
    agent_dir = MagicMock()
    chroma_dir = MagicMock()
    kb_dir = MagicMock()


def _make_ctx():
    retriever = MagicMock(spec=KBMetadata)
    agent = MagicMock()
    ctx = MagicMock(spec=AppContext)
    ctx.config = FakeConfig()
    ctx.retriever = retriever
    ctx.agent = agent
    return ctx


def test_handle_slash_help():
    ctx = _make_ctx()
    result = _handle_slash_command("help", "", ctx)
    assert result is True


def test_handle_slash_quit():
    ctx = _make_ctx()
    result = _handle_slash_command("quit", "", ctx)
    assert result is False


def test_handle_slash_exit():
    ctx = _make_ctx()
    result = _handle_slash_command("exit", "", ctx)
    assert result is False


def test_handle_slash_unknown():
    ctx = _make_ctx()
    result = _handle_slash_command("unknown", "", ctx)
    assert result is True


def test_setup_tools():
    retriever = MagicMock(spec=KBMetadata)
    tavily_mock = MagicMock()
    registry = _setup_tools(retriever, tavily_mock)
    assert isinstance(registry, ToolRegistry)
    schemas = registry.schemas()
    assert len(schemas) == 5
    names = {s["function"]["name"] for s in schemas}
    assert names == {"kb_search", "web_search", "kb_ingest", "kb_list", "kb_remove"}
