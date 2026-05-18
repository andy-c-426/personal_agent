import os
from personal_agent.config import Config


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    cfg = Config.from_env()
    assert cfg.deepseek_api_key == "sk-test"
    assert cfg.tavily_api_key == "tvly-test"


def test_config_defaults(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    cfg = Config.from_env()
    assert cfg.deepseek_model == "deepseek-chat"
    assert cfg.deepseek_base_url == "https://api.deepseek.com"
    assert cfg.max_tool_iterations == 10
    assert cfg.max_same_tool_calls == 3
    assert cfg.context_threshold_ratio == 0.8
    assert cfg.recent_message_count == 20
    assert cfg.kb_dir == cfg.agent_dir / "kb"


def test_config_default_dir(monkeypatch):
    monkeypatch.setenv("HOME", "/tmp/fake-home")
    monkeypatch.delenv("PERSONAL_AGENT_DIR", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    cfg = Config.from_env()
    assert str(cfg.agent_dir) == "/tmp/fake-home/.personal_agent"
    assert str(cfg.chroma_dir) == "/tmp/fake-home/.personal_agent/chroma"
    assert str(cfg.kb_dir) == "/tmp/fake-home/.personal_agent/kb"


def test_custom_agent_dir(monkeypatch):
    monkeypatch.setenv("PERSONAL_AGENT_DIR", "/custom/agent")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    cfg = Config.from_env()
    assert str(cfg.agent_dir) == "/custom/agent"
    assert str(cfg.chroma_dir) == "/custom/agent/chroma"
    assert str(cfg.kb_dir) == "/custom/agent/kb"
