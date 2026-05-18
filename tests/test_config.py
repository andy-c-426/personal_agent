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
    assert str(cfg.kb_dir).endswith("/.personal_agent/kb") or str(cfg.kb_dir).endswith("\\.personal_agent\\kb")
    assert cfg.max_tool_iterations == 10
    assert cfg.max_same_tool_calls == 3


def test_config_default_dir():
    import tempfile
    import os
    # Temporarily override HOME
    old_home = os.environ.get("HOME")
    try:
        os.environ["HOME"] = "/tmp/fake-home"
        os.environ["DEEPSEEK_API_KEY"] = "sk-test"
        os.environ["TAVILY_API_KEY"] = "tvly-test"
        cfg = Config.from_env()
        assert str(cfg.agent_dir) == "/tmp/fake-home/.personal_agent"
        assert str(cfg.chroma_dir) == "/tmp/fake-home/.personal_agent/chroma"
        assert str(cfg.kb_dir) == "/tmp/fake-home/.personal_agent/kb"
    finally:
        if old_home:
            os.environ["HOME"] = old_home
        else:
            del os.environ["HOME"]
