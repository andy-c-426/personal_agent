import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    deepseek_api_key: str
    tavily_api_key: str
    deepseek_model: str = "deepseek-chat"
    deepseek_base_url: str = "https://api.deepseek.com"
    agent_dir: Path = field(default_factory=lambda: Path.home() / ".personal_agent")
    kb_dir: Path | None = None
    max_tool_iterations: int = 10
    max_same_tool_calls: int = 3
    context_threshold_ratio: float = 0.8
    recent_message_count: int = 20

    @property
    def chroma_dir(self) -> Path:
        return self.agent_dir / "chroma"

    @classmethod
    def from_env(cls) -> "Config":
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        tavily_key = os.environ.get("TAVILY_API_KEY", "")
        agent_dir = Path(os.environ.get("PERSONAL_AGENT_DIR", Path.home() / ".personal_agent"))
        kb_dir = agent_dir / "kb"
        return cls(
            deepseek_api_key=api_key,
            tavily_api_key=tavily_key,
            agent_dir=agent_dir,
            kb_dir=kb_dir,
        )
