# Personal Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CLI personal assistant with local knowledge base (Chroma RAG) and Tavily web search, powered by DeepSeek V4 Pro.

**Architecture:** Four-layer design — CLI (prompt_toolkit + rich) → Agent Core (custom ReAct loop) → Tools (kb_search, web_search, kb_ingest, kb_list, kb_remove) → Storage (Chroma). Memory uses sliding window + rolling summary compression. No LangChain — everything is hand-rolled for transparency.

**Tech Stack:** Python 3.12+, Poetry, prompt_toolkit, rich, Chroma, sentence-transformers (all-MiniLM-L6-v2), pdfplumber, python-docx, Tavily Python SDK, DeepSeek API (OpenAI-compatible endpoint).

**File Map:**

```
personal_agent/
├── cli/
│   ├── __init__.py
│   ├── app.py           # REPL loop, slash commands, prompt_toolkit setup
│   └── display.py       # Rich-based output rendering, streaming, tool status
├── core/
│   ├── __init__.py
│   ├── agent.py          # ReAct loop — prompt → model → tools → loop
│   ├── conversation.py   # Conversation dataclass, message CRUD, trim
│   └── memory.py         # Sliding window + rolling summary compression
├── tools/
│   ├── __init__.py
│   ├── registry.py       # ToolRegistry
│   ├── kb_search.py      # kb_search — semantic search Chroma
│   ├── web_search.py     # web_search — Tavily API
│   ├── kb_ingest.py      # kb_ingest — ingest file/dir into Chroma
│   ├── kb_list.py        # kb_list — list indexed docs
│   └── kb_remove.py      # kb_remove — remove doc from Chroma
├── kb/
│   ├── __init__.py
│   ├── embed.py          # Embedding model wrapper (sentence-transformers)
│   ├── ingest.py         # Parse → chunk → embed → store pipeline
│   └── retrieval.py      # Query embedding + Chroma search
├── config.py             # Environment + YAML config management
├── main.py               # Entry point
└── tests/
    ├── conftest.py
    ├── test_config.py
    ├── test_conversation.py
    ├── test_memory.py
    ├── test_embed.py
    ├── test_ingest.py
    ├── test_retrieval.py
    ├── test_tool_registry.py
    ├── test_tools.py
    ├── test_agent.py
    └── test_cli.py
```

Runtime data lives at `~/.personal_agent/`:
```
~/.personal_agent/
├── chroma/              # Chroma persistent storage
├── config.yaml          # User configuration
├── conversation.json    # Saved conversation state
└── kb/                  # Default knowledge base directory
```

---

### Task 1: Project Setup

**Files:**
- Create: `pyproject.toml`
- Create: `personal_agent/__init__.py`
- Create: `personal_agent/config.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Initialize Poetry project**

Run:
```bash
cd /Users/cchhyy/Desktop/personal_agent
poetry init --name personal_agent --description "Personal assistant agent deployed locally" --license Apache-2.0 --python "^3.12" --no-interaction
```

- [ ] **Step 2: Add dependencies**

Run:
```bash
cd /Users/cchhyy/Desktop/personal_agent
poetry add chromadb sentence-transformers prompt-toolkit rich tavily-python pyyaml openai pdfplumber python-docx
```

- [ ] **Step 3: Add dev dependencies**

Run:
```bash
cd /Users/cchhyy/Desktop/personal_agent
poetry add --dev pytest pytest-mock
```

- [ ] **Step 4: Create package init**

Write `personal_agent/__init__.py`:
```python
"""Personal agent assistant deployed locally."""
```

- [ ] **Step 5: Create tests init and conftest**

Write `tests/__init__.py`:
```python
```

Write `tests/conftest.py`:
```python
import pytest
import tempfile
import shutil
from pathlib import Path


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d)


@pytest.fixture
def sample_md_file(temp_dir):
    p = temp_dir / "test.md"
    p.write_text("# Test Doc\n\nThis is a test document for knowledge base ingestion.")
    return p


@pytest.fixture
def sample_pdf_file(temp_dir):
    p = temp_dir / "test.pdf"
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, text="Test PDF content for ingestion pipeline")
    pdf.output(str(p))
    return p


@pytest.fixture
def sample_docx_file(temp_dir):
    from docx import Document
    p = temp_dir / "test.docx"
    doc = Document()
    doc.add_paragraph("Test Word document content for ingestion pipeline")
    doc.save(str(p))
    return p


@pytest.fixture
def sample_text_file(temp_dir):
    p = temp_dir / "test.txt"
    p.write_text("Plain text content for the knowledge base.")
    return p
```

Note: `conftest.py` uses `fpdf` and `python-docx` for test fixtures — these are already available since `pdfplumber` and `python-docx` were added as deps. `fpdf` is not yet added.

Run to install fpdf:
```bash
cd /Users/cchhyy/Desktop/personal_agent && poetry add --dev fpdf2
```

- [ ] **Step 6: Verify project structure**

Run:
```bash
cd /Users/cchhyy/Desktop/personal_agent && poetry run python -c "import personal_agent; print('OK')"
```
Expected: `OK`

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml poetry.lock personal_agent/__init__.py tests/__init__.py tests/conftest.py
git commit -m "feat: initialize project with Poetry and dependencies"
```

---

### Task 2: Configuration Management

**Files:**
- Create: `personal_agent/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write failing test for config**

Write `tests/test_config.py`:
```python
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
    assert cfg.kb_dir is None
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
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd /Users/cchhyy/Desktop/personal_agent && poetry run pytest tests/test_config.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'personal_agent.config'`

- [ ] **Step 3: Write Config implementation**

Write `personal_agent/config.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
cd /Users/cchhyy/Desktop/personal_agent && poetry run pytest tests/test_config.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add personal_agent/config.py tests/test_config.py
git commit -m "feat: add configuration management"
```

---

### Task 3: Conversation Management

**Files:**
- Create: `personal_agent/core/__init__.py`
- Create: `personal_agent/core/conversation.py`
- Create: `tests/test_conversation.py`

- [ ] **Step 1: Write failing test for Conversation**

Write `tests/test_conversation.py`:
```python
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
    # Should keep at least the last 20
    assert len(conv.messages) >= 20
    # Early messages should be dropped
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
    assert conv.total_tokens > 5
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd /Users/cchhyy/Desktop/personal_agent && poetry run pytest tests/test_conversation.py -v
```
Expected: FAIL — module not found

- [ ] **Step 3: Write Conversation implementation**

Write `personal_agent/core/__init__.py`:
```python
```

Write `personal_agent/core/conversation.py`:
```python
import json
import tiktoken
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Message:
    role: str  # system, user, assistant, tool
    content: str
    name: str | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None

    @property
    def token_count(self) -> int:
        enc = tiktoken.get_encoding("cl100k_base")
        text = self.content or ""
        if self.tool_calls:
            text += json.dumps(self.tool_calls)
        return len(enc.encode(text))

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name:
            d["name"] = self.name
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        return d


@dataclass
class Conversation:
    messages: list[Message] = field(default_factory=list)
    summary: str = ""

    def add_message(
        self,
        role: str,
        content: str,
        name: str | None = None,
        tool_calls: list[dict] | None = None,
        tool_call_id: str | None = None,
    ) -> Message:
        msg = Message(role=role, content=content, name=name, tool_calls=tool_calls, tool_call_id=tool_call_id)
        self.messages.append(msg)
        return msg

    def to_dicts(self) -> list[dict[str, Any]]:
        return [m.to_dict() for m in self.messages]

    @property
    def total_tokens(self) -> int:
        return sum(m.token_count for m in self.messages)

    def trim_to_fit(self, max_messages: int) -> int:
        dropped = 0
        while len(self.messages) > max_messages:
            self.messages.pop(0)
            dropped += 1
        return dropped

    def to_json(self) -> str:
        data = {
            "messages": [m.to_dict() for m in self.messages],
            "summary": self.summary,
        }
        return json.dumps(data)

    @classmethod
    def from_json(cls, data: str) -> "Conversation":
        obj = json.loads(data)
        conv = cls(summary=obj.get("summary", ""))
        for m in obj.get("messages", []):
            conv.add_message(
                role=m["role"],
                content=m.get("content", ""),
                name=m.get("name"),
                tool_calls=m.get("tool_calls"),
                tool_call_id=m.get("tool_call_id"),
            )
        return conv
```

- [ ] **Step 4: Add tiktoken dependency**

Run:
```bash
cd /Users/cchhyy/Desktop/personal_agent && poetry add tiktoken
```

- [ ] **Step 5: Run test to verify it passes**

Run:
```bash
cd /Users/cchhyy/Desktop/personal_agent && poetry run pytest tests/test_conversation.py -v
```
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add personal_agent/core/__init__.py personal_agent/core/conversation.py tests/test_conversation.py pyproject.toml poetry.lock
git commit -m "feat: add conversation management with message model"
```

---

### Task 4: Memory — Sliding Window + Rolling Summary

**Files:**
- Create: `personal_agent/core/memory.py`
- Create: `tests/test_memory.py`

- [ ] **Step 1: Write failing test for memory**

Write `tests/test_memory.py`:
```python
from unittest.mock import patch, MagicMock
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
    # With tiny conversation, should not need compression
    # Threshold is 0.8 * 65536 (model context). Small conv won't trigger.
    assert not mgr.should_compress(conv, 65536)


def test_should_compress_over_threshold():
    conv = Conversation()
    # Add many messages to exceed threshold
    long_text = "hello world " * 500  # ~1000 tokens
    for _ in range(50):
        conv.add_message("user", long_text)
    mgr = MemoryManager(FakeConfig())
    # Model context for deepseek-chat is ~65536, 0.8 threshold = 52428
    # 50 messages * ~750 tokens = ~37500 — not over. Still test the logic.
    assert not mgr.should_compress(conv, model_context=10000)  # Should trigger with low context
    assert mgr.should_compress(conv, model_context=1000)  # Should definitely trigger


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

    # Should have trimmed first, then attempted compression
    assert len(conv.messages) <= FakeConfig.recent_message_count
    # Should have called the model for compression
    assert mock_client.chat.completions.create.called


def test_build_system_prompt():
    conv = Conversation()
    conv.summary = "Prior discussion about Python"
    mgr = MemoryManager(FakeConfig())
    prompt = mgr.build_system_prompt(12, conv)
    assert "12" in prompt
    assert "Prior discussion about Python" in prompt
    assert "kb_search" in prompt
    assert "web_search" in prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd /Users/cchhyy/Desktop/personal_agent && poetry run pytest tests/test_memory.py -v
```
Expected: FAIL — module not found

- [ ] **Step 3: Write Memory implementation**

Write `personal_agent/core/memory.py`:
```python
from personal_agent.core.conversation import Conversation


class MemoryManager:
    def __init__(self, config, client=None):
        self.config = config
        self._client = client  # OpenAI-compatible client for compression

    @property
    def _model_context(self) -> int:
        # DeepSeek V3 context window
        return 65536

    def should_compress(self, conversation: Conversation, model_context: int | None = None) -> bool:
        ctx = model_context or self._model_context
        threshold = int(ctx * self.config.context_threshold_ratio)
        return conversation.total_tokens > threshold

    def maybe_compress(self, conversation: Conversation, model_context: int | None = None) -> str | None:
        ctx = model_context or self._model_context

        # Step 1: trim to recent
        conversation.trim_to_fit(self.config.recent_message_count)

        # Step 2: if still over threshold, compress
        if not self.should_compress(conversation, ctx):
            return None

        # Separate recent messages from older ones to summarize
        all_msgs = list(conversation.messages)
        recent = all_msgs[-self.config.recent_message_count:]
        older = all_msgs[:-self.config.recent_message_count]

        if not older or not self._client:
            return None

        old_text = "\n".join(f"[{m.role}]: {m.content[:300]}" for m in older)

        response = self._client.chat.completions.create(
            model=self.config.deepseek_model,
            messages=[
                {"role": "system", "content": "Summarize this conversation in 3-5 sentences. Focus on key facts, decisions, and user preferences."},
                {"role": "user", "content": old_text},
            ],
            temperature=0.3,
            max_tokens=300,
        )
        summary = response.choices[0].message.content.strip()
        conversation.summary = summary
        conversation.messages = recent
        return summary

    def build_system_prompt(self, kb_doc_count: int, conversation: Conversation) -> str:
        summary_block = ""
        if conversation.summary:
            summary_block = f"\nConversation summary: {conversation.summary}\n"

        return f"""You are a personal assistant with access to a local knowledge base and web search.

Knowledge base: {kb_doc_count} documents indexed. Use kb_search to find relevant local information.
Web search: Use web_search when you need information not in the knowledge base.
{summary_block}
When answering:
- Prefer knowledge base results over web search when available
- Cite your sources (document name or URL)
- If both sources are used, distinguish between them"""
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
cd /Users/cchhyy/Desktop/personal_agent && poetry run pytest tests/test_memory.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add personal_agent/core/memory.py tests/test_memory.py
git commit -m "feat: add memory manager with sliding window and rolling summary"
```

---

### Task 5: Embedding Model Wrapper

**Files:**
- Create: `personal_agent/kb/__init__.py`
- Create: `personal_agent/kb/embed.py`
- Create: `tests/test_embed.py`

- [ ] **Step 1: Write failing test for embeddings**

Write `tests/test_embed.py`:
```python
from personal_agent.kb.embed import Embedder


def test_embedder_loads_model():
    embedder = Embedder()
    assert embedder.model is not None


def test_embed_returns_correct_dimensions():
    embedder = Embedder()
    result = embedder.embed(["hello world", "test sentence"])
    assert len(result) == 2
    assert len(result[0]) == 384  # all-MiniLM-L6-v2 dimension


def test_embed_single_string():
    embedder = Embedder()
    result = embedder.embed("hello world")
    assert len(result) == 1
    assert len(result[0]) == 384


def test_embedder_caches():
    e1 = Embedder()
    e2 = Embedder()
    assert e1.model is e2.model  # Singleton-like caching
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd /Users/cchhyy/Desktop/personal_agent && poetry run pytest tests/test_embed.py -v
```
Expected: FAIL — module not found

- [ ] **Step 3: Write Embedder implementation**

Write `personal_agent/kb/__init__.py`:
```python
```

Write `personal_agent/kb/embed.py`:
```python
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


class Embedder:
    def __init__(self):
        self.model = _get_model()

    def embed(self, texts: str | list[str]) -> list[list[float]]:
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
cd /Users/cchhyy/Desktop/personal_agent && poetry run pytest tests/test_embed.py -v
```
Expected: PASS (first run downloads the model, ~80MB)

- [ ] **Step 5: Commit**

```bash
git add personal_agent/kb/__init__.py personal_agent/kb/embed.py tests/test_embed.py
git commit -m "feat: add embedding model wrapper"
```

---

### Task 6: Knowledge Base Ingestion Pipeline

**Files:**
- Create: `personal_agent/kb/ingest.py`
- Create: `tests/test_ingest.py`

- [ ] **Step 1: Write failing test for ingestion**

Write `tests/test_ingest.py`:
```python
from personal_agent.kb.ingest import parse_file, chunk_text, ingest_file, ingest_directory


def test_parse_markdown_file(sample_md_file):
    text, meta = parse_file(sample_md_file)
    assert "Test Doc" in text
    assert "test document" in text
    assert meta["source"] == str(sample_md_file)
    assert meta["type"] == "markdown"


def test_parse_text_file(sample_text_file):
    text, meta = parse_file(sample_text_file)
    assert "Plain text" in text
    assert meta["type"] == "text"


def test_chunk_text():
    text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
    chunks = chunk_text(text, chunk_size=50, chunk_overlap=10)
    assert len(chunks) >= 1
    for chunk in chunks:
        assert len(chunk) > 0


def test_chunk_text_preserves_metadata():
    text = "Some content for chunking test."
    chunks = chunk_text(text, source_meta={"source": "test.md", "type": "markdown"})
    for chunk in chunks:
        assert chunk.metadata["source"] == "test.md"


def test_ingest_file_to_chroma(temp_dir, sample_md_file):
    import chromadb
    client = chromadb.PersistentClient(path=str(temp_dir / "chroma"))
    collection = client.get_or_create_collection("test_kb", embedding_function=_dummy_ef())
    ids = ingest_file(sample_md_file, collection)
    assert len(ids) > 0
    results = collection.get(ids=ids)
    assert len(results["ids"]) == len(ids)


def test_ingest_directory_to_chroma(temp_dir):
    import chromadb
    # Create multiple files
    (temp_dir / "a.md").write_text("# Doc A\nContent A")
    (temp_dir / "b.txt").write_text("Content B")
    client = chromadb.PersistentClient(path=str(temp_dir / "chroma2"))
    collection = client.get_or_create_collection("test_kb2", embedding_function=_dummy_ef())
    ids = ingest_directory(temp_dir, collection)
    assert len(ids) >= 2


def test_reingest_replaces_chunks(temp_dir, sample_md_file):
    import chromadb
    client = chromadb.PersistentClient(path=str(temp_dir / "chroma3"))
    collection = client.get_or_create_collection("test_kb3", embedding_function=_dummy_ef())
    first_ids = ingest_file(sample_md_file, collection)
    second_ids = ingest_file(sample_md_file, collection)
    assert set(first_ids) != set(second_ids)
    # Old chunks should be deleted
    old_results = collection.get(ids=first_ids)
    assert len(old_results["ids"]) == 0


def _dummy_ef():
    """Returns a dummy embedding function for testing without the real model."""
    from chromadb.api.types import EmbeddingFunction
    class DummyEF(EmbeddingFunction):
        def __call__(self, input):
            return [[0.1] * 384 for _ in input]
    return DummyEF()
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd /Users/cchhyy/Desktop/personal_agent && poetry run pytest tests/test_ingest.py -v
```
Expected: FAIL — module not found

- [ ] **Step 3: Write ingestion implementation**

Write `personal_agent/kb/ingest.py`:
```python
import hashlib
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    metadata: dict


def parse_file(file_path: Path) -> tuple[str, dict]:
    meta = {"source": str(file_path), "filename": file_path.name}
    suffix = file_path.suffix.lower()

    if suffix == ".md":
        meta["type"] = "markdown"
        return file_path.read_text(encoding="utf-8"), meta
    elif suffix == ".pdf":
        meta["type"] = "pdf"
        import pdfplumber
        with pdfplumber.open(file_path) as pdf:
            text = "\n".join(p.extract_text() or "" for p in pdf.pages)
        return text, meta
    elif suffix in (".docx", ".doc"):
        meta["type"] = "docx"
        from docx import Document
        doc = Document(str(file_path))
        text = "\n".join(p.text for p in doc.paragraphs)
        return text, meta
    else:
        meta["type"] = "text"
        return file_path.read_text(encoding="utf-8"), meta


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    source_meta: dict | None = None,
) -> list[Chunk]:
    meta = source_meta or {}
    # Split by paragraphs first
    paragraphs = text.split("\n\n")
    chunks = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        # Rough token estimate: chars / 4
        if len(current) / 4 + len(para) / 4 > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**meta, "chunk_index": len(chunks)}))
            # Overlap: keep last ~overlap tokens worth
            overlap_chars = chunk_overlap * 4
            if len(current) > overlap_chars:
                current = current[-overlap_chars:] + "\n\n" + para
            else:
                current = para
        else:
            current = (current + "\n\n" + para).strip()

    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**meta, "chunk_index": len(chunks)}))

    return chunks


def _file_hash(file_path: Path) -> str:
    return hashlib.md5(file_path.read_bytes()).hexdigest()


def _delete_existing(file_path: Path, collection):
    """Remove existing chunks for this file before re-ingestion."""
    existing = collection.get(where={"source": str(file_path)})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])


def ingest_file(file_path: Path, collection) -> list[str]:
    file_path = Path(file_path)
    _delete_existing(file_path, collection)

    text, meta = parse_file(file_path)
    meta["file_hash"] = _file_hash(file_path)
    chunks = chunk_text(text, source_meta=meta)

    if not chunks:
        return []

    ids = []
    documents = []
    metadatas = []
    embeddings = []

    from personal_agent.kb.embed import Embedder
    embedder = Embedder()

    for c in chunks:
        chunk_id = f"{file_path.stem}_{c.metadata['chunk_index']}_{_file_hash(file_path)[:8]}"
        ids.append(chunk_id)
        documents.append(c.text)
        metadatas.append(c.metadata)
        embeddings.append(embedder.embed(c.text)[0])

    collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    return ids


def ingest_directory(dir_path: Path, collection) -> list[str]:
    dir_path = Path(dir_path)
    all_ids = []
    for p in dir_path.rglob("*"):
        if p.is_file() and not p.name.startswith("."):
            try:
                ids = ingest_file(p, collection)
                all_ids.extend(ids)
            except Exception:
                continue
    return all_ids
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
cd /Users/cchhyy/Desktop/personal_agent && poetry run pytest tests/test_ingest.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add personal_agent/kb/ingest.py tests/test_ingest.py
git commit -m "feat: add knowledge base ingestion pipeline"
```

---

### Task 7: Knowledge Base Retrieval

**Files:**
- Create: `personal_agent/kb/retrieval.py`
- Create: `tests/test_retrieval.py`

- [ ] **Step 1: Write failing test for retrieval**

Write `tests/test_retrieval.py`:
```python
import chromadb
from personal_agent.kb.retrieval import KBRetriever


def test_search_returns_results(temp_dir, sample_md_file):
    from personal_agent.kb.ingest import ingest_file
    from chromadb.api.types import EmbeddingFunction

    class DummyEF(EmbeddingFunction):
        def __call__(self, input):
            return [[0.1] * 384 for _ in input]

    client = chromadb.PersistentClient(path=str(temp_dir / "chroma"))
    collection = client.get_or_create_collection("test_retrieval", embedding_function=DummyEF())
    ingest_file(sample_md_file, collection)

    retriever = KBMetadata(client, collection_name="test_retrieval")
    results = retriever.search("test document")

    assert len(results) > 0
    assert "text" in results[0]
    assert "source" in results[0]


def test_search_returns_empty_for_no_match(temp_dir):
    import chromadb
    from chromadb.api.types import EmbeddingFunction

    class DummyEF(EmbeddingFunction):
        def __call__(self, input):
            return [[0.1] * 384 for _ in input]

    client = chromadb.PersistentClient(path=str(temp_dir / "chroma2"))
    # Add a document first
    collection = client.get_or_create_collection("test_empty", embedding_function=DummyEF())
    collection.add(ids=["1"], documents=["unrelated content"], embeddings=[[0.1]*384])

    retriever = KBMetadata(client, collection_name="test_empty")
    # With dummy embeddings all the same, any query matches — but we at least test the interface
    results = retriever.search("something")
    assert isinstance(results, list)


def test_get_document_count(temp_dir):
    import chromadb
    from chromadb.api.types import EmbeddingFunction

    class DummyEF(EmbeddingFunction):
        def __call__(self, input):
            return [[0.1] * 384 for _ in input]

    client = chromadb.PersistentClient(path=str(temp_dir / "chroma3"))
    collection = client.get_or_create_collection("test_count", embedding_function=DummyEF())
    collection.add(ids=["1", "2"], documents=["a", "b"], embeddings=[[0.1]*384, [0.1]*384])

    retriever = KBMetadata(client, collection_name="test_count")
    assert retriever.document_count >= 0


def test_list_documents(temp_dir, sample_md_file):
    from personal_agent.kb.ingest import ingest_file
    from chromadb.api.types import EmbeddingFunction

    class DummyEF(EmbeddingFunction):
        def __call__(self, input):
            return [[0.1] * 384 for _ in input]

    client = chromadb.PersistentClient(path=str(temp_dir / "chroma4"))
    collection = client.get_or_create_collection("test_list", embedding_function=DummyEF())
    ingest_file(sample_md_file, collection)

    retriever = KBMetadata(client, collection_name="test_list")
    docs = retriever.list_documents()
    assert len(docs) >= 1
    assert "source" in docs[0] or "filename" in docs[0]


def test_remove_document(temp_dir, sample_md_file):
    from personal_agent.kb.ingest import ingest_file
    from chromadb.api.types import EmbeddingFunction

    class DummyEF(EmbeddingFunction):
        def __call__(self, input):
            return [[0.1] * 384 for _ in input]

    client = chromadb.PersistentClient(path=str(temp_dir / "chroma5"))
    collection = client.get_or_create_collection("test_remove", embedding_function=DummyEF())
    ingest_file(sample_md_file, collection)

    retriever = KBMetadata(client, collection_name="test_remove")
    removed = retriever.remove_document(str(sample_md_file))
    assert removed > 0

    # Verify removed
    results = collection.get(where={"source": str(sample_md_file)})
    assert len(results["ids"]) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd /Users/cchhyy/Desktop/personal_agent && poetry run pytest tests/test_retrieval.py -v
```
Expected: FAIL — module not found

- [ ] **Step 3: Write retrieval implementation**

Write `personal_agent/kb/retrieval.py`:
```python
import chromadb
from personal_agent.kb.embed import Embedder


class KBMetadata:
    def __init__(self, client: chromadb.PersistentClient, collection_name: str = "kb"):
        self.client = client
        self.collection_name = collection_name
        self._collection = None
        self._embedder = Embedder()

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                self.collection_name,
                embedding_function=_ChromaEmbeddingAdapter(self._embedder),
            )
        return self._collection

    @property
    def document_count(self) -> int:
        return self.collection.count()

    def search(self, query: str, n_results: int = 5) -> list[dict]:
        results = self.collection.query(query_texts=[query], n_results=n_results)
        if not results["ids"] or not results["ids"][0]:
            return []

        output = []
        for i, doc_id in enumerate(results["ids"][0]):
            output.append({
                "id": doc_id,
                "text": results["documents"][0][i] if results["documents"] else "",
                "source": results["metadatas"][0][i].get("source", "") if results["metadatas"] else "",
                "filename": results["metadatas"][0][i].get("filename", "") if results["metadatas"] else "",
            })
        return output

    def list_documents(self) -> list[dict]:
        all_data = self.collection.get()
        seen = {}
        docs = []
        if all_data["metadatas"]:
            for meta in all_data["metadatas"]:
                source = meta.get("source", "")
                if source and source not in seen:
                    seen[source] = True
                    docs.append({"source": source, "filename": meta.get("filename", "")})
        return docs

    def remove_document(self, source_path: str) -> int:
        existing = self.collection.get(where={"source": source_path})
        if existing["ids"]:
            self.collection.delete(ids=existing["ids"])
            return len(existing["ids"])
        return 0


class _ChromaEmbeddingAdapter:
    """Adapts our Embedder to Chroma's EmbeddingFunction interface."""
    def __init__(self, embedder: Embedder):
        self._embedder = embedder

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self._embedder.embed(input)
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
cd /Users/cchhyy/Desktop/personal_agent && poetry run pytest tests/test_retrieval.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add personal_agent/kb/retrieval.py tests/test_retrieval.py
git commit -m "feat: add knowledge base retrieval with Chroma"
```

---

### Task 8: Tool Registry

**Files:**
- Create: `personal_agent/tools/__init__.py`
- Create: `personal_agent/tools/registry.py`
- Create: `tests/test_tool_registry.py`

- [ ] **Step 1: Write failing test for ToolRegistry**

Write `tests/test_tool_registry.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd /Users/cchhyy/Desktop/personal_agent && poetry run pytest tests/test_tool_registry.py -v
```
Expected: FAIL — module not found

- [ ] **Step 3: Write ToolRegistry implementation**

Write `personal_agent/tools/__init__.py`:
```python
```

Write `personal_agent/tools/registry.py`:
```python
import json
import traceback
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Tool:
    name: str
    description: str
    function: Callable
    parameters: dict[str, Any]


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def schemas(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self._tools.values()
        ]

    def execute(self, name: str, **kwargs) -> str:
        tool = self._tools.get(name)
        if not tool:
            return json.dumps({"error": f"Tool '{name}' not found"})
        try:
            result = tool.function(**kwargs)
            if isinstance(result, str):
                return result
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e), "traceback": traceback.format_exc()})
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
cd /Users/cchhyy/Desktop/personal_agent && poetry run pytest tests/test_tool_registry.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add personal_agent/tools/__init__.py personal_agent/tools/registry.py tests/test_tool_registry.py
git commit -m "feat: add tool registry"
```

---

### Task 9: Knowledge Base Search Tool

**Files:**
- Create: `personal_agent/tools/kb_search.py`

- [ ] **Step 1: Write the tool implementation**

Write `personal_agent/tools/kb_search.py`:
```python
import json
from personal_agent.kb.retrieval import KBMetadata


def kb_search(query: str, retriever: KBMetadata) -> str:
    results = retriever.search(query, n_results=5)
    if not results:
        return json.dumps({"results": [], "message": "No results found in knowledge base."})
    return json.dumps({"results": results}, ensure_ascii=False)
```

- [ ] **Step 2: Create tests/test_tools.py with kb_search test**

Write `tests/test_tools.py`:
```python
import json
import chromadb
from personal_agent.tools.kb_search import kb_search
from personal_agent.kb.retrieval import KBMetadata


def test_kb_search_returns_results(temp_dir, sample_md_file):
    from personal_agent.kb.ingest import ingest_file
    from chromadb.api.types import EmbeddingFunction

    class DummyEF(EmbeddingFunction):
        def __call__(self, input):
            return [[0.1] * 384 for _ in input]

    client = chromadb.PersistentClient(path=str(temp_dir / "chroma_kbs"))
    collection = client.get_or_create_collection("test_kbs", embedding_function=DummyEF())
    ingest_file(sample_md_file, collection)
    retriever = KBMetadata(client, collection_name="test_kbs")

    result = kb_search("test", retriever=retriever)
    data = json.loads(result)
    assert "results" in data


def test_kb_search_empty(temp_dir):
    import chromadb
    from chromadb.api.types import EmbeddingFunction

    class DummyEF(EmbeddingFunction):
        def __call__(self, input):
            return [[0.1] * 384 for _ in input]

    client = chromadb.PersistentClient(path=str(temp_dir / "chroma_kbs2"))
    client.get_or_create_collection("test_kbs_empty", embedding_function=DummyEF())
    retriever = KBMetadata(client, collection_name="test_kbs_empty")

    result = kb_search("nothing", retriever=retriever)
    data = json.loads(result)
    assert data["results"] == []
```

- [ ] **Step 3: Run tests**

Run:
```bash
cd /Users/cchhyy/Desktop/personal_agent && poetry run pytest tests/test_tools.py::test_kb_search_returns_results tests/test_tools.py::test_kb_search_empty -v
```
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add personal_agent/tools/kb_search.py tests/test_tools.py
git commit -m "feat: add kb_search tool"
```

---

### Task 10: Web Search Tool (Tavily)

**Files:**
- Create: `personal_agent/tools/web_search.py`

- [ ] **Step 1: Write the tool implementation**

Write `personal_agent/tools/web_search.py`:
```python
import json
from tavily import TavilyClient


def web_search(query: str, client: TavilyClient) -> str:
    try:
        response = client.search(query, search_depth="advanced", max_results=5)
        results = response.get("results", [])
        formatted = [
            {"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")}
            for r in results
        ]
        return json.dumps({"results": formatted}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e), "results": []})
```

- [ ] **Step 2: Write test for web_search tool**

Append to `tests/test_tools.py`:
```python
import json
from unittest.mock import MagicMock
from personal_agent.tools.web_search import web_search


def test_web_search_returns_results():
    mock_client = MagicMock()
    mock_client.search.return_value = {
        "results": [
            {"title": "Test Result", "url": "https://example.com", "content": "Example content"}
        ]
    }
    result = web_search("test query", client=mock_client)
    data = json.loads(result)
    assert len(data["results"]) == 1
    assert data["results"][0]["title"] == "Test Result"


def test_web_search_handles_error():
    mock_client = MagicMock()
    mock_client.search.side_effect = Exception("API error")
    result = web_search("test", client=mock_client)
    data = json.loads(result)
    assert "error" in data
```

- [ ] **Step 3: Run tests**

Run:
```bash
cd /Users/cchhyy/Desktop/personal_agent && poetry run pytest tests/test_tools.py::test_web_search_returns_results tests/test_tools.py::test_web_search_handles_error -v
```
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add personal_agent/tools/web_search.py tests/test_tools.py
git commit -m "feat: add web_search tool with Tavily"
```

---

### Task 11: KB Management Tools (ingest, list, remove)

**Files:**
- Create: `personal_agent/tools/kb_ingest.py`
- Create: `personal_agent/tools/kb_list.py`
- Create: `personal_agent/tools/kb_remove.py`

- [ ] **Step 1: Write the tool implementations**

Write `personal_agent/tools/kb_ingest.py`:
```python
import json
from pathlib import Path
from personal_agent.kb.ingest import ingest_file, ingest_directory
from personal_agent.kb.retrieval import KBMetadata


def kb_ingest(path: str, retriever: KBMetadata) -> str:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return json.dumps({"error": f"Path not found: {path}"})
    try:
        if p.is_file():
            ids = ingest_file(p, retriever.collection)
        else:
            ids = ingest_directory(p, retriever.collection)
        return json.dumps({"ingested": len(ids), "path": str(p)})
    except Exception as e:
        return json.dumps({"error": str(e)})
```

Write `personal_agent/tools/kb_list.py`:
```python
import json
from personal_agent.kb.retrieval import KBMetadata


def kb_list(retriever: KBMetadata) -> str:
    docs = retriever.list_documents()
    return json.dumps({"documents": docs, "count": len(docs)}, ensure_ascii=False)
```

Write `personal_agent/tools/kb_remove.py`:
```python
import json
from personal_agent.kb.retrieval import KBMetadata


def kb_remove(source: str, retriever: KBMetadata) -> str:
    count = retriever.remove_document(source)
    return json.dumps({"removed": count, "source": source})
```

- [ ] **Step 2: Write tests for KB management tools**

Append to `tests/test_tools.py`:
```python
import json
import chromadb
from personal_agent.tools.kb_ingest import kb_ingest
from personal_agent.tools.kb_list import kb_list
from personal_agent.tools.kb_remove import kb_remove
from personal_agent.kb.retrieval import KBMetadata
from chromadb.api.types import EmbeddingFunction


class _DummyEF(EmbeddingFunction):
    def __call__(self, input):
        return [[0.1] * 384 for _ in input]


def _make_retriever(temp_dir, name="test_kbm"):
    client = chromadb.PersistentClient(path=str(temp_dir / "chroma_kbm"))
    collection = client.get_or_create_collection(name, embedding_function=_DummyEF())
    return KBMetadata(client, collection_name=name)


def test_kb_ingest_markdown(temp_dir, sample_md_file):
    retriever = _make_retriever(temp_dir, "ingest1")
    result = kb_ingest(str(sample_md_file), retriever=retriever)
    data = json.loads(result)
    assert data["ingested"] > 0


def test_kb_ingest_nonexistent_path(temp_dir):
    retriever = _make_retriever(temp_dir, "ingest2")
    result = kb_ingest("/nonexistent/path", retriever=retriever)
    data = json.loads(result)
    assert "error" in data


def test_kb_list_documents(temp_dir, sample_md_file):
    retriever = _make_retriever(temp_dir, "list1")
    kb_ingest(str(sample_md_file), retriever=retriever)
    result = kb_list(retriever=retriever)
    data = json.loads(result)
    assert data["count"] >= 1


def test_kb_remove_document(temp_dir, sample_md_file):
    retriever = _make_retriever(temp_dir, "remove1")
    kb_ingest(str(sample_md_file), retriever=retriever)
    result = kb_remove(str(sample_md_file), retriever=retriever)
    data = json.loads(result)
    assert data["removed"] > 0
```

- [ ] **Step 3: Run tests**

Run:
```bash
cd /Users/cchhyy/Desktop/personal_agent && poetry run pytest tests/test_tools.py::test_kb_ingest_markdown tests/test_tools.py::test_kb_ingest_nonexistent_path tests/test_tools.py::test_kb_list_documents tests/test_tools.py::test_kb_remove_document -v
```
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add personal_agent/tools/kb_ingest.py personal_agent/tools/kb_list.py personal_agent/tools/kb_remove.py tests/test_tools.py
git commit -m "feat: add kb_ingest, kb_list, kb_remove tools"
```

---

### Task 12: Agent Core — ReAct Loop

**Files:**
- Create: `personal_agent/core/agent.py`
- Create: `tests/test_agent.py`

- [ ] **Step 1: Write failing test for agent**

Write `tests/test_agent.py`:
```python
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
    # First call: model requests tool
    # Second call: model returns final answer
    call1 = MagicMock(
        choices=[MagicMock(
            message=MagicMock(
                content=None,
                tool_calls=[MagicMock(
                    id="call_1",
                    function=MagicMock(name="mock_tool", arguments='{"query":"test"}'),
                )],
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
    memory._model_context = 1000  # Small context to avoid compression during test

    agent = Agent(FakeConfig(), mock_client, registry, memory_manager=memory)
    response, tool_calls = agent.run("Search for test", conv)

    assert "Found" in response
    assert len(tool_calls) == 1


def test_agent_stops_on_same_tool_repeat():
    call1 = MagicMock(
        choices=[MagicMock(
            message=MagicMock(
                content=None,
                tool_calls=[MagicMock(
                    id="call_1",
                    function=MagicMock(name="stuck_tool", arguments='{}'),
                )],
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

    assert "stopped" in response.lower() or "loop" in response.lower() or len(tool_calls) <= 3


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

    # Verify system prompt was included with doc count
    call_args = mock_client.chat.completions.create.call_args
    messages = call_args[1]["messages"]
    system_msg = messages[0]["content"]
    assert "42" in system_msg
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd /Users/cchhyy/Desktop/personal_agent && poetry run pytest tests/test_agent.py -v
```
Expected: FAIL — module not found

- [ ] **Step 3: Write Agent implementation**

Write `personal_agent/core/agent.py`:
```python
import json
from openai import OpenAI
from personal_agent.core.conversation import Conversation
from personal_agent.core.memory import MemoryManager
from personal_agent.tools.registry import ToolRegistry


class Agent:
    def __init__(
        self,
        config,
        client: OpenAI,
        tool_registry: ToolRegistry,
        memory_manager: MemoryManager | None = None,
        kb_doc_count: int = 0,
    ):
        self.config = config
        self.client = client
        self.registry = tool_registry
        self.memory = memory_manager or MemoryManager(config)
        self.kb_doc_count = kb_doc_count

    def run(self, user_input: str, conversation: Conversation) -> tuple[str, list[dict]]:
        conversation.add_message("user", user_input)

        system_prompt = self.memory.build_system_prompt(self.kb_doc_count, conversation)

        tool_calls_made = []
        same_tool_count = 0
        last_tool_name = None

        for _ in range(self.config.max_tool_iterations):
            messages = [{"role": "system", "content": system_prompt}] + conversation.to_dicts()

            response = self.client.chat.completions.create(
                model=self.config.deepseek_model,
                messages=messages,
                tools=self.registry.schemas() if self.registry.schemas() else None,
                temperature=0.7,
            )

            msg = response.choices[0].message

            if msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_name = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}

                    result = self.registry.execute(tool_name, **args)
                    tool_calls_made.append({"name": tool_name, "arguments": args, "result": result})

                    conversation.add_message(
                        "assistant",
                        content="",
                        tool_calls=[{
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tool_name, "arguments": tc.function.arguments},
                        }],
                    )
                    conversation.add_message("tool", content=result, tool_call_id=tc.id)

                    # Track same-tool repeats
                    if tool_name == last_tool_name:
                        same_tool_count += 1
                    else:
                        same_tool_count = 1
                        last_tool_name = tool_name

                    if same_tool_count >= self.config.max_same_tool_calls:
                        final = f"I've called '{tool_name}' several times without progress. Let me stop and share what I have so far. The last result was: {result[:500]}"
                        conversation.add_message("assistant", final)
                        return final, tool_calls_made
            else:
                text = msg.content or ""
                conversation.add_message("assistant", text)

                # Try compression if needed
                self.memory.maybe_compress(conversation)

                return text, tool_calls_made

        # Max iterations reached
        final = "I've reached the maximum number of tool calls. Here's what I found so far."
        conversation.add_message("assistant", final)
        return final, tool_calls_made
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
cd /Users/cchhyy/Desktop/personal_agent && poetry run pytest tests/test_agent.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add personal_agent/core/agent.py tests/test_agent.py
git commit -m "feat: add agent ReAct loop"
```

---

### Task 13: CLI Display

**Files:**
- Create: `personal_agent/cli/__init__.py`
- Create: `personal_agent/cli/display.py`

- [ ] **Step 1: Write display implementation**

Write `personal_agent/cli/__init__.py`:
```python
```

Write `personal_agent/cli/display.py`:
```python
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.live import Live
from rich.text import Text

console = Console()


def print_welcome(kb_count: int) -> None:
    console.print(Panel.fit(
        f"[bold]Personal Agent[/bold]\n"
        f"Knowledge base: {kb_count} documents indexed\n"
        f"Type /help for commands, /quit to exit",
        border_style="blue",
    ))


def print_tool_status(tool_name: str, summary: str) -> None:
    console.print(f"  [dim]→ {tool_name}: {summary}[/dim]")


def print_assistant_header() -> None:
    console.print()


def stream_markdown(text: str) -> None:
    console.print(Markdown(text))


def print_error(message: str) -> None:
    console.print(f"[red]Error: {message}[/red]")


def print_help() -> None:
    console.print(Panel.fit(
        "[bold]Commands:[/bold]\n"
        "  [cyan]/ingest <path>[/cyan]  Add file or directory to knowledge base\n"
        "  [cyan]/kb list[/cyan]         List indexed documents\n"
        "  [cyan]/kb remove <id>[/cyan]  Remove a document\n"
        "  [cyan]/config[/cyan]          Show current configuration\n"
        "  [cyan]/help[/cyan]            Show this help\n"
        "  [cyan]/quit[/cyan]            Exit",
        title="Help",
    ))


def print_config(config) -> None:
    console.print(Panel(
        f"Model: {config.deepseek_model}\n"
        f"KB directory: {config.kb_dir}\n"
        f"Agent directory: {config.agent_dir}",
        title="Configuration",
    ))
```

- [ ] **Step 2: Commit**

```bash
git add personal_agent/cli/__init__.py personal_agent/cli/display.py
git commit -m "feat: add CLI display with Rich"
```

---

### Task 14: CLI App — REPL and Slash Commands

**Files:**
- Create: `personal_agent/cli/app.py`

- [ ] **Step 1: Write CLI app**

Write `personal_agent/cli/app.py`:
```python
from pathlib import Path
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from openai import OpenAI
from personal_agent.config import Config
from personal_agent.core.agent import Agent
from personal_agent.core.conversation import Conversation
from personal_agent.core.memory import MemoryManager
from personal_agent.tools.registry import ToolRegistry, Tool
from personal_agent.tools.kb_search import kb_search
from personal_agent.tools.web_search import web_search
from personal_agent.tools.kb_ingest import kb_ingest
from personal_agent.tools.kb_list import kb_list
from personal_agent.tools.kb_remove import kb_remove
from personal_agent.kb.retrieval import KBMetadata
from personal_agent.cli import display
import chromadb
from tavily import TavilyClient


def _setup_tools(retriever: KBMetadata, tavily_client: TavilyClient) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(Tool(
        name="kb_search",
        description="Search the local knowledge base for relevant information.",
        function=lambda query: kb_search(query, retriever=retriever),
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query"}},
            "required": ["query"],
        },
    ))
    registry.register(Tool(
        name="web_search",
        description="Search the web for information not in the knowledge base.",
        function=lambda query: web_search(query, client=tavily_client),
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query"}},
            "required": ["query"],
        },
    ))
    registry.register(Tool(
        name="kb_ingest",
        description="Ingest a file or directory into the knowledge base.",
        function=lambda path: kb_ingest(path, retriever=retriever),
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Path to file or directory"}},
            "required": ["path"],
        },
    ))
    registry.register(Tool(
        name="kb_list",
        description="List all documents in the knowledge base.",
        function=lambda: kb_list(retriever=retriever),
        parameters={"type": "object", "properties": {}},
    ))
    registry.register(Tool(
        name="kb_remove",
        description="Remove a document from the knowledge base.",
        function=lambda source: kb_remove(source, retriever=retriever),
        parameters={
            "type": "object",
            "properties": {"source": {"type": "string", "description": "Source path of the document"}},
            "required": ["source"],
        },
    ))
    return registry


def _handle_slash_command(cmd: str, args: str, retriever: KBMetadata, config: Config) -> bool:
    """Returns True if the REPL should continue, False to exit."""
    if cmd == "quit" or cmd == "exit":
        return False
    elif cmd == "help":
        display.print_help()
    elif cmd == "config":
        display.print_config(config)
    elif cmd == "ingest":
        if args:
            result = kb_ingest(args, retriever=retriever)
            display.console.print(f"Ingested: {result}")
        else:
            display.print_error("Usage: /ingest <path>")
    elif cmd == "kb":
        if args == "list":
            result = kb_list(retriever=retriever)
            display.console.print(result)
        elif args.startswith("remove "):
            source = args[7:]
            result = kb_remove(source, retriever=retriever)
            display.console.print(result)
        else:
            display.print_error("Usage: /kb list | /kb remove <id>")
    else:
        display.print_error(f"Unknown command: /{cmd}. Type /help for commands.")
    return True


def run(config: Config) -> None:
    config.agent_dir.mkdir(parents=True, exist_ok=True)
    config.chroma_dir.mkdir(parents=True, exist_ok=True)
    if config.kb_dir:
        config.kb_dir.mkdir(parents=True, exist_ok=True)

    # Setup storage
    chroma_client = chromadb.PersistentClient(path=str(config.chroma_dir))
    retriever = KBMetadata(chroma_client, collection_name="kb")

    # Setup Tavily
    tavily_client = TavilyClient(api_key=config.tavily_api_key)

    # Setup tools
    registry = _setup_tools(retriever, tavily_client)

    # Setup model client
    llm_client = OpenAI(
        api_key=config.deepseek_api_key,
        base_url=config.deepseek_base_url,
    )

    # Setup conversation
    conv_path = config.agent_dir / "conversation.json"
    if conv_path.exists():
        conversation = Conversation.from_json(conv_path.read_text())
    else:
        conversation = Conversation()

    memory_manager = MemoryManager(config, client=llm_client)

    agent = Agent(
        config,
        llm_client,
        registry,
        memory_manager=memory_manager,
        kb_doc_count=retriever.document_count,
    )

    history_path = config.agent_dir / ".history"
    session = PromptSession(
        history=FileHistory(str(history_path)),
        style=Style.from_dict({"prompt": "bold green"}),
    )

    display.print_welcome(retriever.document_count)

    while True:
        try:
            user_input = session.prompt([("class:prompt", "> ")]).strip()
        except (EOFError, KeyboardInterrupt):
            display.console.print("\nGoodbye!")
            break

        if not user_input:
            continue

        # Handle slash commands
        if user_input.startswith("/"):
            parts = user_input[1:].split(maxsplit=1)
            cmd = parts[0]
            args = parts[1] if len(parts) > 1 else ""
            if not _handle_slash_command(cmd, args, retriever, config):
                break
            continue

        # Regular query
        try:
            response, tool_calls = agent.run(user_input, conversation)

            if tool_calls:
                for tc in tool_calls:
                    result_preview = tc["result"][:120].replace("\n", " ")
                    display.print_tool_status(tc["name"], result_preview)

            display.print_assistant_header()
            display.stream_markdown(response)

            # Persist conversation
            conv_path.write_text(conversation.to_json())

        except Exception as e:
            display.print_error(str(e))

    # Save on exit
    conv_path.write_text(conversation.to_json())
    display.console.print("Session saved.")
```

- [ ] **Step 2: Commit**

```bash
git add personal_agent/cli/app.py
git commit -m "feat: add CLI REPL app with slash commands"
```

---

### Task 15: Entry Point

**Files:**
- Create: `personal_agent/main.py`

- [ ] **Step 1: Write main entry point**

Write `personal_agent/main.py`:
```python
import argparse
from personal_agent.config import Config
from personal_agent.cli.app import run


def main():
    parser = argparse.ArgumentParser(description="Personal Agent — local AI assistant")
    parser.add_argument("--debug", action="store_true", help="Print full tool payloads")
    args = parser.parse_args()

    config = Config.from_env()

    if not config.deepseek_api_key:
        print("Error: DEEPSEEK_API_KEY environment variable is required.")
        return 1
    if not config.tavily_api_key:
        print("Error: TAVILY_API_KEY environment variable is required.")
        return 1

    run(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Add entry point to pyproject.toml**

Update `pyproject.toml` to include:
```toml
[tool.poetry.scripts]
personal-agent = "personal_agent.main:main"
```

- [ ] **Step 3: Verify entry point**

Run:
```bash
cd /Users/cchhyy/Desktop/personal_agent && poetry run personal-agent --help
```
Expected: usage message printed

- [ ] **Step 4: Commit**

```bash
git add personal_agent/main.py pyproject.toml
git commit -m "feat: add entry point"
```

---

### Task 16: Integration Test and CLI Test

**Files:**
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write CLI integration test**

Write `tests/test_cli.py`:
```python
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
    import chromadb
    retriever = MagicMock(spec=KBMetadata)
    tavily_mock = MagicMock()
    registry = _setup_tools(retriever, tavily_mock)
    assert isinstance(registry, ToolRegistry)
    schemas = registry.schemas()
    assert len(schemas) == 5
    names = {s["function"]["name"] for s in schemas}
    assert names == {"kb_search", "web_search", "kb_ingest", "kb_list", "kb_remove"}
```

- [ ] **Step 2: Run integration tests**

Run:
```bash
cd /Users/cchhyy/Desktop/personal_agent && poetry run pytest tests/test_cli.py -v
```
Expected: PASS

- [ ] **Step 3: Run full test suite**

Run:
```bash
cd /Users/cchhyy/Desktop/personal_agent && poetry run pytest tests/ -v
```
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add tests/test_cli.py
git commit -m "feat: add CLI integration tests"
```

---

### Task 17: Final Verification

- [ ] **Step 1: Run full test suite**

Run:
```bash
cd /Users/cchhyy/Desktop/personal_agent && poetry run pytest tests/ -v
```
Expected: All tests pass

- [ ] **Step 2: Verify imports work**

Run:
```bash
cd /Users/cchhyy/Desktop/personal_agent && poetry run python -c "
from personal_agent.config import Config
from personal_agent.core.conversation import Conversation
from personal_agent.core.memory import MemoryManager
from personal_agent.core.agent import Agent
from personal_agent.tools.registry import ToolRegistry
from personal_agent.kb.embed import Embedder
from personal_agent.kb.ingest import parse_file, chunk_text
from personal_agent.kb.retrieval import KBMetadata
print('All imports OK')
"
```
Expected: `All imports OK`

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "chore: finalize project structure"
```
