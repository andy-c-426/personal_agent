# RAG Pipeline Improvement — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the knowledge base RAG pipeline from basic semantic search to a production-quality hybrid system: BGE-M3 embeddings (dense + sparse), semantic chunking, hybrid retrieval with cross-encoder reranking, and LLM query rewriting.

**Architecture:** Swap `all-MiniLM-L6-v2` for `BAAI/bge-m3` (FlagEmbedding) providing both dense 1024-dim vectors and sparse lexical weights. Ingest path: parse → semantic chunking (sentence-boundary + embedding similarity) → BGE-M3 embed → Chroma + in-memory BM25 index. Query path: LLM rewrite → dense search (Chroma top-20) + sparse search (BM25 top-20) → RRF fusion → cross-encoder reranker → top-5.

**Tech Stack:** Python 3.12+, FlagEmbedding (BGE-M3, reranker), Chroma, sentence-transformers (removed for BGE-M3, kept for compatibility), DeepSeek API.

---

## File Map

```
personal_agent/
├── kb/
│   ├── embed.py          # BGE-M3 FlagEmbedding wrapper (dense + sparse)
│   ├── ingest.py         # Semantic chunker replacing paragraph splitter
│   ├── retrieval.py      # Hybrid search + BM25 index + reranker + query rewrite
│   └── __init__.py
├── tools/
│   └── kb_search.py      # Wire query rewrite before search
├── cli/
│   └── app.py            # Auto-migration on startup
tests/
├── test_embed.py         # 1024-dim + sparse output tests
├── test_ingest.py        # Semantic chunker tests
├── test_retrieval.py     # Hybrid search, rerank, rewrite tests
├── test_rag_quality.py   # **New** — quality gate
└── conftest.py           # Updated dummy EF dimensions
```

---

### Task 1: Add FlagEmbedding Dependency + Rewrite Embedder

**Files:**
- Modify: `pyproject.toml`
- Modify: `personal_agent/kb/embed.py`
- Modify: `tests/test_embed.py`

- [ ] **Step 1: Add FlagEmbedding dependency**

Run:
```bash
cd /Users/cchhyy/Desktop/personal_agent && poetry add FlagEmbedding
```

- [ ] **Step 2: Write failing test for BGE-M3 embedder**

Write `tests/test_embed.py`:
```python
from personal_agent.kb.embed import Embedder


def test_embedder_loads_model():
    embedder = Embedder()
    assert embedder.model is not None


def test_embed_dense_returns_correct_dimensions():
    embedder = Embedder()
    result = embedder.embed(["hello world", "test sentence"])
    assert len(result) == 2
    assert len(result[0]) == 1024  # BGE-M3 dimension


def test_embed_single_string():
    embedder = Embedder()
    result = embedder.embed("hello world")
    assert len(result) == 1
    assert len(result[0]) == 1024


def test_embedder_caches():
    e1 = Embedder()
    e2 = Embedder()
    assert e1.model is e2.model


def test_embed_sparse_returns_lexical_weights():
    embedder = Embedder()
    result = embedder.embed_sparse(["hello world", "test sentence"])
    assert len(result) == 2
    assert isinstance(result[0], dict)
    assert len(result[0]) > 0  # Non-empty sparse weights


def test_embed_sparse_single_string():
    embedder = Embedder()
    result = embedder.embed_sparse("hello world")
    assert len(result) == 1
    assert isinstance(result[0], dict)
```

- [ ] **Step 3: Run test to verify it fails**

Run:
```bash
cd /Users/cchhyy/Desktop/personal_agent && poetry run pytest tests/test_embed.py -v
```
Expected: FAIL — 384 → 1024 assertion fails, `embed_sparse` not found

- [ ] **Step 4: Rewrite embed.py with BGEM3FlagModel**

Write `personal_agent/kb/embed.py`:
```python
from FlagEmbedding import BGEM3FlagModel

_MODEL_NAME = "BAAI/bge-m3"
_model: BGEM3FlagModel | None = None


def _get_model() -> BGEM3FlagModel:
    global _model
    if _model is None:
        _model = BGEM3FlagModel(_MODEL_NAME, use_fp16=True)
    return _model


class Embedder:
    def __init__(self):
        self.model = _get_model()

    def embed(self, texts: str | list[str]) -> list[list[float]]:
        """Return dense embeddings (1024-dim per text)."""
        if isinstance(texts, str):
            texts = [texts]
        output = self.model.encode(texts, return_dense=True, return_sparse=False)
        return output["dense_vecs"].tolist()

    def embed_sparse(self, texts: str | list[str]) -> list[dict[int, float]]:
        """Return sparse lexical weights (token_id -> weight per text)."""
        if isinstance(texts, str):
            texts = [texts]
        output = self.model.encode(texts, return_dense=False, return_sparse=True)
        return output["lexical_weights"]
```

- [ ] **Step 5: Run test to verify it passes**

Run:
```bash
cd /Users/cchhyy/Desktop/personal_agent && poetry run pytest tests/test_embed.py -v
```
Expected: PASS (first run downloads BGE-M3, ~2GB)

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml poetry.lock personal_agent/kb/embed.py tests/test_embed.py
git commit -m "feat: swap embeddings to BGE-M3 with dense + sparse output"
```

---

### Task 2: Semantic Chunking

**Files:**
- Modify: `personal_agent/kb/ingest.py`
- Modify: `tests/test_ingest.py`

- [ ] **Step 1: Write failing tests for semantic chunker**

Write `tests/test_ingest.py`:
```python
from pathlib import Path
from personal_agent.kb.ingest import parse_file, chunk_text, ingest_file, ingest_directory, Chunk


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


def test_semantic_chunker_splits_on_topic_boundary():
    text = (
        "The Python programming language is widely used for data science. "
        "It has many libraries for machine learning and statistics.\n\n"
        "Football is a popular sport played with 11 players on each team. "
        "The World Cup is held every four years and draws global audiences."
    )
    chunks = chunk_text(text, chunk_size=500)
    # Two distinct topics — should produce at least 2 chunks
    assert len(chunks) >= 2
    for chunk in chunks:
        assert len(chunk.text) > 0


def test_semantic_chunker_keeps_similar_content_together():
    text = (
        "Python is a high-level programming language. "
        "It was created by Guido van Rossum. "
        "Python emphasizes code readability."
    )
    chunks = chunk_text(text, chunk_size=500)
    # Same topic — should be 1 chunk (or few)
    assert len(chunks) >= 1
    # The chunk should contain key content
    combined = " ".join(c.text for c in chunks)
    assert "Python" in combined
    assert "Guido van Rossum" in combined


def test_semantic_chunker_handles_short_text():
    text = "Hello world."
    chunks = chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0].text == "Hello world."


def test_semantic_chunker_preserves_metadata():
    text = "Some content for chunking test. More content here."
    chunks = chunk_text(text, source_meta={"source": "test.md", "type": "markdown"})
    for chunk in chunks:
        assert chunk.metadata["source"] == "test.md"


def test_semantic_chunker_adds_heading_metadata():
    text = (
        "# Introduction\n\n"
        "This is the introduction paragraph with enough content to form a chunk. "
        "It continues with more details about the topic.\n\n"
        "## Methods\n\n"
        "The methods section describes the approach taken in this research. "
        "We used several techniques to gather and analyze data."
    )
    chunks = chunk_text(text)
    assert len(chunks) >= 1
    # At least one chunk should have heading metadata
    headings_found = any("heading" in c.metadata for c in chunks)
    assert headings_found


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
    import tempfile
    (temp_dir / "a.md").write_text("# Doc A\nContent A")
    (temp_dir / "b.txt").write_text("Content B")
    with tempfile.TemporaryDirectory() as chroma_dir:
        client = chromadb.PersistentClient(path=str(Path(chroma_dir) / "chroma2"))
        collection = client.get_or_create_collection("test_kb2", embedding_function=_dummy_ef())
        ids, errors = ingest_directory(temp_dir, collection)
        assert len(ids) >= 2
        assert len(errors) == 0


def test_reingest_replaces_chunks(temp_dir, sample_md_file):
    import chromadb
    client = chromadb.PersistentClient(path=str(temp_dir / "chroma3"))
    collection = client.get_or_create_collection("test_kb3", embedding_function=_dummy_ef())
    first_ids = ingest_file(sample_md_file, collection)
    second_ids = ingest_file(sample_md_file, collection)
    assert set(first_ids) != set(second_ids)
    old_results = collection.get(ids=first_ids)
    assert len(old_results["ids"]) == 0


def _dummy_ef():
    from chromadb.api.types import EmbeddingFunction
    class DummyEF(EmbeddingFunction):
        def __call__(self, input):
            return [[0.1] * 1024 for _ in input]
    return DummyEF()
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd /Users/cchhyy/Desktop/personal_agent && poetry run pytest tests/test_ingest.py -v
```
Expected: FAIL — semantic chunking tests fail (old chunker still in place), plus 384-dim dummy EF creates mismatches

- [ ] **Step 3: Rewrite chunk_text() in ingest.py**

Write `personal_agent/kb/ingest.py`:
```python
import hashlib
import logging
import re
import time
from pathlib import Path
from dataclasses import dataclass

from personal_agent.kb.embed import Embedder


logger = logging.getLogger(__name__)

_embedder: Embedder | None = None


def _get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


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


def _extract_headings(text: str) -> list[tuple[int, str, str]]:
    """Extract markdown headings. Returns list of (char_position, level, text)."""
    headings = []
    for m in re.finditer(r'^(#{1,6})\s+(.+)$', text, re.MULTILINE):
        level = len(m.group(1))
        headings.append((m.start(), level, m.group(2).strip()))
    return headings


def _find_nearest_heading(pos: int, headings: list[tuple[int, str, str]]) -> str | None:
    """Return the nearest heading text that precedes pos, or None."""
    best = None
    for h_pos, _level, h_text in headings:
        if h_pos < pos:
            best = h_text
        else:
            break
    return best


def _sentences(text: str) -> list[str]:
    """Split text into sentences using boundary patterns."""
    raw = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in raw if s.strip()]


def _token_estimate(text: str) -> int:
    return len(text) // 4


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    source_meta: dict | None = None,
) -> list[Chunk]:
    """Semantic chunking: sentence-boundary-aware, merges by embedding similarity."""
    meta = source_meta or {}
    headings = _extract_headings(text)
    sentences = _sentences(text)
    if not sentences:
        return []

    embedder = _get_embedder()

    # Encode each sentence for similarity checking
    sent_embeddings = embedder.embed(sentences)

    chunks = []
    group = [sentences[0]]
    group_pos = 0  # char offset of first sentence in current group

    # Pre-compute sentence character positions for heading tracking
    char_positions = [0]
    for s in sentences[:-1]:
        next_pos = char_positions[-1] + len(s) + 1
        char_positions.append(next_pos)

    for i in range(1, len(sentences)):
        current_sent = sentences[i]
        combined_tokens = _token_estimate(" ".join(group))

        # Compute cosine similarity between group centroid and next sentence
        group_emb = embedder.embed(" ".join(group))[0]
        next_emb = sent_embeddings[i]

        dot = sum(a * b for a, b in zip(group_emb, next_emb))
        norm_a = sum(a * a for a in group_emb) ** 0.5
        norm_b = sum(b * b for b in next_emb) ** 0.5
        sim = dot / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0.0

        # Determine threshold: tighter when approaching max chunk size
        if combined_tokens < 256:
            threshold = 0.3
        elif combined_tokens > 1024:
            threshold = 0.9  # Force split
        else:
            threshold = 0.5

        if sim >= threshold and combined_tokens < 1024:
            group.append(current_sent)
        else:
            # Flush current group
            chunk_text_str = " ".join(group)
            chunk_meta = {**meta, "chunk_index": len(chunks)}
            heading = _find_nearest_heading(char_positions[group_pos], headings)
            if heading:
                chunk_meta["heading"] = heading
            chunks.append(Chunk(text=chunk_text_str, metadata=chunk_meta))

            # Start new group with overlap
            if chunk_overlap > 0 and len(group) > 1:
                # Keep last sentence for overlap context
                overlap_chars = chunk_overlap * 4
                overlap_text = " ".join(group[-1:])
                # Find where to resume: prepend overlap sentence
                group = group[-1:] + [current_sent]
                group_pos = i - 1
            else:
                group = [current_sent]
                group_pos = i

    # Flush final group
    if group:
        chunk_text_str = " ".join(group)
        chunk_meta = {**meta, "chunk_index": len(chunks)}
        heading = _find_nearest_heading(char_positions[group_pos], headings)
        if heading:
            chunk_meta["heading"] = heading
        chunks.append(Chunk(text=chunk_text_str, metadata=chunk_meta))

    return chunks


def _file_hash(file_path: Path) -> str:
    return hashlib.md5(file_path.read_bytes()).hexdigest()


def _delete_existing(file_path: Path, collection):
    existing = collection.get(where={"source": str(file_path)})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])


def ingest_file(file_path: Path, collection) -> list[str]:
    file_path = Path(file_path)
    _delete_existing(file_path, collection)

    text, meta = parse_file(file_path)
    file_hash = _file_hash(file_path)
    meta["file_hash"] = file_hash
    chunks = chunk_text(text, source_meta=meta)

    if not chunks:
        return []

    ids = []
    documents = []
    metadatas = []
    embeddings = []

    embedder = _get_embedder()

    for c in chunks:
        chunk_id = f"{file_path.stem}_{c.metadata['chunk_index']}_{file_hash[:8]}_{time.time_ns()}"
        ids.append(chunk_id)
        documents.append(c.text)
        metadatas.append(c.metadata)
        embeddings.append(embedder.embed(c.text)[0])

    collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    return ids


def ingest_directory(dir_path: Path, collection) -> tuple[list[str], list[str]]:
    dir_path = Path(dir_path)
    all_ids = []
    errors = []
    for p in dir_path.rglob("*"):
        if p.is_file() and not p.name.startswith("."):
            try:
                ids = ingest_file(p, collection)
                all_ids.extend(ids)
            except Exception as e:
                msg = f"Failed to ingest {p}: {e}"
                logger.warning(msg)
                errors.append(msg)
    return all_ids, errors
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
cd /Users/cchhyy/Desktop/personal_agent && poetry run pytest tests/test_ingest.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add personal_agent/kb/ingest.py tests/test_ingest.py
git commit -m "feat: add semantic chunking with sentence-boundary splits"
```

---

### Task 3: Hybrid Search + BM25 Index + Reranker

**Files:**
- Modify: `personal_agent/kb/retrieval.py`
- Modify: `tests/test_retrieval.py`

- [ ] **Step 1: Write failing tests for hybrid search + reranker**

Write `tests/test_retrieval.py`:
```python
import chromadb
from chromadb.api.types import EmbeddingFunction
from personal_agent.kb.retrieval import KBMetadata, rewrite_query
from unittest.mock import MagicMock, patch


class _DummyEF(EmbeddingFunction):
    def __call__(self, input):
        return [[0.1] * 1024 for _ in input]


def _make_client(temp_dir, name):
    return chromadb.PersistentClient(path=str(temp_dir / name))


def _add_docs_to_collection(client, name, docs):
    """Add documents to a collection for search testing."""
    collection = client.get_or_create_collection(name, embedding_function=_DummyEF())
    for i, doc in enumerate(docs):
        collection.add(
            ids=[f"doc_{i}"],
            documents=[doc],
            embeddings=[[0.1] * 1024],
            metadatas=[{"source": f"file_{i}.md", "filename": f"file_{i}.md", "chunk_index": i}],
        )


def test_search_returns_results(temp_dir, sample_md_file):
    from personal_agent.kb.ingest import ingest_file

    client = _make_client(temp_dir, "chroma_s1")
    collection = client.get_or_create_collection("test_search", embedding_function=_DummyEF())
    ingest_file(sample_md_file, collection)

    retriever = KBMetadata(client, collection_name="test_search")
    results = retriever.search("test document")

    assert len(results) > 0
    assert "id" in results[0]
    assert "text" in results[0]
    assert "source" in results[0]


def test_search_returns_empty_for_no_docs(temp_dir):
    client = _make_client(temp_dir, "chroma_s2")
    client.get_or_create_collection("test_empty", embedding_function=_DummyEF())

    retriever = KBMetadata(client, collection_name="test_empty")
    results = retriever.search("something")
    assert len(results) == 0


def test_hybrid_search_dense_path(temp_dir):
    """Search works even without the BM25 index populated."""
    client = _make_client(temp_dir, "chroma_s3")
    _add_docs_to_collection(client, "test_hybrid", [
        "Python is a programming language used for data science.",
        "Football is a popular sport worldwide.",
        "Cooking requires fresh ingredients and patience.",
    ])

    retriever = KBMetadata(client, collection_name="test_hybrid")
    results = retriever.search("programming with Python")

    assert len(results) > 0
    assert "Python" in results[0]["text"]


def test_sparse_index_builds_on_search(temp_dir):
    """The sparse BM25 index is lazily built on first search."""
    client = _make_client(temp_dir, "chroma_s4")
    _add_docs_to_collection(client, "test_sparse_build", [
        "machine learning models and training data",
        "soccer world cup championship finals",
        "baking bread with sourdough starter",
    ])

    retriever = KBMetadata(client, collection_name="test_sparse_build")
    assert retriever._sparse_index is None

    retriever.search("machine learning")
    assert retriever._sparse_index is not None


def test_reranker_loads():
    """Reranker model loads successfully."""
    from personal_agent.kb.retrieval import _get_reranker
    reranker = _get_reranker()
    assert reranker is not None


def test_rerank_scores_pairs():
    """Reranker scores (query, chunk) pairs."""
    from personal_agent.kb.retrieval import _get_reranker
    reranker = _get_reranker()
    scores = reranker.compute_score([
        ("machine learning", "Machine learning is a subset of artificial intelligence."),
        ("machine learning", "The football match was exciting."),
    ])
    assert len(scores) == 2
    assert isinstance(scores[0], float)
    # First pair should score higher (relevant)
    assert scores[0] > scores[1]


def test_get_document_count(temp_dir):
    client = _make_client(temp_dir, "chroma_s5")
    _add_docs_to_collection(client, "test_count", ["doc a", "doc b"])
    retriever = KBMetadata(client, collection_name="test_count")
    assert retriever.document_count == 2


def test_list_documents(temp_dir, sample_md_file):
    from personal_agent.kb.ingest import ingest_file

    client = _make_client(temp_dir, "chroma_s6")
    collection = client.get_or_create_collection("test_list", embedding_function=_DummyEF())
    ingest_file(sample_md_file, collection)

    retriever = KBMetadata(client, collection_name="test_list")
    docs = retriever.list_documents()
    assert len(docs) >= 1
    assert "source" in docs[0] and "filename" in docs[0]


def test_remove_document(temp_dir, sample_md_file):
    from personal_agent.kb.ingest import ingest_file

    client = _make_client(temp_dir, "chroma_s7")
    collection = client.get_or_create_collection("test_remove", embedding_function=_DummyEF())
    ingest_file(sample_md_file, collection)

    retriever = KBMetadata(client, collection_name="test_remove")
    removed = retriever.remove_document(str(sample_md_file))
    assert removed > 0

    results = collection.get(where={"source": str(sample_md_file)})
    assert len(results["ids"]) == 0


def test_query_rewrite():
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="machine learning attention mechanism transformer architecture"))]
    )
    result = rewrite_query("what did that paper say about attention?", mock_client)
    assert "attention" in result.lower()
    assert len(result) > 10


def test_query_rewrite_handles_error():
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("API down")
    result = rewrite_query("test query", mock_client)
    assert result == "test query"  # Falls back to original
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd /Users/cchhyy/Desktop/personal_agent && poetry run pytest tests/test_retrieval.py -v
```
Expected: FAIL — import errors for new functions, 384-dim mismatches

- [ ] **Step 3: Write retrieval.py with hybrid search, BM25 index, reranker, and query rewrite**

Write `personal_agent/kb/retrieval.py`:
```python
import math
import re
from collections import defaultdict
from pathlib import Path

import chromadb
from FlagEmbedding import FlagReranker

from personal_agent.kb.embed import Embedder


_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
_reranker: FlagReranker | None = None


def _get_reranker() -> FlagReranker:
    global _reranker
    if _reranker is None:
        _reranker = FlagReranker(_RERANKER_MODEL, use_fp16=True)
    return _reranker


class _SparseIndex:
    """In-memory BM25-style index for keyword search."""

    def __init__(self):
        self._docs: dict[str, dict[str, int]] = {}  # doc_id -> {token -> tf}
        self._df: dict[str, int] = defaultdict(int)  # document frequency
        self._total_docs = 0
        self._avg_dl = 0.0
        self._total_terms = 0

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r'[a-zA-Z0-9]+', text.lower())

    def add(self, doc_id: str, text: str) -> None:
        tokens = self._tokenize(text)
        tf = defaultdict(int)
        for t in tokens:
            tf[t] += 1
        self._docs[doc_id] = dict(tf)
        for t in set(tokens):
            self._df[t] += 1
        self._total_docs += 1
        self._total_terms += len(tokens)
        self._avg_dl = self._total_terms / self._total_docs

    def remove(self, doc_id: str) -> None:
        if doc_id not in self._docs:
            return
        for t in self._docs[doc_id]:
            self._df[t] -= 1
            if self._df[t] <= 0:
                del self._df[t]
        self._total_terms -= sum(self._docs[doc_id].values())
        self._total_docs -= 1
        self._avg_dl = self._total_terms / self._total_docs if self._total_docs > 0 else 0.0
        del self._docs[doc_id]

    def search(self, query_weights: dict[str, float], top_k: int = 20) -> list[tuple[str, float]]:
        """Score documents against query using BM25 with query-side weights."""
        k1 = 1.5
        b = 0.75
        scores = []

        for doc_id, tf in self._docs.items():
            doc_len = sum(tf.values())
            score = 0.0
            for token, q_weight in query_weights.items():
                if token not in tf:
                    continue
                df = self._df.get(token, 0)
                if df == 0:
                    continue
                idf = math.log((self._total_docs - df + 0.5) / (df + 0.5) + 1.0)
                tf_norm = (tf[token] * (k1 + 1)) / (tf[token] + k1 * (1 - b + b * doc_len / self._avg_dl))
                score += q_weight * idf * tf_norm
            if score > 0:
                scores.append((doc_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def count(self) -> int:
        return self._total_docs

    def display_token(self, token: str) -> str:
        return token


class KBMetadata:
    def __init__(self, client: chromadb.PersistentClient, collection_name: str = "kb"):
        self.client = client
        self.collection_name = collection_name
        self._collection = None
        self._embedder = Embedder()
        self._sparse_index: _SparseIndex | None = None

    @property
    def collection(self) -> chromadb.Collection:
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                self.collection_name,
                embedding_function=_ChromaEmbeddingAdapter(self._embedder),
            )
        return self._collection

    @property
    def document_count(self) -> int:
        return self.collection.count()

    def _ensure_sparse_index(self) -> _SparseIndex:
        """Lazily build the BM25 index from Chroma documents."""
        if self._sparse_index is not None:
            return self._sparse_index

        self._sparse_index = _SparseIndex()
        existing = self.collection.get(include=["documents"])
        if existing["ids"] and existing["documents"]:
            for doc_id, doc_text in zip(existing["ids"], existing["documents"]):
                if doc_text:
                    self._sparse_index.add(doc_id, doc_text)
        return self._sparse_index

    def _dense_search(self, query: str, n_results: int = 20) -> list[tuple[str, float]]:
        results = self.collection.query(query_texts=[query], n_results=n_results)
        if not results["ids"] or not results["ids"][0]:
            return []
        pairs = []
        distances = results.get("distances", [[]])
        for i, doc_id in enumerate(results["ids"][0]):
            dist = distances[0][i] if distances[0] and i < len(distances[0]) else 0.0
            score = 1.0 - dist if dist else 1.0
            pairs.append((doc_id, score))
        return pairs

    def _sparse_search(self, query: str, n_results: int = 20) -> list[tuple[str, float]]:
        sparse_index = self._ensure_sparse_index()
        if sparse_index.count() == 0:
            return []

        # Get query lexical weights from BGE-M3, convert token IDs to strings
        sparse_output = self._embedder.embed_sparse(query)
        if not sparse_output or not sparse_output[0]:
            return []

        # Convert token_id dict to token_string dict
        token_weights: dict[str, float] = {}
        for token_id, weight in sparse_output[0].items():
            token = self._embedder.model.tokenizer.decode([token_id]).strip().lower()
            if token and len(token) > 1:
                token_weights[token] = weight

        return sparse_index.search(token_weights, top_k=n_results)

    @staticmethod
    def _rrf_fuse(
        dense_results: list[tuple[str, float]],
        sparse_results: list[tuple[str, float]],
        k: int = 60,
        top_k: int = 20,
    ) -> list[str]:
        """Reciprocal rank fusion: combine two ranked lists into one."""
        scores: dict[str, float] = {}

        for rank, (doc_id, _) in enumerate(dense_results):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)

        for rank, (doc_id, _) in enumerate(sparse_results):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in sorted_docs[:top_k]]

    def _rerank(self, query: str, doc_ids: list[str], top_k: int = 5) -> list[str]:
        if not doc_ids:
            return []

        # Fetch document texts
        results = self.collection.get(ids=doc_ids, include=["documents"])
        doc_map = {}
        if results["ids"] and results["documents"]:
            for doc_id, doc_text in zip(results["ids"], results["documents"]):
                doc_map[doc_id] = doc_text or ""

        pairs = [(query, doc_map.get(doc_id, "")) for doc_id in doc_ids]
        reranker = _get_reranker()
        scores = reranker.compute_score(pairs)

        # Handle single score case
        if not isinstance(scores, list):
            scores = [scores]

        scored = list(zip(doc_ids, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in scored[:top_k]]

    def search(self, query: str, n_results: int = 5) -> list[dict]:
        # Step 1: Dense search
        dense_results = self._dense_search(query, n_results=20)

        # Step 2: Sparse search
        sparse_results = self._sparse_search(query, n_results=20)

        # Step 3: RRF fusion
        merged_ids = self._rrf_fuse(dense_results, sparse_results, top_k=20)

        # Step 4: Rerank
        reranked_ids = self._rerank(query, merged_ids, top_k=n_results)

        # Step 5: Fetch and format results
        if not reranked_ids:
            return []

        results = self.collection.get(ids=reranked_ids, include=["documents", "metadatas"])
        output = []
        for doc_id in reranked_ids:
            idx = results["ids"].index(doc_id) if doc_id in results["ids"] else -1
            if idx < 0:
                continue
            meta = {}
            if results["metadatas"] and results["metadatas"][idx] is not None:
                meta = results["metadatas"][idx]
            output.append({
                "id": doc_id,
                "text": results["documents"][idx] if results["documents"] else "",
                "source": meta.get("source", ""),
                "filename": meta.get("filename", ""),
            })
        return output

    def list_documents(self) -> list[dict]:
        all_data = self.collection.get(include=["metadatas"])
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
        if not source_path:
            return 0
        existing = self.collection.get(where={"source": source_path})
        if existing["ids"]:
            self.collection.delete(ids=existing["ids"])
            # Also remove from sparse index
            if self._sparse_index:
                for doc_id in existing["ids"]:
                    self._sparse_index.remove(doc_id)
            return len(existing["ids"])
        return 0

    def add_to_sparse_index(self, doc_id: str, text: str) -> None:
        """Called after ingestion to keep the sparse index in sync."""
        if self._sparse_index is not None:
            self._sparse_index.add(doc_id, text)


def rewrite_query(query: str, client, model: str = "deepseek-chat") -> str:
    """Rewrite a user query for better retrieval. Falls back to original on failure."""
    if client is None:
        return query
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": (
                    "Rewrite the user query into a self-contained search query optimized "
                    "for semantic retrieval. Include key terms, expand abbreviations, and "
                    "add relevant context. Output only the rewritten query."
                )},
                {"role": "user", "content": query},
            ],
            temperature=0.3,
            max_tokens=100,
        )
        rewritten = response.choices[0].message.content.strip()
        return rewritten if rewritten else query
    except Exception:
        return query


class _ChromaEmbeddingAdapter:
    def __init__(self, embedder: Embedder):
        self._embedder = embedder

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self._embedder.embed(input)

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return self._embedder.embed(input)

    def embed_document(self, input: list[str]) -> list[list[float]]:
        return self._embedder.embed(input)

    def name(self) -> str:
        return "BAAI/bge-m3"
```

- [ ] **Step 4: Run tests**

Run:
```bash
cd /Users/cchhyy/Desktop/personal_agent && poetry run pytest tests/test_retrieval.py -v
```
Expected: PASS (first run downloads reranker model, ~1GB)

- [ ] **Step 5: Commit**

```bash
git add personal_agent/kb/retrieval.py tests/test_retrieval.py
git commit -m "feat: add hybrid search with BM25, RRF fusion, and cross-encoder reranker"
```

---

### Task 4: Wire Query Rewriting into kb_search Tool

**Files:**
- Modify: `personal_agent/tools/kb_search.py`
- Modify: `tests/test_tools.py`

- [ ] **Step 1: Update kb_search tool to accept optional client for query rewrite**

Write `personal_agent/tools/kb_search.py`:
```python
import json
from personal_agent.kb.retrieval import KBMetadata, rewrite_query


def kb_search(query: str, retriever: KBMetadata, llm_client=None, model: str = "deepseek-chat") -> str:
    if llm_client is not None:
        query = rewrite_query(query, llm_client, model=model)
    results = retriever.search(query, n_results=5)
    if not results:
        return json.dumps({"results": [], "message": "No results found in knowledge base."})
    return json.dumps({"results": results}, ensure_ascii=False)
```

- [ ] **Step 2: Add rewrite test to tests/test_tools.py**

Read current `tests/test_tools.py` and append:
```python
from unittest.mock import MagicMock


def test_kb_search_with_query_rewrite(temp_dir, sample_md_file):
    import chromadb
    import json
    from personal_agent.kb.ingest import ingest_file
    from personal_agent.kb.retrieval import KBMetadata
    from personal_agent.tools.kb_search import kb_search

    class DummyEF:
        def __call__(self, input):
            return [[0.1] * 1024 for _ in input]

    client = chromadb.PersistentClient(path=str(temp_dir / "chroma_rewrite"))
    collection = client.get_or_create_collection("test_rewrite", embedding_function=DummyEF())
    ingest_file(sample_md_file, collection)
    retriever = KBMetadata(client, collection_name="test_rewrite")

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="expanded test document query"))]
    )

    result = kb_search("short query", retriever=retriever, llm_client=mock_client)
    data = json.loads(result)
    assert "results" in data

    # Verify the LLM was called for rewrite
    assert mock_client.chat.completions.create.called


def test_kb_search_without_llm_client_skips_rewrite(temp_dir, sample_md_file):
    import chromadb
    import json
    from personal_agent.kb.ingest import ingest_file
    from personal_agent.kb.retrieval import KBMetadata
    from personal_agent.tools.kb_search import kb_search

    class DummyEF:
        def __call__(self, input):
            return [[0.1] * 1024 for _ in input]

    client = chromadb.PersistentClient(path=str(temp_dir / "chroma_norewrite"))
    collection = client.get_or_create_collection("test_norewrite", embedding_function=DummyEF())
    ingest_file(sample_md_file, collection)
    retriever = KBMetadata(client, collection_name="test_norewrite")

    result = kb_search("test document", retriever=retriever)
    data = json.loads(result)
    assert "results" in data
```

- [ ] **Step 3: Run tests**

Run:
```bash
cd /Users/cchhyy/Desktop/personal_agent && poetry run pytest tests/test_tools.py::test_kb_search_with_query_rewrite tests/test_tools.py::test_kb_search_without_llm_client_skips_rewrite -v
```
Expected: PASS

- [ ] **Step 4: Update cli/app.py _setup_tools to pass LLM client to kb_search**

In `personal_agent/cli/app.py`, change `_setup_tools` signature and the kb_search registration:

```python
def _setup_tools(retriever: KBMetadata, tavily_client: TavilyClient, config=None, llm_client=None) -> ToolRegistry:
    model = config.deepseek_model if config else "deepseek-chat"
    registry = ToolRegistry()
    registry.register(Tool(
        name="kb_search",
        description="Search the local knowledge base for relevant information.",
        function=lambda query: kb_search(query, retriever=retriever, llm_client=llm_client, model=model),
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query"}},
            "required": ["query"],
        },
    ))
    # ... rest of the tool registrations remain the same
    return registry
```

And inside `run()`, update the call:
```python
llm_client = OpenAI(
    api_key=config.deepseek_api_key,
    base_url=config.deepseek_base_url,
)
registry = _setup_tools(retriever, tavily_client, config, llm_client)
```

- [ ] **Step 5: Run existing CLI tests**

Run:
```bash
cd /Users/cchhyy/Desktop/personal_agent && poetry run pytest tests/test_cli.py -v
```
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add personal_agent/tools/kb_search.py tests/test_tools.py personal_agent/cli/app.py
git commit -m "feat: wire LLM query rewriting into kb_search tool"
```

---

### Task 5: Migration — Auto-Rebuild Old Collections

**Files:**
- Modify: `personal_agent/cli/app.py`

- [ ] **Step 1: Add migration check function to kb/retrieval.py**

Append to `personal_agent/kb/retrieval.py`:
```python
def check_and_migrate_kb(chroma_dir: str, kb_dir: str | None) -> bool:
    """Check if existing KB needs migration (old 384-dim → new 1024-dim).
    Returns True if a migration was performed.
    """
    import logging
    import shutil

    logger = logging.getLogger(__name__)

    chroma_dir_path = Path(chroma_dir)
    if not chroma_dir_path.exists():
        return False

    client = chromadb.PersistentClient(path=str(chroma_dir_path))
    try:
        collection = client.get_collection("kb")
    except Exception:
        return False

    if collection.count() == 0:
        return False

    # Check dimension by getting one embedding
    sample = collection.get(limit=1, include=["embeddings"])
    if not sample["embeddings"] or not sample["embeddings"][0]:
        return False

    dim = len(sample["embeddings"][0])
    if dim == 1024:
        return False  # Already migrated

    logger.warning("KB collection uses %s-dim embeddings (now 1024). Rebuilding index.", dim)
    client.delete_collection("kb")

    if kb_dir:
        kb_path = Path(kb_dir).expanduser()
        if kb_path.exists():
            retriever = KBMetadata(client, collection_name="kb")
            from personal_agent.kb.ingest import ingest_directory
            _, errors = ingest_directory(kb_path, retriever.collection)
            for err in errors:
                logger.warning(err)
            logger.info("KB migration complete: %d documents re-indexed", retriever.document_count)
            return True
    return False
```

- [ ] **Step 2: Call migration in cli/app.py run()**

In `personal_agent/cli/app.py` `run()` function, add after `config.agent_dir.mkdir(...)`:
```python
# Check and migrate KB from old dimension to new (384 -> 1024)
from personal_agent.kb.retrieval import check_and_migrate_kb
check_and_migrate_kb(str(config.chroma_dir), str(config.kb_dir) if config.kb_dir else None)
```

- [ ] **Step 3: Commit**

```bash
git add personal_agent/kb/retrieval.py personal_agent/cli/app.py
git commit -m "feat: add KB auto-migration from 384-dim to 1024-dim embeddings"
```

---

### Task 6: Update Remaining Tests

**Files:**
- Modify: `tests/conftest.py`
- Modify: `tests/test_tools.py`
- Modify: `tests/test_agent.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Update all 384-dim references in tests**

Search for and update all `384` → `1024` in test files:
```bash
cd /Users/cchhyy/Desktop/personal_agent && grep -rn "384" tests/
```

Expected locations to fix:
- `tests/test_tools.py` — DummyEF and _DummyEF classes (384 → 1024)
- Any remaining `384` in test assertions

- [ ] **Step 2: Update conftest.py if needed**

No changes needed — conftest doesn't reference embed dimensions directly.

- [ ] **Step 3: Run full test suite**

Run:
```bash
cd /Users/cchhyy/Desktop/personal_agent && poetry run pytest tests/ -v --ignore=tests/test_rag_quality.py
```
Expected: All non-slow tests PASS

- [ ] **Step 4: Commit**

```bash
git add tests/
git commit -m "test: update test dimensions from 384 to 1024 for BGE-M3"
```

---

### Task 7: RAG Quality Gate

**Files:**
- Create: `tests/test_rag_quality.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Create quality test documents in conftest**

Append to `tests/conftest.py`:
```python
@pytest.fixture
def quality_docs(temp_dir):
    """Hand-crafted documents for RAG quality evaluation."""
    docs = {
        "python_intro.md": (
            "# Python Programming\n\n"
            "Python is a high-level, interpreted programming language known for "
            "its readability and versatility. It was created by Guido van Rossum "
            "and first released in 1991. Python supports multiple programming "
            "paradigms including procedural, object-oriented, and functional programming.\n\n"
            "## Key Features\n\n"
            "Python's design philosophy emphasizes code readability through its "
            "use of significant indentation. Its language constructs and "
            "object-oriented approach aim to help programmers write clear, "
            "logical code for small and large-scale projects."
        ),
        "machine_learning.md": (
            "# Machine Learning Fundamentals\n\n"
            "Machine learning is a subset of artificial intelligence that enables "
            "systems to learn and improve from experience without being explicitly "
            "programmed. The core idea is to develop algorithms that can receive "
            "input data and use statistical analysis to predict an output.\n\n"
            "## Types of Machine Learning\n\n"
            "Supervised learning uses labeled training data to learn a mapping "
            "from inputs to outputs. Unsupervised learning finds hidden patterns "
            "in unlabeled data. Reinforcement learning trains agents to make "
            "sequences of decisions through reward signals.\n\n"
            "## Deep Learning\n\n"
            "Deep learning is a specialized form of machine learning that uses "
            "neural networks with many layers (hence 'deep') to progressively "
            "extract higher-level features from raw input."
        ),
        "cooking_tips.md": (
            "# Essential Cooking Tips\n\n"
            "Good cooking starts with fresh ingredients. Always read the entire "
            "recipe before you begin cooking. Mise en place — having all your "
            "ingredients prepared and measured before you start — is the "
            "foundation of efficient cooking.\n\n"
            "## Seasoning\n\n"
            "Salt is the most important seasoning in any kitchen. Add salt in "
            "layers throughout the cooking process rather than all at once at "
            "the end. Taste as you go and adjust seasoning accordingly."
        ),
        "climate_science.md": (
            "# Climate Change Overview\n\n"
            "Climate change refers to long-term shifts in temperatures and "
            "weather patterns. These shifts may be natural, but since the 1800s, "
            "human activities have been the main driver of climate change, "
            "primarily due to the burning of fossil fuels.\n\n"
            "## Greenhouse Effect\n\n"
            "The greenhouse effect is the process through which heat is trapped "
            "near Earth's surface by greenhouse gases. Carbon dioxide, methane, "
            "and water vapor are the primary greenhouse gases."
        ),
        "product_review.md": (
            "# Smartphone Review: Model X Pro\n\n"
            "The Model X Pro features a 6.7-inch OLED display with 120Hz refresh "
            "rate. Battery life is excellent at 5000mAh, lasting a full day of "
            "heavy use. The camera system includes a 108MP main sensor, 12MP "
            "ultrawide, and 10MP telephoto with 3x optical zoom.\n\n"
            "## Performance\n\n"
            "Powered by the latest Snapdragon processor and 12GB of RAM, the "
            "Model X Pro handles multitasking and gaming with ease. Storage "
            "options include 128GB, 256GB, and 512GB variants."
        ),
    }
    for name, content in docs.items():
        p = temp_dir / name
        p.write_text(content)
    return temp_dir
```

- [ ] **Step 2: Write the RAG quality test**

Write `tests/test_rag_quality.py`:
```python
"""RAG quality gate — must pass before merging any RAG pipeline change.

Uses real BGE-M3 embedder and reranker. Marked @pytest.mark.slow.
"""

import json
import math
from collections import defaultdict

import chromadb
import pytest

from personal_agent.kb.ingest import ingest_directory
from personal_agent.kb.retrieval import KBMetadata


# Quality thresholds
RECALL_AT_5_MIN = 0.80
MRR_MIN = 0.60
NDCG_AT_5_MIN = 0.70


@pytest.mark.slow
def test_rag_quality_recall_at_5(quality_docs, temp_dir):
    """Recall@5: percentage of queries where at least one relevant chunk is in the top 5."""
    client, retriever = _build_index(quality_docs, temp_dir)
    qrels = _get_qrels()

    hits = 0
    total = 0
    for query, relevant_ids in qrels.items():
        results = retriever.search(query, n_results=5)
        retrieved_ids = {r["id"] for r in results}
        if retrieved_ids & relevant_ids:
            hits += 1
        total += 1

    recall = hits / total if total > 0 else 0.0
    assert recall >= RECALL_AT_5_MIN, (
        f"Recall@5 = {recall:.3f} (threshold: {RECALL_AT_5_MIN})"
    )


@pytest.mark.slow
def test_rag_quality_mrr(quality_docs, temp_dir):
    """MRR: Mean Reciprocal Rank — how high the first relevant chunk ranks."""
    client, retriever = _build_index(quality_docs, temp_dir)
    qrels = _get_qrels()

    total_rr = 0.0
    total = 0
    for query, relevant_ids in qrels.items():
        results = retriever.search(query, n_results=5)
        rr = 0.0
        for rank, r in enumerate(results, start=1):
            if r["id"] in relevant_ids:
                rr = 1.0 / rank
                break
        total_rr += rr
        total += 1

    mrr = total_rr / total if total > 0 else 0.0
    assert mrr >= MRR_MIN, (
        f"MRR = {mrr:.3f} (threshold: {MRR_MIN})"
    )


@pytest.mark.slow
def test_rag_quality_ndcg_at_5(quality_docs, temp_dir):
    """NDCG@5: Normalized Discounted Cumulative Gain — ranking quality."""
    client, retriever = _build_index(quality_docs, temp_dir)
    qrels = _get_qrels()

    total_ndcg = 0.0
    total = 0
    for query, relevant_ids in qrels.items():
        results = retriever.search(query, n_results=5)
        retrieved_ids = [r["id"] for r in results]

        # DCG
        dcg = 0.0
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant_ids:
                dcg += 1.0 / math.log2(rank + 1)

        # IDCG (ideal: all relevant docs at the top)
        ideal_rel_count = min(len(relevant_ids), 5)
        idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_rel_count + 1))

        ndcg = dcg / idcg if idcg > 0 else 0.0
        total_ndcg += ndcg
        total += 1

    avg_ndcg = total_ndcg / total if total > 0 else 0.0
    assert avg_ndcg >= NDCG_AT_5_MIN, (
        f"NDCG@5 = {avg_ndcg:.3f} (threshold: {NDCG_AT_5_MIN})"
    )


@pytest.mark.slow
def test_rag_quality_report(quality_docs, temp_dir):
    """Print a full quality report for human inspection."""
    client, retriever = _build_index(quality_docs, temp_dir)
    qrels = _get_qrels()

    print("\n--- RAG Quality Report ---")
    for query, relevant_ids in qrels.items():
        results = retriever.search(query, n_results=5)
        print(f"\nQuery: {query}")
        print(f"Expected relevant chunks: {len(relevant_ids)}")
        for rank, r in enumerate(results, start=1):
            marker = "✓" if r["id"] in relevant_ids else " "
            text_preview = r["text"][:80].replace("\n", " ")
            print(f"  [{marker}] #{rank}: {r['source']} | {text_preview}...")

    # Compute overall metrics
    print("\n--- Metrics ---")
    hits = sum(
        1 for q, rels in qrels.items()
        if {r["id"] for r in retriever.search(q, n_results=5)} & rels
    )
    recall = hits / len(qrels) if qrels else 0
    print(f"Recall@5: {recall:.3f} (threshold: {RECALL_AT_5_MIN})")

    total_rr = sum(
        _first_relevant_rank(retriever.search(q, n_results=5), rels)
        for q, rels in qrels.items()
    )
    mrr = total_rr / len(qrels) if qrels else 0
    print(f"MRR: {mrr:.3f} (threshold: {MRR_MIN})")

    # All assertions pass if this test reaches here from the individual metric tests
    assert True


def _build_index(quality_docs, temp_dir):
    import shutil
    chroma_dir = temp_dir / "chroma_quality"
    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_or_create_collection("kb_quality")
    ingest_directory(quality_docs, collection)
    retriever = KBMetadata(client, collection_name="kb_quality")
    return client, retriever


def _get_qrels() -> dict[str, set[str]]:
    """Query relevance judgments: query -> set of relevant chunk IDs.

    Chunk IDs are in the format: {stem}_{chunk_index}_{file_hash[:8]}_{timestamp}
    We use filename-based matching since chunk IDs contain timestamps.
    Instead, we match by source filename substring.
    """
    return {
        # Exact match — keywords from python_intro
        "Who created the Python programming language?": {"python_intro"},
        "What is Python's design philosophy?": {"python_intro"},

        # Semantic match — synonyms
        "How do neural nets with multiple layers work?": {"machine_learning"},
        "What is the difference between labeled and unlabeled training?": {"machine_learning"},

        # Keyword + semantic
        "explain the greenhouse effect and its causes": {"climate_science"},
        "what causes climate warming?": {"climate_science"},

        # Cross-document (not applicable with this small set, but test one)
        "what are the specifications of the phone camera?": {"product_review"},

        # Hard negative — should return machine_learning, not climate_science
        "learning from data without explicit programming": {"machine_learning"},

        # Cooking domain
        "how should I season my food while cooking?": {"cooking_tips"},
        "what is mise en place?": {"cooking_tips"},

        # Product review
        "how much RAM does the Model X Pro have?": {"product_review"},
        "what is the battery capacity of the smartphone?": {"product_review"},
    }


def _first_relevant_rank(results: list[dict], relevant_source_substrings: set[str]) -> float:
    """Return 1/rank of first relevant result, or 0 if none."""
    for rank, r in enumerate(results, start=1):
        source = r.get("source", "")
        for rel_substring in relevant_source_substrings:
            if rel_substring in source:
                return 1.0 / rank
    return 0.0
```

- [ ] **Step 3: Add pytest marker for slow tests in pyproject.toml**

If `[tool.pytest.ini_options]` does not exist, append:
```toml
[tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
]
```

- [ ] **Step 4: Update _get_qrels to use source-based matching in assertions**

The `_get_qrels` returns source-substring sets, and the metric tests check `retrieved_ids & relevant_ids`. Since chunk IDs are dynamic (contain timestamps), update the metric tests to match against source substrings instead:

Rewrite `_get_qrels` to return list of source substrings, and update each metric function to check `"source"` field against substrings.

Actually, let me redesign the qrels approach to be simpler. Instead of matching by ID, match by source filename:

Replace the quality test file with:

```python
"""RAG quality gate — must pass before merging any RAG pipeline change.

Uses real BGE-M3 embedder and reranker. Marked @pytest.mark.slow.
"""

import math

import chromadb
import pytest

from personal_agent.kb.ingest import ingest_directory
from personal_agent.kb.retrieval import KBMetadata


# Quality thresholds
RECALL_AT_5_MIN = 0.80
MRR_MIN = 0.60
NDCG_AT_5_MIN = 0.70


@pytest.mark.slow
def test_rag_quality_recall_at_5(quality_docs, temp_dir):
    client, retriever = _build_index(quality_docs, temp_dir)
    qrels = _get_qrels()

    hits = 0
    total = 0
    for query, expected_sources in qrels.items():
        results = retriever.search(query, n_results=5)
        retrieved_sources = {r["source"] for r in results}
        if _any_source_matches(retrieved_sources, expected_sources):
            hits += 1
        total += 1

    recall = hits / total if total > 0 else 0.0
    assert recall >= RECALL_AT_5_MIN, (
        f"Recall@5 = {recall:.3f} (threshold: {RECALL_AT_5_MIN})"
    )


@pytest.mark.slow
def test_rag_quality_mrr(quality_docs, temp_dir):
    client, retriever = _build_index(quality_docs, temp_dir)
    qrels = _get_qrels()

    total_rr = 0.0
    total = 0
    for query, expected_sources in qrels.items():
        results = retriever.search(query, n_results=5)
        rr = _first_relevant_rank(results, expected_sources)
        total_rr += rr
        total += 1

    mrr = total_rr / total if total > 0 else 0.0
    assert mrr >= MRR_MIN, (
        f"MRR = {mrr:.3f} (threshold: {MRR_MIN})"
    )


@pytest.mark.slow
def test_rag_quality_ndcg_at_5(quality_docs, temp_dir):
    client, retriever = _build_index(quality_docs, temp_dir)
    qrels = _get_qrels()

    total_ndcg = 0.0
    total = 0
    for query, expected_sources in qrels.items():
        results = retriever.search(query, n_results=5)

        dcg = 0.0
        for rank, r in enumerate(results, start=1):
            if _source_matches(r["source"], expected_sources):
                dcg += 1.0 / math.log2(rank + 1)

        ideal_rel_count = min(len(expected_sources), 5)
        idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_rel_count + 1))

        ndcg = dcg / idcg if idcg > 0 else 0.0
        total_ndcg += ndcg
        total += 1

    avg_ndcg = total_ndcg / total if total > 0 else 0.0
    assert avg_ndcg >= NDCG_AT_5_MIN, (
        f"NDCG@5 = {avg_ndcg:.3f} (threshold: {NDCG_AT_5_MIN})"
    )


@pytest.mark.slow
def test_rag_quality_report(quality_docs, temp_dir):
    """Print a full quality report for human inspection."""
    client, retriever = _build_index(quality_docs, temp_dir)
    qrels = _get_qrels()

    print("\n--- RAG Quality Report ---")
    for query, expected_sources in qrels.items():
        results = retriever.search(query, n_results=5)
        print(f"\nQuery: {query}")
        print(f"Expected sources: {expected_sources}")
        for rank, r in enumerate(results, start=1):
            marker = "✓" if _source_matches(r["source"], expected_sources) else " "
            text_preview = r["text"][:80].replace("\n", " ")
            print(f"  [{marker}] #{rank}: {r['source']} | {text_preview}...")

    hits = sum(
        1 for q, exp in qrels.items()
        if _any_source_matches(
            {r["source"] for r in retriever.search(q, n_results=5)}, exp
        )
    )
    recall = hits / len(qrels) if qrels else 0
    print(f"\nRecall@5: {recall:.3f} (threshold: {RECALL_AT_5_MIN})")

    total_rr = sum(
        _first_relevant_rank(retriever.search(q, n_results=5), exp)
        for q, exp in qrels.items()
    )
    mrr = total_rr / len(qrels) if qrels else 0
    print(f"MRR: {mrr:.3f} (threshold: {MRR_MIN})")

    assert True


def _build_index(quality_docs, temp_dir):
    chroma_dir = temp_dir / "chroma_quality"
    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_or_create_collection("kb_quality")
    ingest_directory(quality_docs, collection)
    retriever = KBMetadata(client, collection_name="kb_quality")
    return client, retriever


def _get_qrels():
    """Query relevance judgments: query -> set of expected source filename substrings."""
    return {
        "Who created the Python programming language?": {"python_intro"},
        "What is Python's design philosophy?": {"python_intro"},
        "How do neural nets with multiple layers work?": {"machine_learning"},
        "What is the difference between labeled and unlabeled training?": {"machine_learning"},
        "explain the greenhouse effect and its causes": {"climate_science"},
        "what causes climate warming?": {"climate_science"},
        "what are the specifications of the phone camera?": {"product_review"},
        "learning from data without explicit programming": {"machine_learning"},
        "how should I season my food while cooking?": {"cooking_tips"},
        "what is mise en place?": {"cooking_tips"},
        "how much RAM does the Model X Pro have?": {"product_review"},
        "what is the battery capacity of the smartphone?": {"product_review"},
    }


def _source_matches(source: str, expected_substrings: set[str]) -> bool:
    """Check if a source path contains any of the expected substrings."""
    return any(sub in source for sub in expected_substrings)


def _any_source_matches(sources: set[str], expected_substrings: set[str]) -> bool:
    """Check if any source in the set matches expected substrings."""
    return any(_source_matches(s, expected_substrings) for s in sources)


def _first_relevant_rank(results: list[dict], expected_substrings: set[str]) -> float:
    """Return 1/rank of first relevant result, or 0 if none."""
    for rank, r in enumerate(results, start=1):
        if _source_matches(r.get("source", ""), expected_substrings):
            return 1.0 / rank
    return 0.0
```

- [ ] **Step 5: Run the quality tests**

Run:
```bash
cd /Users/cchhyy/Desktop/personal_agent && poetry run pytest tests/test_rag_quality.py -v -s
```
Expected: PASS with printed quality report (first run downloads models if not cached)

- [ ] **Step 6: Commit**

```bash
git add tests/test_rag_quality.py tests/conftest.py pyproject.toml
git commit -m "test: add RAG quality gate with recall, MRR, and NDCG metrics"
```

---

### Task 8: Final Verification

- [ ] **Step 1: Run full test suite (excluding slow)**

Run:
```bash
cd /Users/cchhyy/Desktop/personal_agent && poetry run pytest tests/ -v -m "not slow"
```
Expected: All tests PASS

- [ ] **Step 2: Run slow quality tests**

Run:
```bash
cd /Users/cchhyy/Desktop/personal_agent && poetry run pytest tests/test_rag_quality.py -v -s
```
Expected: Quality metrics all above thresholds

- [ ] **Step 3: Verify imports**

Run:
```bash
cd /Users/cchhyy/Desktop/personal_agent && poetry run python -c "
from personal_agent.kb.embed import Embedder
from personal_agent.kb.ingest import parse_file, chunk_text
from personal_agent.kb.retrieval import KBMetadata, rewrite_query, check_and_migrate_kb
from personal_agent.tools.kb_search import kb_search
print('All imports OK')
"
```
Expected: `All imports OK`

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "chore: finalize RAG pipeline upgrade"
```
