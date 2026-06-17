# RAG Pipeline Improvement — Design Spec

## Overview

Upgrade the knowledge base retrieval pipeline from basic semantic search to a production-quality hybrid RAG system. Every component improves: embeddings, chunking, retrieval strategy, reranking, and query preprocessing. A quality gate ensures regressions are caught before they land.

## Architecture

```
User Query
    │
    ▼
Query Rewriting (LLM)
    │
    ▼
KBMetadata.search()
    ├── Dense Search (BGE-M3 → Chroma vector)  ─┐
    ├── Sparse Search (BGE-M3 sparse + BM25)    ─┤── RRF Fusion ──► Reranker ──► Top-5
    └── (top-20 candidates from each)            ─┘
```

```
Ingestion Pipeline
    │
    ▼
Parse File → Semantic Chunking (sentence-boundary + embedding similarity) → BGE-M3 Embed → Chroma
```

## Section 1: Embeddings Upgrade (`kb/embed.py`)

Replace `all-MiniLM-L6-v2` (384-dim) with `BAAI/bge-m3` (1024-dim, dense + sparse).

BGE-M3 outputs both dense vectors and sparse lexical weights in one call. Dense for semantic matching, sparse for keyword matching — no separate BM25 index needed.

**Changes:**
- `_MODEL_NAME` → `"BAAI/bge-m3"`
- `embed()` returns `list[list[float]]` (dense, 1024-dim each) — same signature
- New `embed_sparse(texts)` method returns sparse lexical weights
- Model ~2GB, cached on first load (same singleton pattern as today)

**Breaking:** Existing Chroma collections use 384-dim vectors. Migration handled in Section 5.

## Section 2: Hybrid Search + Reranking (`kb/retrieval.py`)

**Hybrid retrieval** using BGE-M3's combined output:

1. Dense retrieval — query embedding → Chroma vector search, top-20 candidates
2. Sparse retrieval — query sparse weights → BM25-style keyword scoring against document text, top-20
3. Fusion — combine dense and sparse scores via reciprocal rank fusion (RRF), merged top-20
4. Rerank — cross-encoder scores each (query, chunk) pair, top-5 survive

**Reranker:** `BAAI/bge-reranker-v2-m3` (~1GB), loaded once and cached like the embedder.

**Changes to `KBMetadata`:**
- `search()` signature unchanged — `(query, n_results=5)` — internal pipeline expands
- New methods: `_dense_search`, `_sparse_search`, `_rrf_fuse`, `_rerank`
- Reranker stored as module-level singleton

**Sparse handling:** Chroma does not natively store sparse vectors. Sparse scoring runs as a BM25-style keyword match: BGE-M3 sparse weights provide per-token importance scores for the query, scored against an in-memory dictionary of per-document token frequencies that's maintained alongside the Chroma collection.

## Section 3: Semantic Chunking (`kb/ingest.py`)

Replace the current paragraph-based chunker with a sentence-boundary-aware semantic chunker.

**Algorithm:**
1. Split text into sentences (via `re.split` with sentence boundary patterns)
2. Encode each sentence with the embedder
3. Walk sentences in order, merging consecutive ones while the running group's semantic coherence stays above a similarity threshold (cosine similarity between group embedding and next sentence embedding)
4. When similarity drops below threshold → split, start new group
5. Clamp groups between ~256 and ~1024 tokens by adjusting the threshold

**Metadata enrichment:** Each chunk stores the nearest heading it falls under, extracted from markdown `#` patterns or PDF font-size heuristics.

**Changes:**
- `chunk_text()` rewritten, same signature — callers unchanged
- `_split_oversized()` removed (semantic chunker handles long text differently)
- New internal `_get_headings(text)` for markdown heading extraction

## Section 4: Query Rewriting (`kb/retrieval.py`)

Before retrieval, the raw user query goes through a single-turn LLM rewrite to improve retrieval quality. Uses the agent's own model (DeepSeek) — no new dependency.

**Prompt:**
```
System: "Rewrite the user query into a self-contained search query optimized for semantic retrieval. Include key terms, expand abbreviations, and add relevant context. Output only the rewritten query."
```

**Where it fits:** Hooked into the `kb_search` tool, before `KBMetadata.search()`. Adds ~300-500ms. Shown in debug mode only. Falls back to raw query on failure.

**Changes:**
- Add `rewrite_query(query: str, client) -> str` to `kb/retrieval.py`
- `kb_search` tool calls it before retrieval when an LLM client is available

## Section 5: Migration & Test Updates

**Auto-migration:** On startup, check if `kb` collection exists and has 1024-dim vectors. If mismatched → log warning, delete old collection, re-ingest all files in `KB_DIR`. Re-ingestion is silent — files are local, KB is rebuildable.

**Test dimension updates:** Any test with hardcoded `384` → `1024`. Dummy embedding functions in test fixtures → `[0.1] * 1024`. `test_ingest.py` chunking tests rewritten for semantic chunker behavior.

## Section 6: RAG Quality Gate (`tests/test_rag_quality.py`)

A standalone quality test suite that gates any RAG pipeline change. Must pass before merging.

**Test dataset:** 5-8 hand-crafted documents seeded in a temp KB, with 15-20 query/expected-chunk pairs covering:

- Exact match — query uses words directly in the target chunk
- Semantic match — query uses synonyms or rephrased concepts
- Keyword + semantic — query needs both keyword overlap and conceptual similarity
- Cross-document — answer requires chunks from different documents
- Hard negatives — a similar-but-wrong chunk exists that shouldn't rank above the right one

**Metrics:**

| Metric | What it catches | Threshold |
|--------|----------------|-----------|
| Recall@5 | Are relevant chunks in the top 5? | >= 0.80 |
| MRR | Does the best chunk rank first? | >= 0.60 |
| NDCG@5 | Are chunks in the right order? | >= 0.70 |

**Execution:**
- Uses the real BGE-M3 embedder + reranker (no mocks) — this is a quality test, not a unit test
- Marked `@pytest.mark.slow` so it can be skipped during fast dev cycles
- Run in CI or as a pre-commit gate before RAG changes land
- Outputs a report: metric → score → pass/fail against threshold

**Regression detection:** Store baseline scores. If metrics drop below thresholds, the test fails with a message like "RAG quality regression — recall@5 dropped from 0.85 to 0.72."

**Dependencies:** No new dev dependency. Metrics computed in pure Python.

## Tech Stack Changes

- **Add:** `BAAI/bge-m3` (replaces `all-MiniLM-L6-v2` as embedder)
- **Add:** `BAAI/bge-reranker-v2-m3` (new — cross-encoder for reranking)
- **Remove:** `all-MiniLM-L6-v2` (no longer used)
- **Unchanged:** Chroma, sentence-transformers, pdfplumber, python-docx, DeepSeek API

## Files Changed

| File | Change |
|------|--------|
| `personal_agent/kb/embed.py` | New model, new `embed_sparse()` method |
| `personal_agent/kb/ingest.py` | Rewrite `chunk_text()` for semantic chunking |
| `personal_agent/kb/retrieval.py` | Hybrid search, reranker, query rewriting |
| `personal_agent/tools/kb_search.py` | Wire query rewrite before search |
| `personal_agent/cli/app.py` | Auto-migration check on startup |
| `tests/test_embed.py` | Update dimensions, add sparse tests |
| `tests/test_ingest.py` | Rewrite chunking tests for semantic chunker |
| `tests/test_retrieval.py` | Add hybrid search, rerank, query rewrite tests |
| `tests/test_rag_quality.py` | **New** — quality gate test suite |
| `pyproject.toml` | Add pytest marker for slow tests |
