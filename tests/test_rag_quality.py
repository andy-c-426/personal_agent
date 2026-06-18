"""RAG quality gate — must pass before merging any RAG pipeline change.

Uses real BGE-M3 embedder and reranker. Marked @pytest.mark.slow.
"""

import math
from pathlib import Path

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
    """MRR: Mean Reciprocal Rank — how high the first relevant chunk ranks."""
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
    """NDCG@5: Normalized Discounted Cumulative Gain — ranking quality."""
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
    import tempfile as tf
    chroma_dir = Path(tf.mkdtemp()) / "chroma_quality"
    chroma_dir.mkdir(parents=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))
    from personal_agent.kb.retrieval import _ChromaEmbeddingAdapter
    from personal_agent.kb.embed import Embedder
    collection = client.get_or_create_collection(
        "kb_quality",
        embedding_function=_ChromaEmbeddingAdapter(Embedder()),
    )
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
