import chromadb
import pytest
from chromadb.api.types import EmbeddingFunction
from personal_agent.kb.retrieval import KBMetadata, rewrite_query
from unittest.mock import MagicMock, patch


class _DummyEF(EmbeddingFunction):
    def __call__(self, input):
        return [[0.1] * 1024 for _ in input]


def _make_client(temp_dir, name):
    return chromadb.PersistentClient(path=str(temp_dir / name))


def _add_docs_to_collection(client, name, docs):
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


@pytest.mark.slow
def test_reranker_loads():
    from personal_agent.kb.retrieval import _get_reranker
    reranker = _get_reranker()
    assert reranker is not None


@pytest.mark.slow
def test_rerank_scores_pairs():
    from personal_agent.kb.retrieval import _get_reranker
    reranker = _get_reranker()
    scores = reranker.predict([
        ("machine learning", "Machine learning is a subset of artificial intelligence."),
        ("machine learning", "The football match was exciting."),
    ]).tolist()
    assert len(scores) == 2
    assert isinstance(scores[0], float)
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
    assert result == "test query"
