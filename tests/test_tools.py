import json
import chromadb
from unittest.mock import MagicMock
from personal_agent.tools.kb_search import kb_search
from personal_agent.tools.web_search import web_search
from personal_agent.tools.kb_ingest import kb_ingest
from personal_agent.tools.kb_list import kb_list
from personal_agent.tools.kb_remove import kb_remove
from personal_agent.kb.retrieval import KBMetadata
from chromadb.api.types import EmbeddingFunction


class _DummyEF(EmbeddingFunction):
    def __call__(self, input):
        return [[0.1] * 384 for _ in input]


def _make_retriever(temp_dir, name="test_kbm"):
    client = chromadb.PersistentClient(path=str(temp_dir / f"chroma_{name}"))
    client.get_or_create_collection(name, embedding_function=_DummyEF())
    return KBMetadata(client, collection_name=name)


# KB Search tests
def test_kb_search_returns_results(temp_dir, sample_md_file):
    from personal_agent.kb.ingest import ingest_file
    retriever = _make_retriever(temp_dir, "kbs1")
    ingest_file(sample_md_file, retriever.collection)
    result = kb_search("test", retriever=retriever)
    data = json.loads(result)
    assert "results" in data


def test_kb_search_empty(temp_dir):
    retriever = _make_retriever(temp_dir, "kbs2")
    result = kb_search("nothing", retriever=retriever)
    data = json.loads(result)
    assert data["results"] == []


# Web Search tests
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


# KB Management tests
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
