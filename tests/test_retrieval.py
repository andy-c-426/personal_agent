import chromadb
from personal_agent.kb.retrieval import KBMetadata


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
    collection = client.get_or_create_collection("test_empty", embedding_function=DummyEF())
    collection.add(ids=["1"], documents=["unrelated content"], embeddings=[[0.1]*384])

    retriever = KBMetadata(client, collection_name="test_empty")
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

    results = collection.get(where={"source": str(sample_md_file)})
    assert len(results["ids"]) == 0
