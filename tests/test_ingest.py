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
        assert len(chunk.text) > 0


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
    old_results = collection.get(ids=first_ids)
    assert len(old_results["ids"]) == 0


def _dummy_ef():
    from chromadb.api.types import EmbeddingFunction
    class DummyEF(EmbeddingFunction):
        def __call__(self, input):
            return [[0.1] * 384 for _ in input]
    return DummyEF()
