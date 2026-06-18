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
    # Use longer, clearly different topics to trigger semantic split
    text = (
        "The Python programming language is widely used for data science and machine learning. "
        "It has many libraries including numpy, pandas, and scikit-learn for statistical analysis. "
        "Python's simple syntax and dynamic typing make it ideal for rapid prototyping in AI research.\n\n"
        "Football is a popular sport played with 11 players on each team. "
        "The World Cup is held every four years and draws global audiences from hundreds of countries. "
        "Teams compete in qualifying rounds before the final tournament begins."
    )
    chunks = chunk_text(text, chunk_size=500)
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
    # Same topic — all content preserved
    assert 1 <= len(chunks) <= 3
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
