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
