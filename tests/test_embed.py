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
    assert len(result[0]) > 0
    # Keys should be str, values should be native float
    for k, v in result[0].items():
        assert isinstance(k, str)
        assert isinstance(v, float)


def test_embed_sparse_single_string():
    embedder = Embedder()
    result = embedder.embed_sparse("hello world")
    assert len(result) == 1
    assert isinstance(result[0], dict)
    for k, v in result[0].items():
        assert isinstance(k, str)
        assert isinstance(v, float)
