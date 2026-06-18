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

    def embed_sparse(self, texts: str | list[str]) -> list[dict[str, float]]:
        """Return sparse lexical weights (token -> weight per text)."""
        if isinstance(texts, str):
            texts = [texts]
        output = self.model.encode(texts, return_dense=False, return_sparse=True)
        return [{str(k): float(v) for k, v in w.items()} for w in output["lexical_weights"]]
