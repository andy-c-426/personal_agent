import math
import re
from collections import defaultdict
from pathlib import Path

import chromadb
from sentence_transformers import CrossEncoder

from personal_agent.kb.embed import Embedder


_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
_reranker: CrossEncoder | None = None


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(_RERANKER_MODEL)
    return _reranker


class _SparseIndex:
    """In-memory BM25-style index for keyword search."""

    def __init__(self):
        self._docs: dict[str, dict[str, int]] = {}  # doc_id -> {token -> tf}
        self._df: dict[str, int] = defaultdict(int)  # document frequency
        self._total_docs = 0
        self._avg_dl = 0.0
        self._total_terms = 0

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r'[a-zA-Z0-9]+', text.lower())

    def add(self, doc_id: str, text: str) -> None:
        tokens = self._tokenize(text)
        tf = defaultdict(int)
        for t in tokens:
            tf[t] += 1
        self._docs[doc_id] = dict(tf)
        for t in set(tokens):
            self._df[t] += 1
        self._total_docs += 1
        self._total_terms += len(tokens)
        self._avg_dl = self._total_terms / self._total_docs

    def remove(self, doc_id: str) -> None:
        if doc_id not in self._docs:
            return
        for t in self._docs[doc_id]:
            self._df[t] -= 1
            if self._df[t] <= 0:
                del self._df[t]
        self._total_terms -= sum(self._docs[doc_id].values())
        self._total_docs -= 1
        self._avg_dl = self._total_terms / self._total_docs if self._total_docs > 0 else 0.0
        del self._docs[doc_id]

    def search(self, query_weights: dict[str, float], top_k: int = 20) -> list[tuple[str, float]]:
        k1 = 1.5
        b = 0.75
        scores = []
        for doc_id, tf in self._docs.items():
            doc_len = sum(tf.values())
            score = 0.0
            for token, q_weight in query_weights.items():
                if token not in tf:
                    continue
                df = self._df.get(token, 0)
                if df == 0:
                    continue
                idf = math.log((self._total_docs - df + 0.5) / (df + 0.5) + 1.0)
                tf_norm = (tf[token] * (k1 + 1)) / (tf[token] + k1 * (1 - b + b * doc_len / self._avg_dl))
                score += q_weight * idf * tf_norm
            if score > 0:
                scores.append((doc_id, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def count(self) -> int:
        return self._total_docs


class KBMetadata:
    def __init__(self, client: chromadb.PersistentClient, collection_name: str = "kb"):
        self.client = client
        self.collection_name = collection_name
        self._collection = None
        self._embedder = Embedder()
        self._sparse_index: _SparseIndex | None = None

    @property
    def collection(self) -> chromadb.Collection:
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                self.collection_name,
                embedding_function=_ChromaEmbeddingAdapter(self._embedder),
            )
        return self._collection

    @property
    def document_count(self) -> int:
        return self.collection.count()

    def _ensure_sparse_index(self) -> _SparseIndex:
        if self._sparse_index is not None:
            return self._sparse_index
        self._sparse_index = _SparseIndex()
        existing = self.collection.get(include=["documents"])
        if existing["ids"] and existing["documents"]:
            for doc_id, doc_text in zip(existing["ids"], existing["documents"]):
                if doc_text:
                    self._sparse_index.add(doc_id, doc_text)
        return self._sparse_index

    def _dense_search(self, query: str, n_results: int = 20) -> list[tuple[str, float]]:
        results = self.collection.query(query_texts=[query], n_results=n_results)
        if not results["ids"] or not results["ids"][0]:
            return []
        pairs = []
        distances = results.get("distances", [[]])
        for i, doc_id in enumerate(results["ids"][0]):
            dist = distances[0][i] if distances[0] and i < len(distances[0]) else 0.0
            score = 1.0 - dist if dist else 1.0
            pairs.append((doc_id, score))
        return pairs

    def _sparse_search(self, query: str, n_results: int = 20) -> list[tuple[str, float]]:
        sparse_index = self._ensure_sparse_index()
        if sparse_index.count() == 0:
            return []
        sparse_output = self._embedder.embed_sparse(query)
        if not sparse_output or not sparse_output[0]:
            return []
        token_weights: dict[str, float] = {}
        for token_id, weight in sparse_output[0].items():
            token = self._embedder.model.tokenizer.decode([int(token_id)]).strip().lower()
            if token and len(token) > 1:
                token_weights[token] = weight
        return sparse_index.search(token_weights, top_k=n_results)

    @staticmethod
    def _rrf_fuse(
        dense_results: list[tuple[str, float]],
        sparse_results: list[tuple[str, float]],
        k: int = 60,
        top_k: int = 20,
    ) -> list[str]:
        scores: dict[str, float] = {}
        for rank, (doc_id, _) in enumerate(dense_results):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
        for rank, (doc_id, _) in enumerate(sparse_results):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in sorted_docs[:top_k]]

    def _rerank(self, query: str, doc_ids: list[str], top_k: int = 5) -> list[str]:
        if not doc_ids:
            return []
        results = self.collection.get(ids=doc_ids, include=["documents"])
        doc_map = {}
        if results["ids"] and results["documents"]:
            for doc_id, doc_text in zip(results["ids"], results["documents"]):
                doc_map[doc_id] = doc_text or ""
        pairs = [(query, doc_map.get(doc_id, "")) for doc_id in doc_ids]
        reranker = _get_reranker()
        scores = reranker.predict(pairs).tolist()
        if not isinstance(scores, list):
            scores = [scores]
        scored = list(zip(doc_ids, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in scored[:top_k]]

    def search(self, query: str, n_results: int = 5) -> list[dict]:
        dense_results = self._dense_search(query, n_results=20)
        sparse_results = self._sparse_search(query, n_results=20)
        merged_ids = self._rrf_fuse(dense_results, sparse_results, top_k=20)
        reranked_ids = self._rerank(query, merged_ids, top_k=n_results)
        if not reranked_ids:
            return []
        results = self.collection.get(ids=reranked_ids, include=["documents", "metadatas"])
        output = []
        for doc_id in reranked_ids:
            idx = results["ids"].index(doc_id) if doc_id in results["ids"] else -1
            if idx < 0:
                continue
            meta = {}
            if results["metadatas"] and results["metadatas"][idx] is not None:
                meta = results["metadatas"][idx]
            output.append({
                "id": doc_id,
                "text": results["documents"][idx] if results["documents"] else "",
                "source": meta.get("source", ""),
                "filename": meta.get("filename", ""),
            })
        return output

    def list_documents(self) -> list[dict]:
        all_data = self.collection.get(include=["metadatas"])
        seen = {}
        docs = []
        if all_data["metadatas"]:
            for meta in all_data["metadatas"]:
                source = meta.get("source", "")
                if source and source not in seen:
                    seen[source] = True
                    docs.append({"source": source, "filename": meta.get("filename", "")})
        return docs

    def remove_document(self, source_path: str) -> int:
        if not source_path:
            return 0
        existing = self.collection.get(where={"source": source_path})
        if existing["ids"]:
            self.collection.delete(ids=existing["ids"])
            if self._sparse_index:
                for doc_id in existing["ids"]:
                    self._sparse_index.remove(doc_id)
            return len(existing["ids"])
        return 0

    def add_to_sparse_index(self, doc_id: str, text: str) -> None:
        if self._sparse_index is not None:
            self._sparse_index.add(doc_id, text)


def rewrite_query(query: str, client, model: str = "deepseek-chat") -> str:
    """Rewrite a user query for better retrieval. Falls back to original on failure."""
    if client is None:
        return query
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": (
                    "Rewrite the user query into a self-contained search query optimized "
                    "for semantic retrieval. Include key terms, expand abbreviations, and "
                    "add relevant context. Output only the rewritten query."
                )},
                {"role": "user", "content": query},
            ],
            temperature=0.3,
            max_tokens=100,
        )
        rewritten = response.choices[0].message.content.strip()
        return rewritten if rewritten else query
    except Exception:
        return query


class _ChromaEmbeddingAdapter:
    def __init__(self, embedder: Embedder):
        self._embedder = embedder

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self._embedder.embed(input)

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return self._embedder.embed(input)

    def embed_document(self, input: list[str]) -> list[list[float]]:
        return self._embedder.embed(input)

    def name(self) -> str:
        return "BAAI/bge-m3"
