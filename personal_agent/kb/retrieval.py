import chromadb
from personal_agent.kb.embed import Embedder


class KBMetadata:
    def __init__(self, client: chromadb.PersistentClient, collection_name: str = "kb"):
        self.client = client
        self.collection_name = collection_name
        self._collection = None
        self._embedder = Embedder()

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

    def search(self, query: str, n_results: int = 5) -> list[dict]:
        results = self.collection.query(query_texts=[query], n_results=n_results)
        if not results["ids"] or not results["ids"][0]:
            return []

        output = []
        for i, doc_id in enumerate(results["ids"][0]):
            meta = {}
            if results["metadatas"] and results["metadatas"][0] and results["metadatas"][0][i] is not None:
                meta = results["metadatas"][0][i]
            output.append({
                "id": doc_id,
                "text": results["documents"][0][i] if results["documents"] else "",
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
            return len(existing["ids"])
        return 0


class _ChromaEmbeddingAdapter:
    """Adapts our Embedder to Chroma's EmbeddingFunction interface."""
    def __init__(self, embedder: Embedder):
        self._embedder = embedder

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self._embedder.embed(input)

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return self._embedder.embed(input)

    def embed_document(self, input: list[str]) -> list[list[float]]:
        return self._embedder.embed(input)

    def name(self) -> str:
        return "default"
