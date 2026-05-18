import json
from personal_agent.kb.retrieval import KBMetadata


def kb_search(query: str, retriever: KBMetadata) -> str:
    results = retriever.search(query, n_results=5)
    if not results:
        return json.dumps({"results": [], "message": "No results found in knowledge base."})
    return json.dumps({"results": results}, ensure_ascii=False)
