import json
from personal_agent.kb.retrieval import KBMetadata, rewrite_query


def kb_search(query: str, retriever: KBMetadata, llm_client=None, model: str = "deepseek-chat") -> str:
    if llm_client is not None:
        query = rewrite_query(query, llm_client, model=model)
    results = retriever.search(query, n_results=5)
    if not results:
        return json.dumps({"results": [], "message": "No results found in knowledge base."})
    return json.dumps({"results": results}, ensure_ascii=False)
