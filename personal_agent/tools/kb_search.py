import json
from personal_agent.kb.retrieval import KBMetadata, rewrite_query


def kb_search(
    query: str,
    retriever: KBMetadata,
    llm_client=None,
    model: str = "deepseek-chat",
    top_k: int = 5,
    use_reranker: bool = True,
    use_rewrite: bool = True,
    debug: bool = False,
) -> str:
    if llm_client is not None and use_rewrite:
        query = rewrite_query(query, llm_client, model=model)

    if debug:
        debug_info = retriever.search_debug(query, n_results=top_k, use_reranker=use_reranker)
        return json.dumps(debug_info, ensure_ascii=False)

    results = retriever.search(query, n_results=top_k, use_reranker=use_reranker)
    if not results:
        return json.dumps({"results": [], "message": "No results found in knowledge base."})

    formatted = []
    for r in results:
        citation = r["filename"]
        if r["heading"]:
            citation += f" > {r['heading']}"
        citation += f" (chunk {r['chunk_index']})"
        formatted.append({
            "text": r["text"],
            "source": r["source"],
            "citation": citation,
        })
    return json.dumps({"results": formatted}, ensure_ascii=False)
