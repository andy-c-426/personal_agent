import json
from tavily import TavilyClient


def web_search(query: str, client: TavilyClient) -> str:
    try:
        response = client.search(query, search_depth="advanced", max_results=5)
        results = response.get("results", [])
        formatted = [
            {"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")}
            for r in results
        ]
        return json.dumps({"results": formatted}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e), "results": []})
