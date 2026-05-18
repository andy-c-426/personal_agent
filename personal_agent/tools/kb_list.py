import json
from personal_agent.kb.retrieval import KBMetadata


def kb_list(retriever: KBMetadata) -> str:
    docs = retriever.list_documents()
    return json.dumps({"documents": docs, "count": len(docs)}, ensure_ascii=False)
