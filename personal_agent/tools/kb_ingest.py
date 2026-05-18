import json
from pathlib import Path
from personal_agent.kb.ingest import ingest_file, ingest_directory
from personal_agent.kb.retrieval import KBMetadata


def kb_ingest(path: str, retriever: KBMetadata) -> str:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return json.dumps({"error": f"Path not found: {path}"})
    try:
        if p.is_file():
            ids = ingest_file(p, retriever.collection)
        else:
            ids, errors = ingest_directory(p, retriever.collection)
        return json.dumps({"ingested": len(ids), "path": str(p)})
    except Exception as e:
        return json.dumps({"error": str(e)})
