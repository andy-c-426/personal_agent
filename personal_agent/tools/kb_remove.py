import json
from pathlib import Path
from personal_agent.kb.retrieval import KBMetadata


def kb_remove(source: str, retriever: KBMetadata) -> str:
    resolved = str(Path(source).expanduser().resolve())
    count = retriever.remove_document(resolved)
    return json.dumps({"removed": count, "source": resolved})
