import hashlib
import logging
import re
import time
from pathlib import Path
from dataclasses import dataclass

from personal_agent.kb.embed import Embedder


logger = logging.getLogger(__name__)

_embedder: Embedder | None = None


def _get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


@dataclass
class Chunk:
    text: str
    metadata: dict


def parse_file(file_path: Path) -> tuple[str, dict]:
    meta = {"source": str(file_path), "filename": file_path.name}
    suffix = file_path.suffix.lower()

    if suffix == ".md":
        meta["type"] = "markdown"
        return file_path.read_text(encoding="utf-8"), meta
    elif suffix == ".pdf":
        meta["type"] = "pdf"
        import pdfplumber
        with pdfplumber.open(file_path) as pdf:
            text = "\n".join(p.extract_text() or "" for p in pdf.pages)
        return text, meta
    elif suffix in (".docx", ".doc"):
        meta["type"] = "docx"
        from docx import Document
        doc = Document(str(file_path))
        text = "\n".join(p.text for p in doc.paragraphs)
        return text, meta
    else:
        meta["type"] = "text"
        return file_path.read_text(encoding="utf-8"), meta


def _extract_headings(text: str) -> list[tuple[int, int, str]]:
    """Extract markdown headings. Returns list of (char_position, level, text)."""
    headings = []
    for m in re.finditer(r'^(#{1,6})\s+(.+)$', text, re.MULTILINE):
        level = len(m.group(1))
        headings.append((m.start(), level, m.group(2).strip()))
    return headings


def _find_nearest_heading(pos: int, headings: list[tuple[int, int, str]]) -> str | None:
    """Return the nearest heading text that precedes pos, or None."""
    best = None
    for h_pos, _level, h_text in headings:
        if h_pos <= pos:
            best = h_text
        else:
            break
    return best


def _sentences(text: str) -> list[str]:
    """Split text into sentences using boundary patterns."""
    raw = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in raw if s.strip()]


def _token_estimate(text: str) -> int:
    return len(text) // 4


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    source_meta: dict | None = None,
) -> list[Chunk]:
    """Semantic chunking: sentence-boundary-aware, merges by embedding similarity."""
    meta = source_meta or {}
    headings = _extract_headings(text)
    sentences = _sentences(text)
    if not sentences:
        return []

    embedder = _get_embedder()

    # Encode each sentence for similarity checking
    sent_embeddings = embedder.embed(sentences)

    chunks = []
    group = [sentences[0]]
    group_pos = 0  # char offset of first sentence in current group

    # Pre-compute absolute character positions for each sentence in the original text
    char_positions = []
    search_start = 0
    for s in sentences:
        pos = text.find(s, search_start)
        if pos == -1:
            pos = search_start
        char_positions.append(pos)
        search_start = pos + len(s)

    for i in range(1, len(sentences)):
        current_sent = sentences[i]
        combined_tokens = _token_estimate(" ".join(group))

        # Compute cosine similarity between group centroid and next sentence
        group_emb = embedder.embed(" ".join(group))[0]
        next_emb = sent_embeddings[i]

        dot = sum(a * b for a, b in zip(group_emb, next_emb))
        norm_a = sum(a * a for a in group_emb) ** 0.5
        norm_b = sum(b * b for b in next_emb) ** 0.5
        sim = dot / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0.0

        # Determine threshold: tighter when approaching max chunk size
        if combined_tokens < 256:
            threshold = 0.5
        elif combined_tokens > 1024:
            threshold = 0.9  # Force split
        else:
            threshold = 0.5

        if sim >= threshold and combined_tokens < 1024:
            group.append(current_sent)
        else:
            # Flush current group
            chunk_text_str = " ".join(group)
            chunk_meta = {**meta, "chunk_index": len(chunks)}
            heading = _find_nearest_heading(char_positions[group_pos], headings)
            if heading:
                chunk_meta["heading"] = heading
            chunks.append(Chunk(text=chunk_text_str, metadata=chunk_meta))

            # Start new group with overlap
            if chunk_overlap > 0 and len(group) > 1:
                group = group[-1:] + [current_sent]
                group_pos = i - 1
            else:
                group = [current_sent]
                group_pos = i

    # Flush final group
    if group:
        chunk_text_str = " ".join(group)
        chunk_meta = {**meta, "chunk_index": len(chunks)}
        heading = _find_nearest_heading(char_positions[group_pos], headings)
        if heading:
            chunk_meta["heading"] = heading
        chunks.append(Chunk(text=chunk_text_str, metadata=chunk_meta))

    return chunks


def _file_hash(file_path: Path) -> str:
    return hashlib.md5(file_path.read_bytes()).hexdigest()


def _delete_existing(file_path: Path, collection):
    existing = collection.get(where={"source": str(file_path)})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])


def ingest_file(file_path: Path, collection) -> list[str]:
    file_path = Path(file_path)
    _delete_existing(file_path, collection)

    text, meta = parse_file(file_path)
    file_hash = _file_hash(file_path)
    meta["file_hash"] = file_hash
    chunks = chunk_text(text, source_meta=meta)

    if not chunks:
        return []

    ids = []
    documents = []
    metadatas = []
    embeddings = []

    embedder = _get_embedder()

    for c in chunks:
        chunk_id = f"{file_path.stem}_{c.metadata['chunk_index']}_{file_hash[:8]}_{time.time_ns()}"
        ids.append(chunk_id)
        documents.append(c.text)
        metadatas.append(c.metadata)
        embeddings.append(embedder.embed(c.text)[0])

    collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    return ids


def ingest_directory(dir_path: Path, collection) -> tuple[list[str], list[str]]:
    dir_path = Path(dir_path)
    all_ids = []
    errors = []
    for p in dir_path.rglob("*"):
        if p.is_file() and not p.name.startswith("."):
            try:
                ids = ingest_file(p, collection)
                all_ids.extend(ids)
            except Exception as e:
                msg = f"Failed to ingest {p}: {e}"
                logger.warning(msg)
                errors.append(msg)
    return all_ids, errors
