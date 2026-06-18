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


_ABBREVIATIONS = {"Mr", "Ms", "Mrs", "Dr", "Prof", "Inc", "Ltd", "Jr", "Sr",
                  "e.g", "i.e", "etc", "vs", "St", "Ave", "Rd", "Blvd",
                  "Jan", "Feb", "Mar", "Apr", "Jun", "Jul", "Aug", "Sep",
                  "Oct", "Nov", "Dec", "U.S", "U.K", "E.U", "No", "Vol",
                  "approx", "dept", "est", "govt"}


def _sentences(text: str) -> list[tuple[str, int]]:
    """Split text into sentences. Returns list of (sentence, char_position)."""
    pattern = r'(?<=[.!?])\s+'
    parts = re.split(pattern, text)
    result = []
    pos = 0
    for part in parts:
        stripped = part.strip()
        if not stripped:
            pos += len(part) + 1
            continue
        # Check if this split was on an abbreviation
        tokens = stripped.rsplit(maxsplit=1)
        if tokens and tokens[-1].rstrip('.') in _ABBREVIATIONS:
            # Merge with next part instead of splitting here
            pos += len(part) + 1
            continue
        result.append((stripped, pos))
        pos += len(part) + 1
    return result


def _token_estimate(text: str) -> int:
    return len(text) // 4


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    source_meta: dict | None = None,
    embedder: Embedder | None = None,
) -> list[Chunk]:
    """Semantic chunking: sentence-boundary-aware, merges by embedding similarity.

    chunk_size drives both min (~chunk_size/2) and max (~chunk_size*2) token bounds.
    Pass embedder=None to use the module-level cached embedder.
    """
    meta = source_meta or {}
    headings = _extract_headings(text)
    sent_positions = _sentences(text)
    sentences = [s for s, _ in sent_positions]
    positions = [p for _, p in sent_positions]
    if not sentences:
        return []

    emb = embedder or _get_embedder()

    # Encode all sentences once
    sent_embeddings = emb.embed(sentences)

    min_tokens = max(64, chunk_size // 2)
    max_tokens = chunk_size * 2

    chunks = []
    group_indices = [0]  # indices into sentences list
    centroid = [float(v) for v in sent_embeddings[0]]  # Running centroid
    centroid_count = 1

    def _flush(group_idx_list: list[int], next_idx: int | None = None) -> None:
        """Flush current group as a chunk, optionally starting overlap."""
        nonlocal centroid, centroid_count
        chunk_text_str = " ".join(sentences[i] for i in group_idx_list)
        chunk_meta = {**meta, "chunk_index": len(chunks)}
        heading = _find_nearest_heading(positions[group_idx_list[0]], headings)
        if heading:
            chunk_meta["heading"] = heading
        chunks.append(Chunk(text=chunk_text_str, metadata=chunk_meta))

        if next_idx is not None and chunk_overlap > 0 and len(group_idx_list) > 1:
            # Overlap: keep last sentence
            overlap_idx = group_idx_list[-1]
            group_idx_list.clear()
            group_idx_list.append(overlap_idx)
            group_idx_list.append(next_idx)
            centroid = [float(v) for v in sent_embeddings[overlap_idx]]
            centroid_count = 1
            # Add the next sentence to centroid
            for dim in range(len(centroid)):
                centroid[dim] = (centroid[dim] * centroid_count + sent_embeddings[next_idx][dim]) / (centroid_count + 1)
            centroid_count += 1
        elif next_idx is not None:
            group_idx_list.clear()
            group_idx_list.append(next_idx)
            centroid = [float(v) for v in sent_embeddings[next_idx]]
            centroid_count = 1

    for i in range(1, len(sentences)):
        current_tokens = _token_estimate(" ".join(sentences[j] for j in group_indices))
        next_emb = sent_embeddings[i]

        # Cosine similarity between running centroid and next sentence
        dot = sum(a * b for a, b in zip(centroid, next_emb))
        norm_a = sum(a * a for a in centroid) ** 0.5
        norm_b = sum(b * b for b in next_emb) ** 0.5
        sim = dot / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0.0

        # Threshold ramps up as we approach max_tokens
        if current_tokens < min_tokens:
            threshold = 0.45
        elif current_tokens >= max_tokens:
            threshold = 1.0  # Force split
        else:
            # Linear ramp from 0.45 to 0.7 between min and max
            ratio = (current_tokens - min_tokens) / (max_tokens - min_tokens)
            threshold = 0.45 + ratio * 0.25

        if sim >= threshold and current_tokens < max_tokens:
            group_indices.append(i)
            # Update running centroid
            for dim in range(len(centroid)):
                centroid[dim] = (centroid[dim] * centroid_count + sent_embeddings[i][dim]) / (centroid_count + 1)
            centroid_count += 1
        else:
            _flush(group_indices, next_idx=i)

    # Flush final group
    if group_indices:
        chunk_text_str = " ".join(sentences[i] for i in group_indices)
        chunk_meta = {**meta, "chunk_index": len(chunks)}
        heading = _find_nearest_heading(positions[group_indices[0]], headings)
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
