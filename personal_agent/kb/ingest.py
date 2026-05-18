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


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    source_meta: dict | None = None,
) -> list[Chunk]:
    meta = source_meta or {}
    paragraphs = text.split("\n\n")
    chunks = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Handle single paragraphs that exceed chunk_size
        if len(para) / 4 > chunk_size:
            if current:
                chunks.append(Chunk(text=current.strip(), metadata={**meta, "chunk_index": len(chunks)}))
                current = ""
            sub_chunks = _split_oversized(para, chunk_size)
            for sub in sub_chunks:
                chunks.append(Chunk(text=sub, metadata={**meta, "chunk_index": len(chunks)}))
            continue

        # Rough token estimate: chars / 4
        if len(current) / 4 + len(para) / 4 > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**meta, "chunk_index": len(chunks)}))
            overlap_chars = chunk_overlap * 4
            if len(current) > overlap_chars:
                current = current[-overlap_chars:] + "\n\n" + para
            else:
                current = para
        else:
            current = (current + "\n\n" + para).strip()

    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**meta, "chunk_index": len(chunks)}))

    return chunks


def _split_oversized(text: str, chunk_size: int) -> list[str]:
    """Split text that exceeds chunk_size into smaller pieces.

    First attempts to split by sentence boundaries, then groups sentences
    into chunk_size-sized groups. Falls back to character-level splitting
    for text without sentence breaks or for individual oversized sentences.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text)

    # If sentence splitting didn't break up the text, split by characters
    if len(sentences) <= 1:
        char_limit = chunk_size * 4
        return [text[i : i + char_limit] for i in range(0, len(text), char_limit) if text[i : i + char_limit]]

    result = []
    buf = ""
    for s in sentences:
        # A single sentence that is itself oversized -- flush buf, then char-split the sentence
        if len(s) / 4 > chunk_size:
            if buf:
                result.append(buf.strip())
                buf = ""
            char_limit = chunk_size * 4
            for i in range(0, len(s), char_limit):
                result.append(s[i : i + char_limit])
            continue

        if len(buf) / 4 + len(s) / 4 > chunk_size and buf:
            result.append(buf.strip())
            buf = s
        else:
            buf = (buf + " " + s).strip()

    if buf:
        result.append(buf.strip())

    return result


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
