"""Chunking strategies. Level 1 uses fixed-size chunks."""

from dataclasses import dataclass

from agentic_jev_rag.corpus import Document


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    collection: str
    language: str
    text: str


def fixed_size_chunks(document: Document, size: int, overlap: int) -> list[Chunk]:
    """Split a document into windows of `size` characters, each overlapping the last by `overlap`."""
    if not 0 <= overlap < size:
        raise ValueError(f"overlap must be in [0, size); got size={size}, overlap={overlap}")
    step = size - overlap
    starts = range(0, max(len(document.text) - overlap, 1), step)
    return [
        Chunk(
            id=f"{document.id}:{n}",
            document_id=document.id,
            collection=document.collection,
            language=document.language,
            text=document.text[start : start + size],
        )
        for n, start in enumerate(starts)
    ]
