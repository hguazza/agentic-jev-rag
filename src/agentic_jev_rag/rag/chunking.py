"""Structure-aware chunking: split documents on CV, Job and Markdown section headings."""

import re
from dataclasses import dataclass

from agentic_jev_rag.corpus.documents import Document

SECTION_NAMES = (
    "summary", "profile", "objective", "experience", "work history", "work experience", "employment",
    "professional experience", "education", "skills", "certifications", "projects", "accomplishments",
    "responsibilities", "requirements", "qualifications", "benefits", "about",
)
_SECTION_LINE = re.compile(rf"^\s*({'|'.join(SECTION_NAMES)})\s*:?\s*$", re.IGNORECASE)
_MARKDOWN_HEADING = re.compile(r"^\s*#{1,6}\s+(.+?)\s*$")


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    collection: str
    section: str
    text: str
    metadata: dict


def chunk_document(document: Document, size: int, overlap: int) -> list[Chunk]:
    """Split on section headings; window sections longer than `size` characters with `overlap`."""
    if not 0 <= overlap < size:
        raise ValueError(f"overlap must be in [0, size); got size={size}, overlap={overlap}")
    pieces = [
        (section, window)
        for section, body in _sections(document.text)
        for window in _windows(body, size, overlap)
    ]
    return [
        Chunk(
            id=f"{document.id}#{n}",
            document_id=document.id,
            collection=document.collection,
            section=section,
            text=f"{document.title} — {section}\n{body}",
            metadata=dict(document.metadata),
        )
        for n, (section, body) in enumerate(pieces)
    ]


def _sections(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, list[str]]] = [("body", [])]
    for line in text.splitlines():
        heading = _heading(line)
        if heading:
            sections.append((heading, []))
        else:
            sections[-1][1].append(line)
    return [(name, "\n".join(lines).strip()) for name, lines in sections if "\n".join(lines).strip()]


def _heading(line: str) -> str | None:
    if match := _MARKDOWN_HEADING.match(line):
        return match.group(1).lower()
    if match := _SECTION_LINE.match(line):
        return match.group(1).lower()
    return None


def _windows(body: str, size: int, overlap: int) -> list[str]:
    if len(body) <= size:
        return [body]
    step = size - overlap
    return [body[start : start + size] for start in range(0, len(body) - overlap, step)]
