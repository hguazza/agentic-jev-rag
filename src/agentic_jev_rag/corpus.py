"""Normalised document records, shared by every Collection and Language."""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Document:
    id: str
    collection: str
    language: str
    court: str
    date: str
    title: str
    source: str
    text: str


def load_documents(directory: Path) -> list[Document]:
    """Load every `*.json` document record under `directory`, in a stable order."""
    return [
        Document(**json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(Path(directory).rglob("*.json"))
    ]
