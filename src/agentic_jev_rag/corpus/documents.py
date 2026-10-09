"""Corpus records: one Document type for every Collection, stored as JSONL plus Markdown Policies."""

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

COLLECTIONS = ("candidates", "jobs", "interviews", "policies")


@dataclass(frozen=True)
class Document:
    id: str
    collection: str
    title: str
    text: str
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.collection not in COLLECTIONS:
            raise ValueError(f"unknown collection {self.collection!r} for {self.id}; expected one of {COLLECTIONS}")

    def consent_valid_on(self, day: date) -> bool:
        """Whether a Candidate's Consent covers `day`. Documents outside Candidates need no Consent."""
        if self.collection != "candidates":
            return True
        until = self.metadata.get("consent_until")
        return bool(until) and date.fromisoformat(until) >= day


def load_corpus(directory: Path) -> list[Document]:
    """Load `<collection>.jsonl` files and `policies/*.md` from `directory`, in a stable order."""
    directory = Path(directory)
    documents: list[Document] = []
    for collection in COLLECTIONS:
        path = directory / f"{collection}.jsonl"
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    documents.append(Document(**json.loads(line)))
    for path in sorted((directory / "policies").glob("*.md")):
        documents.append(_policy(path))
    return documents


def write_jsonl(documents: list[Document], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for document in documents:
            f.write(json.dumps(asdict(document), ensure_ascii=False) + "\n")


def _policy(path: Path) -> Document:
    text = path.read_text(encoding="utf-8")
    first = text.splitlines()[0] if text else path.stem
    title = first.lstrip("#").strip() or path.stem
    return Document(f"policy:{path.stem}", "policies", title, text, {"topic": path.stem})
