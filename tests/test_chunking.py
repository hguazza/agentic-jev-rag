"""Structure-aware chunking: sections of CVs, Jobs and Policies become chunks."""

from agentic_recruiter.corpus.documents import Document
from agentic_recruiter.rag.chunking import chunk_document

CV = Document(
    "C-1",
    "candidates",
    "Ana — CV",
    "Ana Ribeiro\nExperience:\nBuilt Kafka pipelines.\n\nEducation\nBSc Computer Science.\n\nSKILLS\nPython, Kafka\n",
    {"occupation": "software_engineering"},
)


def test_cv_sections_become_named_chunks():
    chunks = chunk_document(CV, size=1000, overlap=100)

    assert [c.section for c in chunks] == ["body", "experience", "education", "skills"]
    assert "Built Kafka pipelines." in chunks[1].text


def test_each_chunk_carries_its_title_section_and_the_documents_metadata():
    chunk = chunk_document(CV, size=1000, overlap=100)[1]

    assert chunk.text.startswith("Ana — CV — experience\n")
    assert chunk.metadata == {"occupation": "software_engineering"}
    assert chunk.document_id == "C-1"
    assert chunk.collection == "candidates"
    assert chunk.id == "C-1#1"


def test_markdown_headings_split_policies():
    policy = Document("policy:p", "policies", "P", "# P\nIntro.\n\n## Retention\nKeep 12 months.\n", {})

    chunks = chunk_document(policy, size=1000, overlap=100)

    assert [c.section for c in chunks] == ["p", "retention"]


def test_long_sections_are_windowed_with_overlap():
    text = "Experience\n" + " ".join(f"w{i}" for i in range(400))
    document = Document("C-2", "candidates", "T", text, {})

    chunks = chunk_document(document, size=600, overlap=100)

    assert len(chunks) > 2
    assert all(c.section == "experience" for c in chunks)
    body = lambda c: c.text.split("\n", 1)[1]
    assert body(chunks[0])[-50:] in body(chunks[1])


def test_text_without_headings_is_one_body_chunk():
    note = Document("I-1", "interviews", "Note", "Strong Kafka answers. Advanced.", {"outcome": "advanced"})

    [chunk] = chunk_document(note, size=1000, overlap=100)

    assert chunk.section == "body"
