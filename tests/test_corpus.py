"""Corpus records: loading every Collection from the committed corpus layout."""

import json
from datetime import date
from pathlib import Path

import pytest

from agentic_jev_rag.corpus.documents import Document, load_corpus, write_jsonl
from agentic_jev_rag.corpus.dossier import add_dossiers
from agentic_jev_rag.gateway import FakeModelGateway

FIXTURE_CORPUS = Path(__file__).parent / "fixtures" / "corpus"


def test_load_corpus_returns_every_collection():
    documents = load_corpus(FIXTURE_CORPUS)

    collections = {d.collection for d in documents}
    assert collections == {"candidates", "jobs", "interviews", "policies"}
    assert len([d for d in documents if d.collection == "candidates"]) == 8


def test_policies_come_from_markdown_with_their_heading_as_title():
    policies = {d.id: d for d in load_corpus(FIXTURE_CORPUS) if d.collection == "policies"}

    assert policies["policy:fair-hiring"].title == "Fair Hiring Policy"
    assert "Protected characteristics" in policies["policy:fair-hiring"].text
    assert policies["policy:fair-hiring"].metadata == {"topic": "fair-hiring"}


def test_consent_is_valid_only_until_the_consent_date():
    expired = next(d for d in load_corpus(FIXTURE_CORPUS) if d.id == "C-004")

    assert expired.consent_valid_on(date(2025, 12, 31))
    assert not expired.consent_valid_on(date(2026, 1, 1))


def test_consent_does_not_apply_outside_the_candidates_collection():
    job = next(d for d in load_corpus(FIXTURE_CORPUS) if d.id == "J-001")

    assert job.consent_valid_on(date(2099, 1, 1))


def test_a_document_in_an_unknown_collection_is_rejected(tmp_path):
    (tmp_path / "candidates.jsonl").write_text(
        json.dumps({"id": "X", "collection": "resumes", "title": "t", "text": "x", "metadata": {}}) + "\n"
    )

    with pytest.raises(ValueError, match="resumes"):
        load_corpus(tmp_path)


def test_written_documents_load_back_unchanged(tmp_path):
    document = Document("J-9", "jobs", "Role", "About\nText", {"client": "Acme"})

    write_jsonl([document], tmp_path / "jobs.jsonl")

    assert load_corpus(tmp_path) == [document]


DOSSIER = {
    "cover_letter": "I am applying for the Linux role.",
    "linkedin": {"headline": "Sysadmin", "about": "Ten years of Linux.", "positions": [{"title": "Sysadmin", "company": "Acme", "years": "2016-2026"}]},
}


def test_dossiers_are_added_to_the_named_candidates_only(tmp_path):
    path = tmp_path / "candidates.jsonl"
    write_jsonl([
        Document("C-1", "candidates", "A — CV", "Linux admin CV.", {"name": "A"}),
        Document("C-2", "candidates", "B — CV", "Designer CV.", {"name": "B"}),
    ], path)
    prompts = []

    def complete(model, prompt, system, schema):
        prompts.append(prompt)
        return json.dumps(DOSSIER)

    add_dossiers(path, ["C-1"], FakeModelGateway(complete_responder=complete), "fake/s2", discrepancy_ids=["C-1"])

    by_id = {d.id: d for d in load_corpus(tmp_path)}
    assert by_id["C-1"].metadata["dossier"] == DOSSIER and by_id["C-1"].metadata["name"] == "A"
    assert "dossier" not in by_id["C-2"].metadata
    assert "Linux admin CV." in prompts[0] and "deliberate discrepancies" in prompts[0]
