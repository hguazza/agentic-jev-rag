"""Hybrid index and Search against real Qdrant with the fake gateway."""

from datetime import date

from agentic_jev_rag.gateway import FakeModelGateway
from agentic_jev_rag.rag.index import SearchFilters, hybrid_search, index_corpus


def search(query, config, qdrant, collections=("candidates",), filters=SearchFilters(), gateway=None, limit=5):
    return hybrid_search(
        query, collections, filters, limit, config=config, gateway=gateway or FakeModelGateway(), qdrant=qdrant
    )


def documents_of(hits):
    return list(dict.fromkeys(h.chunk.document_id for h in hits))


def test_search_finds_the_candidates_matching_the_query(indexed, config, qdrant):
    hits = search("Kafka event pipelines in Python on AWS", config, qdrant)

    assert set(documents_of(hits)[:2]) <= {"C-001", "C-004", "C-002"}
    assert hits[0].chunk.collection == "candidates"
    assert hits[0].chunk.metadata["occupation"] == "software_engineering"


def test_rare_exact_terms_are_found_through_the_sparse_side(indexed, config, qdrant):
    # A differently seeded fake makes dense similarity meaningless, so only sparse can rank these.
    unrelated_dense = FakeModelGateway(seed="unrelated")

    hits = search("Debezium", config, qdrant, gateway=unrelated_dense, limit=8)

    mentioning = [h.chunk.id for h in hits if "debezium" in h.chunk.text.lower()]
    assert len(mentioning) == 2  # C-002's experience and skills sections
    assert all(chunk_id.startswith("C-002#") for chunk_id in mentioning)


def test_occupation_filter_excludes_other_occupations(indexed, config, qdrant):
    hits = search("Python", config, qdrant, filters=SearchFilters(occupation="data_science"), limit=8)

    assert hits
    assert {h.chunk.metadata["occupation"] for h in hits} == {"data_science"}


def test_min_years_and_location_filters(indexed, config, qdrant):
    hits = search(
        "engineer", config, qdrant, filters=SearchFilters(min_years_experience=7, location="Berlin"), limit=8
    )

    assert documents_of(hits) == ["C-004"]


def test_expired_consent_is_filtered_out(indexed, config, qdrant):
    hits = search(
        "principal Kafka streaming platform", config, qdrant,
        filters=SearchFilters(consent_valid_on=date(2026, 10, 8)), limit=8,
    )

    assert "C-004" not in documents_of(hits)


def test_candidate_filter_applies_to_interview_notes_too(indexed, config, qdrant):
    hits = search(
        "interview", config, qdrant, collections=("candidates", "interviews"),
        filters=SearchFilters(candidate_id="C-001"), limit=8,
    )

    assert set(documents_of(hits)) == {"C-001", "I-001"}


def test_filters_a_collection_lacks_do_not_hide_it(indexed, config, qdrant):
    hits = search(
        "consent retention", config, qdrant, collections=("policies",),
        filters=SearchFilters(occupation="software_engineering"),
    )

    assert hits and hits[0].chunk.collection == "policies"


def test_reindexing_is_idempotent(indexed, config, qdrant, fixture_documents):
    again = index_corpus(fixture_documents, config, FakeModelGateway(), qdrant)

    assert again == indexed
    assert set(again) == {"candidates", "jobs", "interviews", "policies"}
