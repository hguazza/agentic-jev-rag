"""RAG API seam: FastAPI app over real Qdrant with the fake Model gateway."""

from datetime import date

import pytest
from fastapi.testclient import TestClient

from agentic_recruiter.rag.api import create_rag_app
from agentic_recruiter.rag.service import RagService
from tests.rag_scripts import scripted_gateway

TODAY = date(2026, 10, 8)


@pytest.fixture
def client_for(indexed, config, qdrant, fixture_documents):
    def make(gateway=None):
        service = RagService(
            config, gateway or scripted_gateway(), qdrant, fixture_documents, today=lambda: TODAY
        )
        return TestClient(create_rag_app(service))

    return make


def test_search_returns_chunks_with_their_metadata(client_for):
    response = client_for().post(
        "/search", json={"query": "Kafka Python", "collections": ["candidates"], "top_k": 3}
    )

    assert response.status_code == 200
    chunks = response.json()["chunks"]
    assert len(chunks) == 3
    assert {"chunk_id", "document_id", "collection", "section", "text", "score", "metadata"} <= set(chunks[0])


def test_search_filters_by_occupation(client_for):
    chunks = client_for().post(
        "/search",
        json={"query": "Python", "collections": ["candidates"], "filters": {"occupation": "data_science"}},
    ).json()["chunks"]

    assert chunks and {c["metadata"]["occupation"] for c in chunks} == {"data_science"}


def test_search_never_returns_candidates_without_valid_consent(client_for):
    chunks = client_for().post(
        "/search", json={"query": "Mei Tanaka principal Streamly", "collections": ["candidates"], "top_k": 20}
    ).json()["chunks"]

    assert "C-004" not in {c["document_id"] for c in chunks}


def test_match_ranks_backend_candidates_first_with_evidence(client_for):
    response = client_for().post("/match", json={"job_id": "J-001", "top_k": 3})

    assert response.status_code == 200
    candidates = response.json()["candidates"]
    assert candidates[0]["candidate_id"] in {"C-001", "C-002"}
    assert candidates[0]["name"] in {"Ana Ribeiro", "Tomás Verhoeven"}
    assert all(c["evidence"] for c in candidates)
    assert all(e["document_id"] == c["candidate_id"] for c in candidates for e in c["evidence"])


def test_match_excludes_the_best_match_when_their_consent_has_expired(client_for):
    candidates = client_for().post("/match", json={"job_id": "J-001", "top_k": 8}).json()["candidates"]

    assert "C-004" not in [c["candidate_id"] for c in candidates]


def test_match_accepts_free_job_text(client_for):
    candidates = client_for().post(
        "/match", json={"job_text": "Data scientist for A/B testing and causal inference", "top_k": 2}
    ).json()["candidates"]

    assert candidates[0]["candidate_id"] == "C-005"


def test_match_for_an_unknown_job_is_404(client_for):
    assert client_for().post("/match", json={"job_id": "J-404"}).status_code == 404


def test_match_needs_exactly_one_of_job_id_and_job_text(client_for):
    assert client_for().post("/match", json={}).status_code == 422


def test_ask_refuses_a_discriminatory_question_before_any_retrieval(client_for):
    gateway = scripted_gateway(verdict="discriminatory")

    body = client_for(gateway).post("/ask", json={"question": "Which candidates are under 30?"}).json()

    assert body["refused"] is True
    assert "discriminatory" in body["refusal_reason"]
    assert body["answer"] is None and body["evidence"] == []
    assert [t["decision"] for t in body["system_1_trace"]] == ["input_guardrail"]
    assert not [kind for kind, _, _ in gateway.calls if kind in ("embed", "rerank")]


def test_ask_routes_answers_with_citations_and_checks_groundedness(client_for):
    gateway = scripted_gateway(route="policies", answer="Consent lasts 24 months [1].", grounded=True)

    body = client_for(gateway).post("/ask", json={"question": "How long is candidate consent valid?"}).json()

    assert body["refused"] is False
    assert body["answer"] == "Consent lasts 24 months [1]."
    assert {e["collection"] for e in body["evidence"]} == {"policies"}
    [citation] = body["citations"]
    assert citation["chunk_id"] == body["evidence"][0]["chunk_id"]
    assert body["grounded"] is True
    assert [t["decision"] for t in body["system_1_trace"]] == [
        "input_guardrail", "route", "chunk_relevance", "groundedness",
    ]


def test_ask_drops_chunks_judged_irrelevant(client_for):
    gateway = scripted_gateway(route="policies", relevant=lambda i: i == 0)

    body = client_for(gateway).post("/ask", json={"question": "How long is consent valid?"}).json()

    assert len(body["evidence"]) == 1


def test_ask_with_explicit_collections_skips_routing(client_for):
    body = client_for().post(
        "/ask", json={"question": "Who knows Kafka?", "collections": ["candidates"]}
    ).json()

    assert "route" not in [t["decision"] for t in body["system_1_trace"]]


def test_ask_ignores_citations_outside_the_sources_and_is_then_ungrounded(client_for):
    gateway = scripted_gateway(answer="Something unsupported [9].")

    response = client_for(gateway).post("/ask", json={"question": "Who knows Kafka?"})

    assert response.status_code == 200
    body = response.json()
    assert body["citations"] == []
    assert body["grounded"] is False


def test_ask_with_the_jev_decider_traces_jev(client_for):
    body = client_for().post("/ask", json={"question": "Who knows Kafka?", "system_1": "jev"}).json()

    assert {t["implementation"] for t in body["system_1_trace"]} == {"jev"}


def test_documents_are_served_by_id(client_for):
    client = client_for()

    assert client.get("/documents/J-001").json()["metadata"]["client"] == "Northwind Logistics"
    assert client.get("/documents/NOPE").status_code == 404
    assert client.get("/health").json() == {"status": "ok"}


def test_filters_naming_an_unknown_candidate_or_job_are_404(client_for):
    client = client_for()

    assert client.post("/ask", json={"question": "Kafka?", "filters": {"candidate_id": "C-999"}}).status_code == 404
    assert client.post("/search", json={"query": "Kafka", "filters": {"job_id": "J-999"}}).status_code == 404
    assert client.post("/search", json={"query": "Kafka", "filters": {"candidate_id": "J-001"}}).status_code == 404
