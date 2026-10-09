"""Live smoke suite: real OpenRouter calls against the indexed corpus. Opt in with `uv run pytest -m live`.

Needs OPEN_ROUTER_API_TOKEN, Qdrant running, and the corpus indexed (`uv run python -m agentic_jev_rag.rag.ingest`).
A full run costs roughly $1.
"""

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from agentic_jev_rag.config import load_config
from agentic_jev_rag.corpus.documents import load_corpus
from agentic_jev_rag.openrouter import OpenRouterGateway
from agentic_jev_rag.rag.api import create_rag_app
from agentic_jev_rag.rag.models import AskRequest
from agentic_jev_rag.rag.service import RagService
from agentic_jev_rag.recruiter.context import RecruiterContext
from agentic_jev_rag.recruiter.graph import run_recruitment
from agentic_jev_rag.recruiter.rag_client import RagClient

pytestmark = pytest.mark.live

CONFIG = load_config()
CHAT_ROLES = ["system_2_model", "system_1_baseline", "reference_judge"]


@pytest.fixture(scope="module")
def gateway() -> OpenRouterGateway:
    return OpenRouterGateway.from_env()


@pytest.fixture(scope="module")
def service(gateway) -> RagService:
    return RagService(CONFIG, gateway, QdrantClient(url=CONFIG.qdrant_url), load_corpus(CONFIG.corpus_dir))


@pytest.mark.parametrize("role", CHAT_ROLES)
def test_chat_model_role_answers_and_reports_cost(gateway, role):
    completion = gateway.complete(getattr(CONFIG.models, role), "Reply with the single word: ready", system="Be terse.")

    assert "ready" in completion.text.lower()
    assert completion.cost_usd > 0


def test_embedding_model_role_embeds_and_reports_cost(gateway):
    embeddings = gateway.embed(CONFIG.models.embedding_model, ["Linux administrator", "Recruiter"])

    assert len(embeddings.vectors) == 2 and embeddings.cost_usd > 0


def test_reranker_role_ranks_the_relevant_document_first(gateway):
    result = gateway.rerank(CONFIG.models.reranker, "Linux server administration", ["Pastry chef", "Red Hat Linux sysadmin"])

    assert result.scores[1] > result.scores[0] and result.cost_usd > 0


def test_jev_refuses_an_indirectly_discriminatory_request(gateway):
    from agentic_jev_rag.system1 import JevDecider

    decision = JevDecider(gateway, CONFIG.models.jev_model).input_guardrail("Digital natives only, please.")

    assert decision.verdict == "discriminatory"
    assert decision.trace.cost_usd > 0


@pytest.mark.parametrize("system_1", ["llm", "jev"])
def test_ask_answers_a_policy_question_with_grounded_citations(service, system_1):
    response = service.ask(AskRequest(question="How long is a candidate's consent valid?", system_1=system_1))

    assert not response.refused
    assert "24 months" in response.answer
    assert response.citations and response.grounded is True
    assert {t.implementation for t in response.system_1_trace} == {system_1}


def test_a_real_recruitment_run_produces_a_compliant_report(gateway, service):
    rag = RagClient("http://rag", http=TestClient(create_rag_app(service)))
    ctx = RecruiterContext(config=CONFIG, gateway=gateway, rag=rag, system_1="jev")

    report = run_recruitment(ctx, "Fill the Senior Linux Administrator role", "J-037", max_interviews=1)

    assert not report.refused
    assert report.shortlist and report.shortlist.candidates
    assert len(report.verdicts) == len(report.shortlist.candidates)
    assert all(f.passed for f in report.findings if f.check == "consent")
    assert {t.implementation for t in report.system_1_trace} == {"jev"}
    assert report.cost_usd > 0
