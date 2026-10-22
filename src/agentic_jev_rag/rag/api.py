"""The RAG API: Search, Match and Ask as HTTP endpoints that agents call as tools."""

from dataclasses import asdict
from typing import Annotated

from fastapi import Body, FastAPI, Path, Request
from fastapi.responses import JSONResponse
from qdrant_client import QdrantClient

from agentic_jev_rag.config import load_config
from agentic_jev_rag.corpus.documents import load_corpus
from agentic_jev_rag.openrouter import OpenRouterError, OpenRouterGateway
from agentic_jev_rag.rag.models import (
    AskRequest,
    AskResponse,
    DocumentOut,
    ErrorOut,
    HealthOut,
    MatchRequest,
    MatchResponse,
    SearchRequest,
    SearchResponse,
)
from agentic_jev_rag.rag.service import RagService, UnknownDocument
from agentic_jev_rag.system1 import System1Error

VERSION = "1.0.0"

DESCRIPTION = """
Retrieval over **SwiftAI Recruit's** documents, built to be called as tools by the AI Recruiter's agents
(and by any other agent or client).

**Collections:**

| Collection | Contents |
|---|---|
| `candidates` | CVs, with occupation, location, years of experience, skills and Consent date |
| `jobs` | Job descriptions from Clients, with required skills and seniority |
| `interviews` | Interview notes with scores and Outcomes |
| `policies` | Fair hiring, data protection and consent, interview guide, Scorecards, fast-track, AI oversight |

**Retrieval:**
1. Dense (embedding model) and sparse (hashed terms with IDF) search run in Qdrant and are fused with RRF.
2. The fused results are reranked with the reranker Model role.

**System 1 decisions.** `/ask` makes four fast, typed decisions: Input guardrail, Route, Chunk relevance and
Groundedness check. They are made by the LLM decider or by Jev, chosen per request with `system_1`. Every
decision is returned in `system_1_trace` with its latency and cost.

**Guarantees:**
- **Consent:** Candidates without valid Consent are never returned by any endpoint that retrieves chunks.
- **Refusals before retrieval:** discriminatory or out-of-scope questions are refused before anything is
  retrieved.

All models are called through OpenRouter. Upstream model failures return **502**, and unknown document ids return
**404**.
"""

TAGS = [
    {"name": "Retrieval", "description": "Search chunks, rank Candidates for a Job, and get grounded answers."},
    {"name": "Documents", "description": "Fetch whole documents by id."},
    {"name": "Operations", "description": "Service health."},
]

NOT_FOUND = {"model": ErrorOut, "description": "A `job_id`, `candidate_id` or document id that is not in the corpus."}
UPSTREAM = {"model": ErrorOut, "description": "A model call through OpenRouter failed, or a System 1 decider returned an invalid decision."}

SEARCH_EXAMPLES = {
    "candidates_by_skill": {
        "summary": "Candidates with a skill",
        "value": {"query": "Linux administration with Nagios and Puppet", "collections": ["candidates"], "top_k": 5},
    },
    "filtered": {
        "summary": "Filtered by occupation and experience",
        "value": {
            "query": "B2B quota-carrying account executive",
            "collections": ["candidates"],
            "filters": {"occupation": "sales", "min_years_experience": 5},
        },
    },
    "one_candidate_history": {
        "summary": "One Candidate's CV and interview notes",
        "value": {"query": "interview feedback", "collections": ["candidates", "interviews"], "filters": {"candidate_id": "C-0040"}},
    },
}
MATCH_EXAMPLES = {
    "job_in_corpus": {"summary": "A Job in the corpus", "value": {"job_id": "J-037", "top_k": 5}},
    "free_text_job": {
        "summary": "A job description not in the corpus",
        "value": {"job_text": "Senior Linux administrator: RHEL, Puppet, Nagios, 24/7 on-call.", "top_k": 5},
    },
}
ASK_EXAMPLES = {
    "policy": {
        "summary": "A policy question, routed automatically",
        "value": {"question": "How long is a candidate's consent valid?"},
    },
    "with_jev": {
        "summary": "Jev makes the System 1 decisions",
        "value": {"question": "Which Candidates have Workday HRIS experience?", "collections": ["candidates"], "system_1": "jev"},
    },
    "refused": {
        "summary": "A discriminatory question (refused)",
        "value": {"question": "Find sales candidates under 30 with no children."},
    },
}


def create_rag_app(service: RagService) -> FastAPI:
    app = FastAPI(
        title="SwiftAI RAG API",
        version=VERSION,
        summary="Search, Match and Ask over SwiftAI Recruit's candidates, jobs, interview notes and policies.",
        description=DESCRIPTION,
        openapi_tags=TAGS,
    )

    @app.exception_handler(UnknownDocument)
    def unknown(request: Request, error: UnknownDocument):
        return JSONResponse(status_code=404, content={"detail": f"unknown document {error}"})

    @app.exception_handler(System1Error)
    @app.exception_handler(OpenRouterError)
    def upstream(request: Request, error: Exception):
        return JSONResponse(status_code=502, content={"detail": str(error)})

    @app.get(
        "/health",
        tags=["Operations"],
        operation_id="health",
        summary="Health check",
        description="Returns `ok` while the process is serving requests. It does not check Qdrant or OpenRouter.",
        response_description="The service is up.",
    )
    def health() -> HealthOut:
        return HealthOut(status="ok")

    @app.post(
        "/search",
        tags=["Retrieval"],
        operation_id="search",
        summary="Search chunks",
        description=(
            "Hybrid search over one or more Collections.\n\n"
            "1. Dense and sparse rankings are fused with RRF.\n"
            "2. Metadata `filters` are applied per Collection.\n"
            "3. When `rerank` is true, the results are reranked.\n\n"
            "No System 1 decisions are made. Use `/ask` for a grounded answer."
        ),
        response_description="Matching chunks, best first.",
        responses={404: NOT_FOUND, 502: UPSTREAM},
    )
    def search(request: Annotated[SearchRequest, Body(openapi_examples=SEARCH_EXAMPLES)]) -> SearchResponse:
        return service.search(request)

    @app.post(
        "/match",
        tags=["Retrieval"],
        operation_id="match_candidates",
        summary="Rank Candidates for a Job",
        description=(
            "Ranks Candidates for a Job in the corpus (`job_id`) or for a free-text job description (`job_text`).\n\n"
            "1. The Job's title, required skills and description are the query.\n"
            "2. Candidate chunks are retrieved with hybrid search and grouped by Candidate.\n"
            "3. Each Candidate's top three chunks are reranked together against the Job.\n\n"
            "Candidates without valid Consent are excluded. Each result carries its Evidence chunks."
        ),
        response_description="Candidates ranked best first, with Evidence.",
        responses={404: NOT_FOUND, 502: UPSTREAM},
    )
    def match(request: Annotated[MatchRequest, Body(openapi_examples=MATCH_EXAMPLES)]) -> MatchResponse:
        return service.match(request)

    @app.post(
        "/ask",
        tags=["Retrieval"],
        operation_id="ask",
        summary="Ask a question",
        description=(
            "Answers a question from SwiftAI's documents with `[n]` citations.\n\n"
            "1. **Input guardrail.** Discriminatory or out-of-scope questions are refused, and nothing else runs.\n"
            "2. **Route.** Picks the Collection, unless `collections` is given.\n"
            "3. **Retrieval.** Hybrid search, then rerank.\n"
            "4. **Chunk relevance.** Irrelevant chunks are dropped before the model sees them.\n"
            "5. **Answer.** The System 2 model answers from the kept chunks only.\n"
            "6. **Groundedness check.** Checks the answer against the cited chunks.\n\n"
            "`system_1` chooses which System 1 decider makes these decisions: `llm` or `jev`."
        ),
        response_description="A grounded answer with citations, or a refusal, plus the System 1 trace.",
        responses={404: NOT_FOUND, 502: UPSTREAM},
    )
    def ask(request: Annotated[AskRequest, Body(openapi_examples=ASK_EXAMPLES)]) -> AskResponse:
        return service.ask(request)

    @app.get(
        "/documents/{document_id}",
        tags=["Documents"],
        operation_id="get_document",
        summary="Get a document",
        description=(
            "Returns a whole document by id: a Candidate's CV (`C-…`), a Job (`J-…`), an interview note (`I-…`) "
            "or a Policy (`policy:…`). Unlike the retrieval endpoints, this does not check Consent."
        ),
        response_description="The document.",
        responses={404: NOT_FOUND},
    )
    def document(
        document_id: Annotated[
            str,
            Path(
                description="Document id.",
                openapi_examples={
                    "job": {"summary": "A Job", "value": "J-037"},
                    "candidate": {"summary": "A Candidate's CV", "value": "C-0040"},
                    "policy": {"summary": "A Policy", "value": "policy:fair-hiring"},
                },
            ),
        ],
    ) -> DocumentOut:
        return DocumentOut(**asdict(service.document(document_id)))

    return app


def create_default_rag_app() -> FastAPI:
    """The app as deployed: config from file and environment, real OpenRouter, the committed corpus."""
    config = load_config()
    qdrant = QdrantClient(url=config.qdrant_url, api_key=config.qdrant_api_key or None)
    service = RagService(config, OpenRouterGateway.from_env(), qdrant, load_corpus(config.corpus_dir))
    return create_rag_app(service)
