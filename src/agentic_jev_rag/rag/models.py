"""Request and response models of the RAG API.

Docstrings and `Field` descriptions are the API documentation: FastAPI publishes them in the
OpenAPI schema served at `/openapi.json` and rendered at `/docs`.
"""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

Collection = Literal["candidates", "jobs", "interviews", "policies"]
System1Name = Literal["llm", "jev"]

CHUNK_EXAMPLE = {
    "chunk_id": "C-0040#2",
    "document_id": "C-0040",
    "collection": "candidates",
    "section": "experience",
    "text": "Lucas Singh — CV — experience\nAdministered 12+ Red Hat/CentOS servers: patching, backups, Nagios monitoring...",
    "score": 0.84,
    "metadata": {"occupation": "information_technology", "location": "Dublin, IE", "years_experience": 3},
}


class Filters(BaseModel):
    """Metadata filters. Each applies only to the Collections that have the field and is ignored by the others
    (a Policy has no occupation, for example). Candidates without valid Consent are always excluded."""

    occupation: str | None = Field(
        default=None,
        description="Candidates and Jobs: occupation code.",
        examples=["information_technology", "human_resources", "sales", "design", "finance"],
    )
    location: str | None = Field(
        default=None,
        description="Candidates and Jobs: matches any word of the location, case-insensitive.",
        examples=["Dublin", "New York"],
    )
    min_years_experience: float | None = Field(
        default=None, ge=0, description="Candidates: at least this many years of professional experience.", examples=[5]
    )
    candidate_id: str | None = Field(
        default=None,
        description="Restrict to one Candidate: their CV chunks and their interview notes. Unknown ids return 404.",
        examples=["C-0040"],
    )
    job_id: str | None = Field(
        default=None,
        description="Restrict to one Job: its description and the interview notes for it. Unknown ids return 404.",
        examples=["J-037"],
    )


class SearchRequest(BaseModel):
    """A hybrid search: dense and sparse retrieval fused with RRF, then optionally reranked."""

    query: str = Field(min_length=1, description="Free-text query.", examples=["Linux administration with Nagios and Puppet"])
    collections: list[Collection] | None = Field(
        default=None, description="Collections to search. Omit to search all four.", examples=[["candidates"]]
    )
    filters: Filters = Field(default=Filters(), description="Metadata filters.")
    top_k: int = Field(default=8, ge=1, le=50, description="Number of chunks to return.")
    rerank: bool = Field(
        default=True, description="Rerank the fused results with the reranker Model role before cutting to `top_k`."
    )


class ChunkOut(BaseModel):
    """One retrieved chunk: a section of a document with its score and the document's metadata."""

    model_config = {"json_schema_extra": {"examples": [CHUNK_EXAMPLE]}}

    chunk_id: str = Field(description="`<document_id>#<n>`, stable across re-indexing.", examples=["C-0040#2"])
    document_id: str = Field(description="The document the chunk comes from.", examples=["C-0040"])
    collection: Collection = Field(description="The Collection the document belongs to.")
    section: str = Field(description="The section heading the chunk came from, or `body`.", examples=["experience"])
    text: str = Field(description="The chunk text, prefixed with the document title and section.")
    score: float = Field(description="Relevance score: the reranker's score when reranked, otherwise the RRF score. Higher is better.")
    metadata: dict = Field(description="The document's metadata, e.g. occupation, location, skills, consent date.")


class SearchResponse(BaseModel):
    """Chunks matching a search, best first."""

    chunks: list[ChunkOut] = Field(description="Matching chunks, best first, at most `top_k`.")


class MatchRequest(BaseModel):
    """Rank Candidates for a Job. Give exactly one of `job_id` and `job_text`."""

    job_id: str | None = Field(default=None, description="A Job in the corpus. Unknown ids return 404.", examples=["J-037"])
    job_text: str | None = Field(
        default=None,
        description="A job description that is not in the corpus, used as the query.",
        examples=["Senior Linux administrator: RHEL, Puppet, Nagios, on-call for 24/7 web platforms."],
    )
    top_k: int = Field(default=5, ge=1, le=50, description="Number of Candidates to return.")
    filters: Filters = Field(default=Filters(), description="Metadata filters on Candidates.")

    @model_validator(mode="after")
    def one_job(self):
        if (self.job_id is None) == (self.job_text is None):
            raise ValueError("give exactly one of job_id and job_text")
        return self


class CandidateMatch(BaseModel):
    """A Candidate ranked for a Job, with the CV chunks that are the Evidence for the ranking."""

    candidate_id: str = Field(description="The Candidate's id.", examples=["C-0040"])
    name: str = Field(description="The Candidate's name.", examples=["Lucas Singh"])
    score: float = Field(description="Reranker score of the Candidate's Evidence against the Job. Higher is better.")
    evidence: list[ChunkOut] = Field(description="Up to three CV chunks that matched the Job, best first.")


class MatchResponse(BaseModel):
    """Candidates ranked for a Job, best first. Only Candidates with valid Consent are considered."""

    job_id: str | None = Field(description="The Job matched, or null when `job_text` was given.", examples=["J-037"])
    candidates: list[CandidateMatch] = Field(description="Ranked Candidates, best first, at most `top_k`.")


class AskRequest(BaseModel):
    """A question for a grounded answer."""

    question: str = Field(min_length=1, description="The question, in plain English.", examples=["How long is a candidate's consent valid?"])
    collections: list[Collection] | None = Field(
        default=None, description="Collections to search. Omit to let the Route decision choose one."
    )
    filters: Filters = Field(default=Filters(), description="Metadata filters.")
    system_1: System1Name | None = Field(
        default=None,
        description="Which System 1 decider makes the Input guardrail, Route, Chunk relevance and Groundedness "
        "check decisions: `llm` (the System 1 baseline) or `jev`. Omit to use the service default.",
    )


class TraceOut(BaseModel):
    """One System 1 decision made while answering."""

    decision: Literal["input_guardrail", "route", "chunk_relevance", "groundedness", "turn_check", "interview_signals"] = Field(
        description="Which System 1 decision this was. `turn_check` and `interview_signals` are made only in live interviews."
    )
    outcome: str = Field(description="What was decided, e.g. `allowed`, `policies`, `kept 3/8`, `grounded`.", examples=["allowed"])
    latency_s: float = Field(description="End-to-end latency of the decision through OpenRouter, in seconds.", examples=[0.44])
    cost_usd: float = Field(description="Cost reported by OpenRouter for the decision, in US dollars.", examples=[0.0000231])
    implementation: System1Name = Field(description="The System 1 decider that made it.")


class Citation(BaseModel):
    """A source cited in the answer as `[n]`."""

    n: int = Field(description="The number used in the answer text, 1-based.", examples=[1])
    chunk_id: str = Field(description="The cited chunk, also present in `evidence`.", examples=["policy:data-protection-and-consent#2"])
    document_id: str = Field(description="The document the cited chunk comes from.", examples=["policy:data-protection-and-consent"])


class AskResponse(BaseModel):
    """A grounded answer with citations, or a refusal, plus the System 1 trace."""

    answer: str | None = Field(description="The answer with `[n]` citations; null when the question was refused.")
    refused: bool = Field(description="True when the Input guardrail refused the question. Nothing was retrieved.")
    refusal_reason: str | None = Field(description="Why the question was refused, with the Policy it breaks.")
    citations: list[Citation] = Field(description="Sources cited in the answer, in first-cited order. Citation numbers outside the sources are ignored.")
    grounded: bool | None = Field(
        description="Groundedness check result: true when the answer's facts are supported by the cited chunks, "
        "false when they are not or nothing is cited, null when no answer was written."
    )
    evidence: list[ChunkOut] = Field(description="The chunks shown to the System 2 model, numbered from 1 in this order.")
    system_1_trace: list[TraceOut] = Field(description="Every System 1 decision made for this question, in order.")
    cost_usd: float = Field(description="Total model cost of the request reported by OpenRouter, in US dollars.")
    latency_s: float = Field(description="Total time to answer, in seconds.")


class DocumentOut(BaseModel):
    """A whole document: a Candidate's CV, a Job, an interview note or a Policy."""

    id: str = Field(description="Document id: `C-…` Candidates, `J-…` Jobs, `I-…` interview notes, `policy:…` Policies.", examples=["J-037"])
    collection: Collection = Field(description="The Collection the document belongs to.")
    title: str = Field(description="Document title.", examples=["Senior Linux Administrator — Lumen Studio Collective"])
    text: str = Field(description="Full text.")
    metadata: dict = Field(description="Collection-specific metadata, e.g. a Job's client and required skills, or a Candidate's consent date.")


class ErrorOut(BaseModel):
    """An error with a human-readable explanation."""

    detail: str = Field(description="What went wrong.", examples=["unknown document J-404"])


class HealthOut(BaseModel):
    """Liveness of the service."""

    status: Literal["ok"] = Field(description="Always `ok` when the process is serving requests.")
