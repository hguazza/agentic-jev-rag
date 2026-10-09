"""Request and response models of the RAG API."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

Collection = Literal["candidates", "jobs", "interviews", "policies"]
System1Name = Literal["llm", "jev"]


class Filters(BaseModel):
    occupation: str | None = None
    location: str | None = None
    min_years_experience: float | None = None
    candidate_id: str | None = None
    job_id: str | None = None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    collections: list[Collection] | None = None
    filters: Filters = Filters()
    top_k: int = Field(default=8, ge=1, le=50)
    rerank: bool = True


class ChunkOut(BaseModel):
    chunk_id: str
    document_id: str
    collection: str
    section: str
    text: str
    score: float
    metadata: dict


class SearchResponse(BaseModel):
    chunks: list[ChunkOut]


class MatchRequest(BaseModel):
    job_id: str | None = None
    job_text: str | None = None
    top_k: int = Field(default=5, ge=1, le=50)
    filters: Filters = Filters()

    @model_validator(mode="after")
    def one_job(self):
        if (self.job_id is None) == (self.job_text is None):
            raise ValueError("give exactly one of job_id and job_text")
        return self


class CandidateMatch(BaseModel):
    candidate_id: str
    name: str
    score: float
    evidence: list[ChunkOut]


class MatchResponse(BaseModel):
    job_id: str | None
    candidates: list[CandidateMatch]


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    collections: list[Collection] | None = None
    filters: Filters = Filters()
    system_1: System1Name | None = None


class TraceOut(BaseModel):
    decision: str
    outcome: str
    latency_s: float
    cost_usd: float
    implementation: str


class Citation(BaseModel):
    n: int
    chunk_id: str
    document_id: str


class AskResponse(BaseModel):
    answer: str | None
    refused: bool
    refusal_reason: str | None
    citations: list[Citation]
    grounded: bool | None
    evidence: list[ChunkOut]
    system_1_trace: list[TraceOut]
    cost_usd: float
    latency_s: float


class DocumentOut(BaseModel):
    id: str
    collection: str
    title: str
    text: str
    metadata: dict
