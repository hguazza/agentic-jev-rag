"""The RAG service: Search, Match and Ask over the hybrid index, with System 1 decisions."""

import re
import time
from collections import OrderedDict
from datetime import date
from typing import Callable

from qdrant_client import QdrantClient

from agentic_jev_rag.config import Config
from agentic_jev_rag.corpus.documents import Document
from agentic_jev_rag.gateway import CostMeter, ModelGateway
from agentic_jev_rag.rag.index import ScoredChunk, SearchFilters, hybrid_search
from agentic_jev_rag.rag.models import (
    AskRequest,
    AskResponse,
    CandidateMatch,
    ChunkOut,
    Citation,
    Filters,
    MatchRequest,
    MatchResponse,
    SearchRequest,
    SearchResponse,
    TraceOut,
)
from agentic_jev_rag.system1 import System1TraceEntry, make_deciders

ANSWER_SYSTEM = (
    "You are SwiftAI Recruit's recruiting assistant. Answer using only the numbered sources. Cite every claim "
    "with its source number in square brackets, e.g. [1]. If the sources do not answer the question, say so. "
    "Never mention or infer protected characteristics such as age, gender, family status, ethnicity, religion "
    "or health."
)
NO_EVIDENCE = "I could not find evidence for this in SwiftAI's documents."
EVIDENCE_PER_CANDIDATE = 3


class UnknownDocument(Exception):
    pass


class RagService:
    def __init__(
        self,
        config: Config,
        gateway: ModelGateway,
        qdrant: QdrantClient,
        documents: list[Document],
        *,
        today: Callable[[], date] = date.today,
    ):
        self._config = config
        self._gateway = gateway
        self._qdrant = qdrant
        self._documents = {d.id: d for d in documents}
        self._today = today

    def document(self, document_id: str) -> Document:
        if document_id not in self._documents:
            raise UnknownDocument(document_id)
        return self._documents[document_id]

    def search(self, request: SearchRequest) -> SearchResponse:
        meter = CostMeter(self._gateway)
        collections = request.collections or list(self._config.collections)
        hits = self._retrieve(meter, request.query, collections, request.filters, request.top_k, request.rerank)
        return SearchResponse(chunks=[_chunk_out(h) for h in hits])

    def match(self, request: MatchRequest) -> MatchResponse:
        meter = CostMeter(self._gateway)
        if request.job_id is not None:
            job = self.document(request.job_id)
            if job.collection != "jobs":
                raise UnknownDocument(request.job_id)
            skills = ", ".join(job.metadata.get("required_skills", []))
            query = f"{job.title}\nRequired skills: {skills}\n{job.text}"
        else:
            query = request.job_text

        hits = hybrid_search(
            query,
            ["candidates"],
            self._filters(request.filters),
            self._config.rerank_candidates,
            config=self._config,
            gateway=meter,
            qdrant=self._qdrant,
        )
        evidence: OrderedDict[str, list[ScoredChunk]] = OrderedDict()
        for hit in hits:
            chunks = evidence.setdefault(hit.chunk.document_id, [])
            if len(chunks) < EVIDENCE_PER_CANDIDATE:
                chunks.append(hit)
        if not evidence:
            return MatchResponse(job_id=request.job_id, candidates=[])

        candidate_ids = list(evidence)
        summaries = ["\n".join(h.chunk.text for h in evidence[cid]) for cid in candidate_ids]
        scores = meter.rerank(self._config.models.reranker, query, summaries).scores
        ranked = sorted(zip(candidate_ids, scores), key=lambda pair: pair[1], reverse=True)[: request.top_k]
        return MatchResponse(
            job_id=request.job_id,
            candidates=[
                CandidateMatch(
                    candidate_id=cid,
                    name=self._documents[cid].metadata.get("name", cid) if cid in self._documents else cid,
                    score=score,
                    evidence=[_chunk_out(h) for h in evidence[cid]],
                )
                for cid, score in ranked
            ],
        )

    def ask(self, request: AskRequest) -> AskResponse:
        started = time.perf_counter()
        meter = CostMeter(self._gateway)
        decider = make_deciders(self._config, meter)[request.system_1 or self._config.system_1]
        trace: list[System1TraceEntry] = []

        def respond(**fields) -> AskResponse:
            defaults = dict(answer=None, refused=False, refusal_reason=None, citations=[], grounded=None, evidence=[])
            return AskResponse(
                **{**defaults, **fields},
                system_1_trace=[TraceOut(**vars(t)) for t in trace],
                cost_usd=meter.cost_usd,
                latency_s=time.perf_counter() - started,
            )

        guardrail = decider.input_guardrail(request.question)
        trace.append(guardrail.trace)
        if not guardrail.allowed:
            return respond(refused=True, refusal_reason=_refusal_reason(guardrail.verdict))

        collections = request.collections
        if not collections:
            route = decider.route(request.question)
            trace.append(route.trace)
            collections = [route.collection]

        hits = self._retrieve(meter, request.question, collections, request.filters, self._config.top_k, True)
        relevance = decider.chunk_relevance(request.question, [h.chunk.text for h in hits])
        trace.append(relevance.trace)
        kept = [h for h, keep in zip(hits, relevance.keep) if keep]
        evidence = [_chunk_out(h) for h in kept]
        if not kept:
            return respond(answer=NO_EVIDENCE, grounded=None)

        sources = "\n\n".join(f"[{n}] {h.chunk.text}" for n, h in enumerate(kept, start=1))
        completion = meter.complete(
            self._config.models.system_2_model,
            f"Sources:\n\n{sources}\n\nQuestion: {request.question}",
            system=ANSWER_SYSTEM,
        )
        citations = _citations(completion.text, kept)
        grounded = False
        if citations:
            check = decider.groundedness(completion.text, [kept[c.n - 1].chunk.text for c in citations])
            trace.append(check.trace)
            grounded = check.grounded
        return respond(answer=completion.text, citations=citations, grounded=grounded, evidence=evidence)

    def _retrieve(
        self, gateway: ModelGateway, query: str, collections: list[str], filters: Filters, top_k: int, rerank: bool
    ) -> list[ScoredChunk]:
        limit = max(top_k, self._config.rerank_candidates) if rerank else top_k
        hits = hybrid_search(
            query, collections, self._filters(filters), limit,
            config=self._config, gateway=gateway, qdrant=self._qdrant,
        )
        if rerank and hits:
            scores = gateway.rerank(self._config.models.reranker, query, [h.chunk.text for h in hits]).scores
            hits = [ScoredChunk(h.chunk, s) for h, s in sorted(zip(hits, scores), key=lambda p: p[1], reverse=True)]
        return hits[:top_k]

    def _filters(self, filters: Filters) -> SearchFilters:
        for document_id, collection in ((filters.candidate_id, "candidates"), (filters.job_id, "jobs")):
            if document_id is not None and self.document(document_id).collection != collection:
                raise UnknownDocument(document_id)
        # Consent is enforced on every Candidate search (Data Protection and Consent Policy).
        return SearchFilters(**filters.model_dump(), consent_valid_on=self._today())


def _chunk_out(hit: ScoredChunk) -> ChunkOut:
    c = hit.chunk
    return ChunkOut(
        chunk_id=c.id, document_id=c.document_id, collection=c.collection, section=c.section,
        text=c.text, score=hit.score, metadata=c.metadata,
    )


def _citations(text: str, kept: list[ScoredChunk]) -> list[Citation]:
    """Citations in first-cited order; numbers outside the sources are ignored."""
    citations = []
    for n in dict.fromkeys(int(n) for n in re.findall(r"\[(\d+)\]", text)):
        if 1 <= n <= len(kept):
            citations.append(Citation(n=n, chunk_id=kept[n - 1].chunk.id, document_id=kept[n - 1].chunk.document_id))
    return citations


def _refusal_reason(verdict: str) -> str:
    if verdict == "discriminatory":
        return (
            "Refused by the Input guardrail as discriminatory: it selects or excludes people by a protected "
            "characteristic or a proxy for one (Fair Hiring Policy). Use job-related criteria instead."
        )
    return "Refused by the Input guardrail as out_of_scope: it is not a recruiting question for SwiftAI."
