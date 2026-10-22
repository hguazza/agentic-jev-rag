"""Hybrid index in Qdrant: one collection per Collection, a dense and a sparse vector per chunk, RRF fusion."""

import uuid
from dataclasses import asdict, dataclass
from datetime import date
from typing import Sequence

from qdrant_client import QdrantClient, models

from agentic_recruiter.config import Config
from agentic_recruiter.corpus.documents import Document
from agentic_recruiter.gateway import ModelGateway
from agentic_recruiter.rag.chunking import Chunk, chunk_document
from agentic_recruiter.rag.sparse import sparse_vector

DENSE = "dense"
SPARSE = "sparse"
EMBED_BATCH = 64


@dataclass(frozen=True)
class SearchFilters:
    occupation: str | None = None
    location: str | None = None
    min_years_experience: float | None = None
    candidate_id: str | None = None
    job_id: str | None = None
    consent_valid_on: date | None = None


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float


def qdrant_name(config: Config, collection: str) -> str:
    return f"{config.collection_prefix}{collection}"


def index_corpus(
    documents: list[Document], config: Config, gateway: ModelGateway, qdrant: QdrantClient
) -> dict[str, int]:
    """(Re)build every Collection present in `documents`. Returns the chunk count per Collection."""
    chunks_by_collection: dict[str, list[Chunk]] = {}
    for document in documents:
        chunks_by_collection.setdefault(document.collection, []).extend(
            chunk_document(document, config.chunk_size, config.chunk_overlap)
        )

    counts = {}
    for collection, chunks in chunks_by_collection.items():
        vectors: list[list[float]] = []
        for start in range(0, len(chunks), EMBED_BATCH):
            batch = [c.text for c in chunks[start : start + EMBED_BATCH]]
            vectors += gateway.embed(config.models.embedding_model, batch).vectors
        name = qdrant_name(config, collection)
        _recreate(qdrant, name, dimensions=len(vectors[0]))
        qdrant.upload_points(
            name,
            points=[
                models.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.id)),
                    vector={DENSE: vector, SPARSE: sparse_vector(chunk.text)},
                    payload=_payload(chunk),
                )
                for chunk, vector in zip(chunks, vectors)
            ],
            wait=True,
        )
        counts[collection] = len(chunks)
    return counts


def hybrid_search(
    query: str,
    collections: Sequence[str],
    filters: SearchFilters,
    limit: int,
    *,
    config: Config,
    gateway: ModelGateway,
    qdrant: QdrantClient,
) -> list[ScoredChunk]:
    """Top `limit` chunks across `collections`, fusing dense and sparse rankings with RRF."""
    [dense] = gateway.embed(config.models.embedding_model, [query]).vectors
    sparse = sparse_vector(query)
    prefetch_limit = max(limit * 3, 20)
    hits: list[ScoredChunk] = []
    for collection in collections:
        condition = _filter(collection, filters)
        points = qdrant.query_points(
            qdrant_name(config, collection),
            prefetch=[
                models.Prefetch(query=dense, using=DENSE, limit=prefetch_limit, filter=condition),
                models.Prefetch(query=sparse, using=SPARSE, limit=prefetch_limit, filter=condition),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        ).points
        hits += [ScoredChunk(_chunk(p.payload), p.score) for p in points]
    return sorted(hits, key=lambda h: h.score, reverse=True)[:limit]


def _recreate(qdrant: QdrantClient, name: str, dimensions: int) -> None:
    if qdrant.collection_exists(name):
        qdrant.delete_collection(name)
    qdrant.create_collection(
        name,
        vectors_config={DENSE: models.VectorParams(size=dimensions, distance=models.Distance.COSINE)},
        sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    qdrant.create_payload_index(
        name,
        "metadata.location",
        field_schema=models.TextIndexParams(
            type=models.TextIndexType.TEXT, tokenizer=models.TokenizerType.WORD, lowercase=True
        ),
    )


def _payload(chunk: Chunk) -> dict:
    payload = asdict(chunk)
    if until := chunk.metadata.get("consent_until"):
        payload["consent_until_ord"] = date.fromisoformat(until).toordinal()
    return payload


def _chunk(payload: dict) -> Chunk:
    return Chunk(**{k: payload[k] for k in ("id", "document_id", "collection", "section", "text", "metadata")})


def _filter(collection: str, filters: SearchFilters) -> models.Filter | None:
    """Conditions for the filter fields this Collection has; others are ignored for it."""
    must: list[models.Condition] = []

    def equals(key: str, value: str) -> None:
        must.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))

    if collection in ("candidates", "jobs"):
        if filters.occupation:
            equals("metadata.occupation", filters.occupation)
        if filters.location:
            must.append(models.FieldCondition(key="metadata.location", match=models.MatchText(text=filters.location)))
    if collection == "candidates":
        if filters.min_years_experience is not None:
            must.append(
                models.FieldCondition(
                    key="metadata.years_experience", range=models.Range(gte=filters.min_years_experience)
                )
            )
        if filters.consent_valid_on:
            must.append(
                models.FieldCondition(
                    key="consent_until_ord", range=models.Range(gte=filters.consent_valid_on.toordinal())
                )
            )
        if filters.candidate_id:
            equals("document_id", filters.candidate_id)
    if collection == "jobs" and filters.job_id:
        equals("document_id", filters.job_id)
    if collection == "interviews":
        if filters.candidate_id:
            equals("metadata.candidate_id", filters.candidate_id)
        if filters.job_id:
            equals("metadata.job_id", filters.job_id)
    return models.Filter(must=must) if must else None
