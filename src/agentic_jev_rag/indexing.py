"""Dense index in Qdrant: one Qdrant collection per Collection x Language x chunking strategy."""

import uuid
from collections import defaultdict
from dataclasses import dataclass

from qdrant_client import QdrantClient, models

from agentic_jev_rag.chunking import Chunk, fixed_size_chunks
from agentic_jev_rag.config import Config
from agentic_jev_rag.corpus import Document
from agentic_jev_rag.gateway import ModelGateway

FIXED = "fixed"
EMBED_BATCH = 64


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float


def qdrant_collection_name(config: Config, collection: str, strategy: str = FIXED) -> str:
    return f"{config.collection_prefix}{collection}_{config.language}_{strategy}"


def index_documents(documents: list[Document], config: Config, gateway: ModelGateway) -> None:
    """(Re)build the fixed-size dense index for each Collection in `documents` of the configured Language."""
    client = QdrantClient(url=config.qdrant_url)
    by_collection: dict[str, list[Chunk]] = defaultdict(list)
    for document in documents:
        if document.language == config.language:
            by_collection[document.collection] += fixed_size_chunks(
                document, config.chunk_size, config.chunk_overlap
            )

    for collection, chunks in by_collection.items():
        vectors: list[list[float]] = []
        for i in range(0, len(chunks), EMBED_BATCH):
            batch = [c.text for c in chunks[i : i + EMBED_BATCH]]
            vectors += gateway.embed(config.models.embedding_model, batch).vectors

        name = qdrant_collection_name(config, collection)
        if client.collection_exists(name):
            client.delete_collection(name)
        client.create_collection(
            name,
            vectors_config=models.VectorParams(size=len(vectors[0]), distance=models.Distance.COSINE),
        )
        client.upload_points(
            name,
            points=[
                models.PointStruct(id=_point_id(c), vector=v, payload=c.__dict__)
                for c, v in zip(chunks, vectors)
            ],
            wait=True,
        )


def dense_search(
    question: str, collections: tuple[str, ...], config: Config, gateway: ModelGateway
) -> list[RetrievedChunk]:
    """Top-k chunks for `question` across `collections`, best first."""
    client = QdrantClient(url=config.qdrant_url)
    [vector] = gateway.embed(config.models.embedding_model, [question]).vectors
    hits = [
        RetrievedChunk(Chunk(**point.payload), point.score)
        for collection in collections
        for point in client.query_points(
            qdrant_collection_name(config, collection), query=vector, limit=config.top_k
        ).points
    ]
    return sorted(hits, key=lambda h: h.score, reverse=True)[: config.top_k]


def _point_id(chunk: Chunk) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.id))
