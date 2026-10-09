import os
import uuid

import pytest
from qdrant_client import QdrantClient

from agentic_jev_rag.config import Config, ModelConfig

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")

FAKE_MODELS = ModelConfig(
    system_2_model="fake/system-2",
    system_1_baseline="fake/system-1-baseline",
    reference_judge="fake/reference-judge",
    embedding_model="fake/embedding",
    reranker="fake/reranker",
    jev_model="fake/jev",
)


@pytest.fixture(scope="session")
def qdrant() -> QdrantClient:
    client = QdrantClient(url=QDRANT_URL)
    try:
        client.get_collections()
    except Exception as exc:  # pragma: no cover - environment guard
        pytest.fail(f"Qdrant is not reachable at {QDRANT_URL}; run `docker compose up -d` ({exc})")
    return client


@pytest.fixture(scope="session")
def config(qdrant: QdrantClient):
    prefix = f"test_{uuid.uuid4().hex[:8]}_"
    yield Config(models=FAKE_MODELS, qdrant_url=QDRANT_URL, collection_prefix=prefix)
    for c in qdrant.get_collections().collections:
        if c.name.startswith(prefix):
            qdrant.delete_collection(c.name)


FIXTURE_CORPUS = __import__("pathlib").Path(__file__).parent / "fixtures" / "corpus"


@pytest.fixture(scope="session")
def fixture_documents():
    from agentic_jev_rag.corpus.documents import load_corpus

    return load_corpus(FIXTURE_CORPUS)


@pytest.fixture(scope="session")
def indexed(config, qdrant, fixture_documents):
    """The fixture corpus indexed once per session with the fake gateway's embeddings."""
    from agentic_jev_rag.gateway import FakeModelGateway
    from agentic_jev_rag.rag.index import index_corpus

    return index_corpus(fixture_documents, config, FakeModelGateway(), qdrant)
