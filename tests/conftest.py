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


TODAY = __import__("datetime").date(2026, 10, 8)


@pytest.fixture
def recruiter_context(indexed, config, qdrant, fixture_documents):
    """Builds a RecruiterContext whose RAG API runs in-process over the indexed fixture corpus."""
    from fastapi.testclient import TestClient

    from agentic_jev_rag.rag.api import create_rag_app
    from agentic_jev_rag.rag.service import RagService
    from agentic_jev_rag.recruiter.context import RecruiterContext
    from agentic_jev_rag.recruiter.rag_client import RagClient
    from tests.rag_scripts import scripted_gateway

    def make(gateway, *, rag_gateway=None, system_1="llm"):
        service = RagService(config, rag_gateway or scripted_gateway(), qdrant, fixture_documents, today=lambda: TODAY)
        rag = RagClient("http://rag", http=TestClient(create_rag_app(service)))
        return RecruiterContext(config=config, gateway=gateway, rag=rag, system_1=system_1, today=lambda: TODAY)

    return make
