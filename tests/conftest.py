import os
import uuid
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from agentic_jev_rag.config import Config, ModelConfig
from agentic_jev_rag.corpus import load_documents
from agentic_jev_rag.gateway import FakeModelGateway
from agentic_jev_rag.indexing import index_documents

FIXTURE_CORPUS = Path(__file__).parent / "fixtures" / "corpus"
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")


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
    yield Config(
        models=ModelConfig(system_2_model="fake/system-2", embedding_model="fake/embedding"),
        qdrant_url=QDRANT_URL,
        collection_prefix=prefix,
        collections=("case_law",),
    )
    for c in qdrant.get_collections().collections:
        if c.name.startswith(prefix):
            qdrant.delete_collection(c.name)


@pytest.fixture
def gateway() -> FakeModelGateway:
    return FakeModelGateway()


@pytest.fixture(scope="session")
def indexed_case_law(config: Config) -> None:
    index_documents(load_documents(FIXTURE_CORPUS), config, FakeModelGateway())
