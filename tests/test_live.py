"""Live smoke suite: real OpenRouter calls. Opt in with `uv run pytest -m live`."""

import uuid
from dataclasses import replace
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from agentic_jev_rag.config import load_config
from agentic_jev_rag.corpus import load_documents
from agentic_jev_rag.indexing import index_documents
from agentic_jev_rag.openrouter import OpenRouterGateway
from agentic_jev_rag.pipeline import answer

pytestmark = pytest.mark.live

ROOT = Path(__file__).parents[1]
CONFIG = load_config(ROOT / "configs" / "en.toml")
CHAT_ROLES = ["system_2_model", "system_1_baseline", "reference_judge"]


@pytest.fixture(scope="module")
def gateway() -> OpenRouterGateway:
    return OpenRouterGateway.from_env()


@pytest.mark.parametrize("role", CHAT_ROLES)
def test_chat_model_role_answers_and_reports_cost(gateway, role):
    completion = gateway.complete(
        getattr(CONFIG.models, role), "Reply with the single word: ready", system="Be terse."
    )

    assert "ready" in completion.text.lower()
    assert completion.cost_usd > 0
    assert completion.latency_s > 0


def test_embedding_model_role_embeds_and_reports_cost(gateway):
    embeddings = gateway.embed(CONFIG.models.embedding_model, ["Court of Justice", "Tribunal"])

    assert len(embeddings.vectors) == 2
    assert len(embeddings.vectors[0]) > 0
    assert embeddings.cost_usd > 0


def test_level_1_answers_end_to_end_on_real_models(gateway):
    config = replace(
        CONFIG, collections=("case_law",), collection_prefix=f"live_{uuid.uuid4().hex[:8]}_"
    )
    try:
        index_documents(load_documents(ROOT / "tests" / "fixtures" / "corpus"), config, gateway)

        record = answer(
            "Did the Court declare the Safe Harbour adequacy decision 2000/520 invalid?",
            level=1,
            config=config,
            gateway=gateway,
        )
    finally:
        qdrant = QdrantClient(url=config.qdrant_url)
        for c in qdrant.get_collections().collections:
            if c.name.startswith(config.collection_prefix):
                qdrant.delete_collection(c.name)

    print(record.answer)
    assert record.citations
    assert any(c.document_id == "62014CJ0362" for c in record.citations)
