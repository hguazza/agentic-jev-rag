"""Live smoke suite: real OpenRouter calls. Opt in with `uv run pytest -m live`."""

import pytest

from agentic_jev_rag.config import load_config
from agentic_jev_rag.openrouter import OpenRouterGateway

pytestmark = pytest.mark.live

CONFIG = load_config()
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
    embeddings = gateway.embed(CONFIG.models.embedding_model, ["Backend engineer", "Recruiter"])

    assert len(embeddings.vectors) == 2
    assert len(embeddings.vectors[0]) > 0
    assert embeddings.cost_usd > 0
