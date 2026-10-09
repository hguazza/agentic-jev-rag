"""Model config: each Model role is named in a config file, not in code."""

from pathlib import Path

from agentic_jev_rag.config import load_config

ENGLISH = Path(__file__).parents[1] / "configs" / "en.toml"


def test_english_config_names_an_openrouter_model_for_every_model_role():
    models = load_config(ENGLISH).models

    assert models.system_2_model == "anthropic/claude-sonnet-5.5"
    assert models.system_1_baseline == "anthropic/claude-haiku-4.5"
    assert models.reference_judge == "openai/gpt-5.5"
    assert models.embedding_model == "baai/bge-m3"
    assert models.reranker == "cohere/rerank-4-fast"
    assert models.jev_model == "typesafe/jev-1.13"


def test_swapping_a_model_is_a_config_edit(tmp_path):
    edited = tmp_path / "swap.toml"
    edited.write_text(
        ENGLISH.read_text().replace("anthropic/claude-haiku-4.5", "openai/gpt-5.4-nano")
    )

    assert load_config(edited).models.system_1_baseline == "openai/gpt-5.4-nano"


def test_qdrant_url_can_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")

    assert load_config(ENGLISH).qdrant_url == "http://qdrant:6333"
