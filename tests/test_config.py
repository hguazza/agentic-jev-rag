"""Model config: each Model role is named in a config file, not in code."""

from pathlib import Path

from agentic_jev_rag.config import load_config

RECRUITER = Path(__file__).parents[1] / "configs" / "recruiter.toml"


def test_recruiter_config_names_an_openrouter_model_for_every_model_role():
    models = load_config(RECRUITER).models

    assert models.system_2_model == "anthropic/claude-sonnet-5.5"
    assert models.system_1_baseline == "anthropic/claude-haiku-4.5"
    assert models.reference_judge == "openai/gpt-5.5"
    assert models.embedding_model == "baai/bge-m3"
    assert models.reranker == "cohere/rerank-4-fast"
    assert models.jev_model == "typesafe/jev-1.13"


def test_recruiter_config_searches_the_four_collections_with_the_llm_decider_by_default():
    config = load_config(RECRUITER)

    assert config.collections == ("candidates", "jobs", "interviews", "policies")
    assert config.system_1 == "llm"


def test_swapping_a_model_is_a_config_edit(tmp_path):
    edited = tmp_path / "swap.toml"
    edited.write_text(
        RECRUITER.read_text().replace("anthropic/claude-haiku-4.5", "openai/gpt-5.4-nano")
    )

    assert load_config(edited).models.system_1_baseline == "openai/gpt-5.4-nano"


def test_service_urls_and_system_1_can_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")
    monkeypatch.setenv("RAG_URL", "http://rag-api:8000")
    monkeypatch.setenv("SYSTEM_1", "jev")

    config = load_config(RECRUITER)

    assert config.qdrant_url == "http://qdrant:6333"
    assert config.rag_url == "http://rag-api:8000"
    assert config.system_1 == "jev"


def test_settings_in_a_dotenv_file_are_honoured_when_running_on_the_host(tmp_path, monkeypatch):
    import os

    (tmp_path / ".env").write_text("SYSTEM_1=jev\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SYSTEM_1", raising=False)
    try:
        assert load_config(RECRUITER).system_1 == "jev"
    finally:
        os.environ.pop("SYSTEM_1", None)
