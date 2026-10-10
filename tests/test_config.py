"""Model config: each Model role is named in a config file, not in code."""

from pathlib import Path

import pytest

from agentic_jev_rag.config import VoiceConfig, load_config

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


MODELS_TOML = """
[models]
system_2_model = "a/s2"
system_1_baseline = "a/s1"
reference_judge = "b/judge"
embedding_model = "c/embed"
reranker = "d/rerank"
jev_model = "typesafe/jev"
voice_host = "a/host"
"""


def test_voice_settings_have_defaults_when_the_file_has_no_voice_table(tmp_path, monkeypatch):
    monkeypatch.delenv("RECRUITER_URL", raising=False)
    path = tmp_path / "config.toml"
    path.write_text(MODELS_TOML)

    config = load_config(path)

    assert config.models.voice_host == "a/host"
    assert config.voice == VoiceConfig()
    assert config.voice.max_reprompts == 2 and config.voice.max_probes_per_topic == 2


def test_voice_table_and_recruiter_url_override(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    path.write_text(MODELS_TOML + '\n[voice]\nstt = "deepgram/nova-2"\ndefault_job_id = "J-001"\n')
    monkeypatch.setenv("RECRUITER_URL", "http://recruiter:8001")

    voice = load_config(path).voice

    assert voice.stt == "deepgram/nova-2"
    assert voice.default_job_id == "J-001"
    assert voice.recruiter_url == "http://recruiter:8001"


def test_aws_settings_can_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("QDRANT_API_KEY", "qdrant-secret")
    monkeypatch.setenv("PERSISTENCE", "dynamodb")
    monkeypatch.setenv("CHECKPOINT_TABLE", "swiftai-checkpoints")
    monkeypatch.setenv("RUNS_TABLE", "swiftai-runs")
    monkeypatch.setenv("DYNAMODB_ENDPOINT", "http://localhost:8002")
    monkeypatch.setenv("RECRUITER_API_KEY", "api-secret")

    config = load_config(RECRUITER)

    assert config.qdrant_api_key == "qdrant-secret"
    assert config.persistence == "dynamodb"
    assert (config.checkpoint_table, config.runs_table) == ("swiftai-checkpoints", "swiftai-runs")
    assert config.dynamodb_endpoint == "http://localhost:8002"
    assert config.api_key == "api-secret"
    assert "qdrant-secret" not in repr(config) and "api-secret" not in repr(config)


def test_persistence_defaults_to_memory_and_rejects_unknown_backends(monkeypatch):
    monkeypatch.delenv("PERSISTENCE", raising=False)
    assert load_config(RECRUITER).persistence == "memory"

    monkeypatch.setenv("PERSISTENCE", "redis")
    with pytest.raises(ValueError, match="persistence"):
        load_config(RECRUITER)
