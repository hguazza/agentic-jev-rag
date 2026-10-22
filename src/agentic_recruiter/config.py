"""Run configuration: which model fills each Model role, and service settings."""

import os
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

DEFAULT_CONFIG = Path(__file__).parents[2] / "configs" / "recruiter.toml"
ENVIRONMENT_OVERRIDES = {
    "QDRANT_URL": "qdrant_url",
    "QDRANT_API_KEY": "qdrant_api_key",
    "RAG_URL": "rag_url",
    "SYSTEM_1": "system_1",
    "PERSISTENCE": "persistence",
    "CHECKPOINT_TABLE": "checkpoint_table",
    "RUNS_TABLE": "runs_table",
    "DYNAMODB_ENDPOINT": "dynamodb_endpoint",
    "RECRUITER_API_KEY": "api_key",
}


@dataclass(frozen=True)
class ModelConfig:
    """OpenRouter model ID for each Model role."""

    system_2_model: str
    system_1_baseline: str
    reference_judge: str
    embedding_model: str
    reranker: str
    jev_model: str
    voice_host: str


@dataclass(frozen=True)
class VoiceConfig:
    """The voice worker (LiveKit Inference model ids, which interview to hold) and the live interview's turn budgets."""

    stt: str = "deepgram/nova-3"
    tts: str = "cartesia/sonic-3"
    default_candidate_id: str = "C-0040"
    default_job_id: str = "J-037"
    recruiter_url: str = "http://localhost:8001"
    max_reprompts: int = 2
    max_probes_per_topic: int = 2


@dataclass(frozen=True)
class Config:
    models: ModelConfig
    collections: tuple[str, ...] = ("candidates", "jobs", "interviews", "policies")
    qdrant_url: str = "http://localhost:6333"
    collection_prefix: str = ""
    corpus_dir: str = "corpus"
    chunk_size: int = 1200
    chunk_overlap: int = 150
    top_k: int = 8
    rerank_candidates: int = 30
    system_1: str = "llm"
    rag_url: str = "http://localhost:8000"
    qdrant_api_key: str = field(default="", repr=False)  # Qdrant Cloud; empty for local Qdrant
    persistence: str = "memory"  # where interview checkpoints and run records live: memory or dynamodb
    checkpoint_table: str = "agentic-recruiter-checkpoints"
    runs_table: str = "agentic-recruiter-runs"
    dynamodb_endpoint: str = ""  # DynamoDB Local in tests; empty means AWS
    api_key: str = field(default="", repr=False)  # the Recruiter API's X-API-Key; empty turns auth off
    voice: VoiceConfig = field(default_factory=VoiceConfig)


def load_config(path: Path | None = None) -> Config:
    """Read a run config from TOML (default: `CONFIG` env or configs/recruiter.toml).

    `QDRANT_URL`, `QDRANT_API_KEY`, `RAG_URL`, `SYSTEM_1`, `PERSISTENCE`, `CHECKPOINT_TABLE`, `RUNS_TABLE`,
    `DYNAMODB_ENDPOINT`, `RECRUITER_API_KEY` and `RECRUITER_URL` in the environment override the file.
    """
    load_dotenv(find_dotenv(usecwd=True))  # never overrides variables already set
    path = Path(path or os.environ.get("CONFIG") or DEFAULT_CONFIG)
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    models = ModelConfig(**data.pop("models"))
    voice = VoiceConfig(**data.pop("voice", {}))
    if url := os.environ.get("RECRUITER_URL"):
        voice = replace(voice, recruiter_url=url)
    if "collections" in data:
        data["collections"] = tuple(data["collections"])
    for variable, key in ENVIRONMENT_OVERRIDES.items():
        if value := os.environ.get(variable):
            data[key] = value
    if data.get("system_1", "llm") not in ("llm", "jev"):
        raise ValueError(f"system_1 must be 'llm' or 'jev', got {data['system_1']!r}")
    if data.get("persistence", "memory") not in ("memory", "dynamodb"):
        raise ValueError(f"persistence must be 'memory' or 'dynamodb', got {data['persistence']!r}")
    return Config(models=models, voice=voice, **data)
