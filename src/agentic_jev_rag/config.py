"""Run configuration: which model fills each Model role, and service settings."""

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).parents[2] / "configs" / "recruiter.toml"
ENVIRONMENT_OVERRIDES = {"QDRANT_URL": "qdrant_url", "RAG_URL": "rag_url", "SYSTEM_1": "system_1"}


@dataclass(frozen=True)
class ModelConfig:
    """OpenRouter model ID for each Model role."""

    system_2_model: str
    system_1_baseline: str
    reference_judge: str
    embedding_model: str
    reranker: str
    jev_model: str


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


def load_config(path: Path | None = None) -> Config:
    """Read a run config from TOML (default: `CONFIG` env or configs/recruiter.toml).

    `QDRANT_URL`, `RAG_URL` and `SYSTEM_1` in the environment override the file.
    """
    path = Path(path or os.environ.get("CONFIG") or DEFAULT_CONFIG)
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    models = ModelConfig(**data.pop("models"))
    if "collections" in data:
        data["collections"] = tuple(data["collections"])
    for variable, field in ENVIRONMENT_OVERRIDES.items():
        if value := os.environ.get(variable):
            data[field] = value
    return Config(models=models, **data)
