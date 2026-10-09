"""Run configuration: which model fills each Model role, and retrieval settings."""

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path


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
    language: str = "en"
    collections: tuple[str, ...] = ("case_law", "legislation")
    qdrant_url: str = "http://localhost:6333"
    collection_prefix: str = ""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5


def load_config(path: Path) -> Config:
    """Read a run config from TOML. `QDRANT_URL` in the environment overrides the file."""
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    models = ModelConfig(**data.pop("models"))
    if "collections" in data:
        data["collections"] = tuple(data["collections"])
    if url := os.environ.get("QDRANT_URL"):
        data["qdrant_url"] = url
    return Config(models=models, **data)
