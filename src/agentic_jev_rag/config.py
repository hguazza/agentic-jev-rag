"""Run configuration: which model fills each Model role, and retrieval settings."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """OpenRouter model ID for each Model role."""

    system_2_model: str
    embedding_model: str


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
