"""Model gateway: the only module that talks to models, and the single fake boundary in tests."""

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Protocol, Sequence


@dataclass(frozen=True)
class Completion:
    text: str
    latency_s: float
    cost_usd: float


@dataclass(frozen=True)
class Embeddings:
    vectors: list[list[float]]
    latency_s: float
    cost_usd: float


class ModelGateway(Protocol):
    def complete(self, model: str, prompt: str, *, system: str | None = None) -> Completion: ...

    def embed(self, model: str, texts: Sequence[str]) -> Embeddings: ...


@dataclass(frozen=True)
class CompletionCall:
    model: str
    system: str | None
    prompt: str


@dataclass
class FakeModelGateway:
    """Deterministic stand-in for OpenRouter.

    Embeddings hash each word into a fixed number of dimensions, so texts sharing
    words land close together and retrieval behaves sensibly without a real model.
    Completions return `answer_text` and are recorded in `completion_calls`.
    """

    answer_text: str = "The Court answered this question [1]."
    dimensions: int = 512
    completion_calls: list[CompletionCall] = field(default_factory=list)

    def complete(self, model: str, prompt: str, *, system: str | None = None) -> Completion:
        self.completion_calls.append(CompletionCall(model, system, prompt))
        return Completion(self.answer_text, latency_s=0.0, cost_usd=0.0)

    def embed(self, model: str, texts: Sequence[str]) -> Embeddings:
        return Embeddings([self._embed_one(t) for t in texts], latency_s=0.0, cost_usd=0.0)

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for word in re.findall(r"\w+", text.lower()):
            digest = hashlib.blake2b(word.encode(), digest_size=4).digest()
            vector[int.from_bytes(digest) % self.dimensions] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]
