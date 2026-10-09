"""Model gateway: the only module that talks to models, and the single fake boundary in tests."""

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence


@dataclass(frozen=True)
class Completion:
    text: str
    latency_s: float
    cost_usd: float
    retries: tuple[str, ...] = ()


@dataclass(frozen=True)
class Embeddings:
    vectors: list[list[float]]
    latency_s: float
    cost_usd: float
    retries: tuple[str, ...] = ()


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass(frozen=True)
class ChatResult:
    """One assistant turn: text, tool calls, or both."""

    content: str | None
    tool_calls: tuple[ToolCall, ...]
    latency_s: float
    cost_usd: float
    retries: tuple[str, ...] = ()

    def as_message(self) -> dict:
        """The OpenAI-format assistant message, to append to the conversation."""
        message: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
                }
                for call in self.tool_calls
            ]
        return message


@dataclass(frozen=True)
class RerankResult:
    scores: list[float]  # one relevance score per document, in input order
    latency_s: float
    cost_usd: float
    retries: tuple[str, ...] = ()


@dataclass(frozen=True)
class Decisions:
    """Jev's answers, keyed by question id."""

    answers: dict[str, dict]
    latency_s: float
    cost_usd: float
    retries: tuple[str, ...] = ()


class ModelGateway(Protocol):
    def complete(
        self,
        model: str,
        prompt: str,
        *,
        system: str | None = None,
        response_schema: tuple[str, dict] | None = None,
    ) -> Completion:
        """Text completion. `response_schema` is (name, JSON schema) for strict structured output."""
        ...

    def chat(self, model: str, messages: list[dict], *, tools: list[dict] | None = None) -> ChatResult:
        """One assistant turn over OpenAI-format messages, optionally offering tools."""
        ...

    def embed(self, model: str, texts: Sequence[str]) -> Embeddings: ...

    def rerank(self, model: str, query: str, documents: list[str]) -> RerankResult: ...

    def decide(self, model: str, state: str | dict, questions: dict[str, dict]) -> Decisions:
        """Ask Jev typed questions (noul, choice, score) about `state`."""
        ...


ChatResponder = Callable[[str, list[dict], list[dict] | None], "str | list[ToolCall]"]
CompleteResponder = Callable[[str, str, "str | None", "tuple[str, dict] | None"], str]
DecideResponder = Callable[["str | dict", dict], dict]


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _default_answer(question: dict) -> dict:
    kind = question["type"]
    if kind == "noul":
        return {"type": "noul", "noul": 0.0}
    if kind == "choice":
        first = next(iter(question["criteria"]))
        return {
            "type": "choice",
            "choice": first,
            "probabilities": {label: float(label == first) for label in question["criteria"]},
            "confidence": 1.0,
        }
    return {"type": "score", "score": 0, "probabilities": {"0": 1.0}, "confidence": 1.0}


@dataclass
class FakeModelGateway:
    """Deterministic stand-in for OpenRouter.

    - Embeddings hash each word into `dimensions` buckets, so texts sharing words are close.
    - Rerank scores are word-overlap (Jaccard) between query and document.
    - Chat, complete and decide answers come from optional scripted responders.
    Every call is recorded in `calls` as (kind, model, payload).
    """

    chat_responder: ChatResponder | None = None
    complete_responder: CompleteResponder | None = None
    decide_responder: DecideResponder | None = None
    dimensions: int = 512
    seed: str = ""
    calls: list[tuple[str, str, Any]] = field(default_factory=list)

    def complete(
        self,
        model: str,
        prompt: str,
        *,
        system: str | None = None,
        response_schema: tuple[str, dict] | None = None,
    ) -> Completion:
        self.calls.append(("complete", model, {"prompt": prompt, "system": system, "schema": response_schema}))
        if self.complete_responder:
            text = self.complete_responder(model, prompt, system, response_schema)
        else:
            text = "The documents answer this [1]."
        return Completion(text, latency_s=0.0, cost_usd=0.0)

    def chat(self, model: str, messages: list[dict], *, tools: list[dict] | None = None) -> ChatResult:
        self.calls.append(("chat", model, {"messages": list(messages), "tools": tools}))
        reply = self.chat_responder(model, messages, tools) if self.chat_responder else "OK"
        if isinstance(reply, str):
            return ChatResult(reply, (), latency_s=0.0, cost_usd=0.0)
        return ChatResult(None, tuple(reply), latency_s=0.0, cost_usd=0.0)

    def embed(self, model: str, texts: Sequence[str]) -> Embeddings:
        self.calls.append(("embed", model, list(texts)))
        return Embeddings([self._embed_one(t) for t in texts], latency_s=0.0, cost_usd=0.0)

    def rerank(self, model: str, query: str, documents: list[str]) -> RerankResult:
        self.calls.append(("rerank", model, {"query": query, "documents": list(documents)}))
        query_words = set(_words(query))
        scores = []
        for document in documents:
            words = set(_words(document))
            union = query_words | words
            scores.append(len(query_words & words) / len(union) if union else 0.0)
        return RerankResult(scores, latency_s=0.0, cost_usd=0.0)

    def decide(self, model: str, state: str | dict, questions: dict[str, dict]) -> Decisions:
        self.calls.append(("decide", model, {"state": state, "questions": questions}))
        if self.decide_responder:
            answers = self.decide_responder(state, questions)
        else:
            answers = {qid: _default_answer(q) for qid, q in questions.items()}
        return Decisions(answers, latency_s=0.0, cost_usd=0.0)

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for word in _words(text):
            digest = hashlib.blake2b((self.seed + word).encode(), digest_size=4).digest()
            vector[int.from_bytes(digest) % self.dimensions] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]


class CostMeter:
    """Wraps a gateway and totals the reported cost of every call made through it."""

    def __init__(self, gateway: ModelGateway):
        self._gateway = gateway
        self.cost_usd = 0.0

    def _add(self, result):
        self.cost_usd += result.cost_usd
        return result

    def complete(self, model, prompt, *, system=None, response_schema=None) -> Completion:
        return self._add(self._gateway.complete(model, prompt, system=system, response_schema=response_schema))

    def chat(self, model, messages, *, tools=None) -> ChatResult:
        return self._add(self._gateway.chat(model, messages, tools=tools))

    def embed(self, model, texts) -> Embeddings:
        return self._add(self._gateway.embed(model, texts))

    def rerank(self, model, query, documents) -> RerankResult:
        return self._add(self._gateway.rerank(model, query, documents))

    def decide(self, model, state, questions) -> Decisions:
        return self._add(self._gateway.decide(model, state, questions))
