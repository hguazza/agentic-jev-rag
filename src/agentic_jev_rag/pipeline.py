"""Level pipeline: `answer(question, level, config)` -> AnswerRecord."""

import re
from dataclasses import dataclass, field

from agentic_jev_rag.config import Config
from agentic_jev_rag.gateway import ModelGateway
from agentic_jev_rag.indexing import RetrievedChunk, dense_search

SYSTEM_PROMPT = (
    "You answer questions about EU law using only the numbered sources provided. "
    "Cite every claim with the source number in square brackets, e.g. [1]. "
    "If the sources do not answer the question, say so."
)


@dataclass(frozen=True)
class Citation:
    chunk_id: str
    document_id: str
    collection: str


@dataclass(frozen=True)
class System1TraceEntry:
    """One System 1 decision made while answering: Route, Chunk relevance, Input guardrail or Groundedness check."""

    decision: str
    outcome: str
    latency_s: float
    cost_usd: float
    implementation: str


@dataclass(frozen=True)
class AnswerRecord:
    question: str
    level: int
    answer: str
    citations: list[Citation]
    retrieved_chunk_ids: list[str]
    system_1_trace: list[System1TraceEntry] = field(default_factory=list)


def answer(question: str, level: int, config: Config, gateway: ModelGateway) -> AnswerRecord:
    if level != 1:
        raise NotImplementedError(f"Level {level} is not built yet")

    retrieved = dense_search(question, config.collections, config, gateway)
    completion = gateway.complete(
        config.models.system_2_model, _prompt(question, retrieved), system=SYSTEM_PROMPT
    )
    return AnswerRecord(
        question=question,
        level=level,
        answer=completion.text,
        citations=_citations(completion.text, retrieved),
        retrieved_chunk_ids=[r.chunk.id for r in retrieved],
    )


def _prompt(question: str, retrieved: list[RetrievedChunk]) -> str:
    sources = "\n\n".join(
        f"[{n}] ({r.chunk.document_id})\n{r.chunk.text}" for n, r in enumerate(retrieved, start=1)
    )
    return f"Sources:\n\n{sources}\n\nQuestion: {question}"


def _citations(text: str, retrieved: list[RetrievedChunk]) -> list[Citation]:
    """Citations in the order the answer first cites them; numbers outside the sources are ignored."""
    cited: list[Citation] = []
    for number in dict.fromkeys(int(n) for n in re.findall(r"\[(\d+)\]", text)):
        if 1 <= number <= len(retrieved):
            chunk = retrieved[number - 1].chunk
            cited.append(Citation(chunk.id, chunk.document_id, chunk.collection))
    return cited
