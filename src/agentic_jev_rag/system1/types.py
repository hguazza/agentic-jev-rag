"""The four System 1 decisions, their typed outcomes and trace entries."""

from dataclasses import dataclass
from typing import Literal, Protocol

GuardrailVerdict = Literal["allowed", "discriminatory", "out_of_scope"]


class System1Error(Exception):
    """A System 1 decider returned something that is not a valid decision."""


@dataclass(frozen=True)
class System1TraceEntry:
    decision: str
    outcome: str
    latency_s: float
    cost_usd: float
    implementation: str


@dataclass(frozen=True)
class RouteDecision:
    collection: str
    trace: System1TraceEntry


@dataclass(frozen=True)
class RelevanceDecision:
    keep: list[bool]
    scores: list[float]
    trace: System1TraceEntry


@dataclass(frozen=True)
class GuardrailDecision:
    verdict: GuardrailVerdict
    trace: System1TraceEntry

    @property
    def allowed(self) -> bool:
        return self.verdict == "allowed"


@dataclass(frozen=True)
class GroundednessDecision:
    grounded: bool
    score: float
    trace: System1TraceEntry


class System1Decider(Protocol):
    name: str

    def route(self, question: str) -> RouteDecision: ...

    def chunk_relevance(self, question: str, chunks: list[str]) -> RelevanceDecision: ...

    def input_guardrail(self, text: str) -> GuardrailDecision: ...

    def groundedness(self, claim: str, evidence: list[str]) -> GroundednessDecision: ...
