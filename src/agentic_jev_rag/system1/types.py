"""The System 1 decisions, their typed outcomes and trace entries."""

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


Intelligibility = Literal["answer", "unintelligible", "off_topic"]
Depth = Literal["substantive", "shallow"]
TURN_LABELS = ("substantive", "shallow", "unintelligible", "off_topic", "injection")
INJECTION_THRESHOLD = 0.5


def turn_label(intelligibility: str, injection: float, depth: str) -> str:
    """The one outcome the interview acts on. A prompt injection outranks everything, so it is always flagged."""
    if injection >= INJECTION_THRESHOLD:
        return "injection"
    if intelligibility != "answer":
        return intelligibility
    return depth


@dataclass(frozen=True)
class TurnCheckDecision:
    intelligibility: Intelligibility
    injection: float
    depth: Depth
    trace: System1TraceEntry

    @property
    def label(self) -> str:
        return turn_label(self.intelligibility, self.injection, self.depth)


class System1Decider(Protocol):
    name: str

    def route(self, question: str) -> RouteDecision: ...

    def chunk_relevance(self, question: str, chunks: list[str]) -> RelevanceDecision: ...

    def input_guardrail(self, text: str) -> GuardrailDecision: ...

    def groundedness(self, claim: str, evidence: list[str]) -> GroundednessDecision: ...

    def turn_check(self, question: str, answer: str) -> TurnCheckDecision: ...
