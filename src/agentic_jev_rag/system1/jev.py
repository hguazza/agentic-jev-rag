"""The Jev decider: typed Choice and Noul questions to Jev through OpenRouter (ADR 0007)."""

from agentic_jev_rag.gateway import Decisions, ModelGateway
from agentic_jev_rag.system1.prompts import (
    GROUNDEDNESS_INSTRUCTIONS,
    GUARDRAIL_CRITERIA,
    GUARDRAIL_INSTRUCTIONS,
    RELEVANCE_INSTRUCTIONS,
    ROUTE_CRITERIA,
    ROUTE_INSTRUCTIONS,
)
from agentic_jev_rag.system1.types import (
    GroundednessDecision,
    GuardrailDecision,
    RelevanceDecision,
    RouteDecision,
    System1Error,
    System1TraceEntry,
)

THRESHOLD = 0.5


class JevDecider:
    name = "jev"

    def __init__(self, gateway: ModelGateway, model: str):
        self._gateway = gateway
        self._model = model

    def route(self, question: str) -> RouteDecision:
        decisions = self._decide(
            {"question": question},
            {"route": {"type": "choice", "instructions": ROUTE_INSTRUCTIONS, "criteria": ROUTE_CRITERIA}},
        )
        collection = self._choice(decisions, "route", ROUTE_CRITERIA)
        return RouteDecision(collection, self._trace("route", collection, decisions))

    def chunk_relevance(self, question: str, chunks: list[str]) -> RelevanceDecision:
        if not chunks:
            return RelevanceDecision([], [], System1TraceEntry("chunk_relevance", "kept 0/0", 0.0, 0.0, self.name))
        passages = {f"chunk_{i}": text for i, text in enumerate(chunks)}
        questions = {
            key: {"type": "noul", "instructions": f"{RELEVANCE_INSTRUCTIONS} The passage to judge is `{key}`."}
            for key in passages
        }
        decisions = self._decide({"question": question, "passages": passages}, questions)
        scores = [self._noul(decisions, key) for key in passages]
        keep = [score >= THRESHOLD for score in scores]
        return RelevanceDecision(keep, scores, self._trace("chunk_relevance", f"kept {sum(keep)}/{len(chunks)}", decisions))

    def input_guardrail(self, text: str) -> GuardrailDecision:
        decisions = self._decide(
            {"request": text},
            {"verdict": {"type": "choice", "instructions": GUARDRAIL_INSTRUCTIONS, "criteria": GUARDRAIL_CRITERIA}},
        )
        verdict = self._choice(decisions, "verdict", GUARDRAIL_CRITERIA)
        return GuardrailDecision(verdict, self._trace("input_guardrail", verdict, decisions))

    def groundedness(self, claim: str, evidence: list[str]) -> GroundednessDecision:
        decisions = self._decide(
            {"statement": claim, "evidence": evidence},
            {"grounded": {"type": "noul", "instructions": GROUNDEDNESS_INSTRUCTIONS}},
        )
        score = self._noul(decisions, "grounded")
        grounded = score >= THRESHOLD
        return GroundednessDecision(grounded, score, self._trace("groundedness", "grounded" if grounded else "ungrounded", decisions))

    def _decide(self, state: dict, questions: dict) -> Decisions:
        return self._gateway.decide(self._model, state, questions)

    def _choice(self, decisions: Decisions, key: str, criteria: dict) -> str:
        label = decisions.answers.get(key, {}).get("choice")
        if label not in criteria:
            raise System1Error(f"{key}: Jev chose {label!r}, expected one of {list(criteria)}")
        return label

    def _noul(self, decisions: Decisions, key: str) -> float:
        try:
            return float(decisions.answers[key]["noul"])
        except (KeyError, TypeError, ValueError) as error:
            raise System1Error(f"{key}: Jev returned no noul answer") from error

    def _trace(self, decision: str, outcome: str, decisions: Decisions) -> System1TraceEntry:
        return System1TraceEntry(decision, outcome, decisions.latency_s, decisions.cost_usd, self.name)
