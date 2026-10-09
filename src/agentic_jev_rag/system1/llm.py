"""The LLM decider: the System 1 baseline prompted with strict structured output."""

import json

from agentic_jev_rag.gateway import ModelGateway
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

SYSTEM = "You make one fast, typed decision for a recruiting pipeline. Reply only with the JSON requested."


def _options(criteria: dict[str, str]) -> str:
    return "\n".join(f"- {label}: {description}" for label, description in criteria.items())


def _numbered(passages: list[str]) -> str:
    return "\n\n".join(f"[{n}] {text}" for n, text in enumerate(passages))


def _object(properties: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


class LlmDecider:
    name = "llm"

    def __init__(self, gateway: ModelGateway, model: str):
        self._gateway = gateway
        self._model = model

    def route(self, question: str) -> RouteDecision:
        answer, trace = self._ask(
            "route",
            f"{ROUTE_INSTRUCTIONS}\n\nCollections:\n{_options(ROUTE_CRITERIA)}\n\nQuestion: {question}",
            _object({"collection": {"type": "string", "enum": list(ROUTE_CRITERIA)}}, ["collection"]),
            lambda a: a["collection"],
        )
        return RouteDecision(answer["collection"], trace)

    def chunk_relevance(self, question: str, chunks: list[str]) -> RelevanceDecision:
        prompt = (
            f"{RELEVANCE_INSTRUCTIONS}\n\nQuestion: {question}\n\nPassages:\n{_numbered(chunks)}\n\n"
            f"Return `relevant` with exactly {len(chunks)} booleans, one per passage in order."
        )
        schema = _object({"relevant": {"type": "array", "items": {"type": "boolean"}}}, ["relevant"])
        answer, trace = self._ask("chunk_relevance", prompt, schema, lambda a: None)
        judged = [bool(v) for v in answer["relevant"]][: len(chunks)]
        keep = judged + [True] * (len(chunks) - len(judged))  # unjudged chunks are kept, not silently lost
        trace = _with_outcome(trace, f"kept {sum(keep)}/{len(chunks)}")
        return RelevanceDecision(keep, [float(k) for k in keep], trace)

    def input_guardrail(self, text: str) -> GuardrailDecision:
        answer, trace = self._ask(
            "input_guardrail",
            f"{GUARDRAIL_INSTRUCTIONS}\n\nVerdicts:\n{_options(GUARDRAIL_CRITERIA)}\n\nRequest: {text}",
            _object({"verdict": {"type": "string", "enum": list(GUARDRAIL_CRITERIA)}}, ["verdict"]),
            lambda a: a["verdict"],
        )
        return GuardrailDecision(answer["verdict"], trace)

    def groundedness(self, claim: str, evidence: list[str]) -> GroundednessDecision:
        answer, trace = self._ask(
            "groundedness",
            f"{GROUNDEDNESS_INSTRUCTIONS}\n\nStatement: {claim}\n\nEvidence:\n{_numbered(evidence)}",
            _object({"grounded": {"type": "boolean"}}, ["grounded"]),
            lambda a: "grounded" if a["grounded"] else "ungrounded",
        )
        return GroundednessDecision(bool(answer["grounded"]), float(answer["grounded"]), trace)

    def _ask(self, decision: str, prompt: str, schema: dict, outcome) -> tuple[dict, System1TraceEntry]:
        completion = self._gateway.complete(
            self._model, prompt, system=SYSTEM, response_schema=(decision, schema)
        )
        try:
            answer = json.loads(completion.text)
            label = outcome(answer)
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise System1Error(f"{decision}: invalid answer {completion.text!r}") from error
        trace = System1TraceEntry(decision, str(label), completion.latency_s, completion.cost_usd, self.name)
        return answer, trace


def _with_outcome(trace: System1TraceEntry, outcome: str) -> System1TraceEntry:
    return System1TraceEntry(trace.decision, outcome, trace.latency_s, trace.cost_usd, trace.implementation)
