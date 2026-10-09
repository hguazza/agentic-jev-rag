"""System 1 deciders: the LLM decider and the Jev decider make the same four typed decisions."""

import json

import pytest

from agentic_jev_rag.gateway import Completion, Decisions, FakeModelGateway
from agentic_jev_rag.system1 import make_deciders
from agentic_jev_rag.system1.prompts import GUARDRAIL_CRITERIA, ROUTE_CRITERIA


def llm_answering(payload_by_schema: dict):
    """A fake whose structured completions answer by schema name."""
    return FakeModelGateway(
        complete_responder=lambda model, prompt, system, schema: json.dumps(payload_by_schema[schema[0]])
    )


def jev_answering(answer_for):
    """A fake whose Jev decisions come from `answer_for(question_id, question)`."""
    return FakeModelGateway(
        decide_responder=lambda state, questions: {qid: answer_for(qid, q) for qid, q in questions.items()}
    )


def choice(label):
    return {"type": "choice", "choice": label, "probabilities": {label: 0.9}, "confidence": 0.9}


def deciders_for(gateway, config):
    return make_deciders(config, gateway)


def test_llm_guardrail_refuses_when_the_model_says_discriminatory(config):
    decider = deciders_for(llm_answering({"input_guardrail": {"verdict": "discriminatory"}}), config)["llm"]

    decision = decider.input_guardrail("Only men under 35.")

    assert decision.verdict == "discriminatory"
    assert not decision.allowed
    assert decision.trace.decision == "input_guardrail"
    assert decision.trace.outcome == "discriminatory"
    assert decision.trace.implementation == "llm"


def test_jev_guardrail_asks_a_choice_over_the_same_verdicts(config):
    gateway = jev_answering(lambda qid, q: choice("discriminatory"))
    decider = deciders_for(gateway, config)["jev"]

    decision = decider.input_guardrail("Digital natives only, please.")

    assert not decision.allowed
    assert decision.trace.implementation == "jev"
    [(kind, model, payload)] = gateway.calls
    assert kind == "decide" and model == config.models.jev_model
    [question] = payload["questions"].values()
    assert question["type"] == "choice"
    assert question["criteria"] == GUARDRAIL_CRITERIA


def test_llm_route_uses_a_strict_schema_enumerating_the_collections(config):
    gateway = llm_answering({"route": {"collection": "policies"}})

    decision = deciders_for(gateway, config)["llm"].route("How long do we keep interview notes?")

    assert decision.collection == "policies"
    [(_, model, payload)] = gateway.calls
    assert model == config.models.system_1_baseline
    name, schema = payload["schema"]
    assert schema["properties"]["collection"]["enum"] == list(ROUTE_CRITERIA)


def test_jev_route_returns_the_chosen_collection(config):
    decision = deciders_for(jev_answering(lambda qid, q: choice("jobs")), config)["jev"].route("Open roles?")

    assert decision.collection == "jobs"
    assert decision.trace.outcome == "jobs"


def test_jev_chunk_relevance_asks_one_noul_per_chunk_and_keeps_those_at_or_above_half(config):
    scores = {"chunk_0": 0.7, "chunk_1": 0.2, "chunk_2": 0.5}
    gateway = jev_answering(lambda qid, q: {"type": "noul", "noul": scores[qid]})

    decision = deciders_for(gateway, config)["jev"].chunk_relevance("Kafka?", ["a", "b", "c"])

    assert decision.keep == [True, False, True]
    assert decision.scores == [0.7, 0.2, 0.5]
    assert decision.trace.outcome == "kept 2/3"
    [(_, _, payload)] = gateway.calls
    assert sorted(payload["questions"]) == ["chunk_0", "chunk_1", "chunk_2"]
    assert all(q["type"] == "noul" for q in payload["questions"].values())


def test_llm_chunk_relevance_reads_one_boolean_per_chunk(config):
    gateway = llm_answering({"chunk_relevance": {"relevant": [False, True]}})

    decision = deciders_for(gateway, config)["llm"].chunk_relevance("Kafka?", ["a", "b"])

    assert decision.keep == [False, True]


def test_llm_chunk_relevance_keeps_chunks_the_model_forgot_to_judge(config):
    gateway = llm_answering({"chunk_relevance": {"relevant": [False]}})

    decision = deciders_for(gateway, config)["llm"].chunk_relevance("Kafka?", ["a", "b", "c"])

    assert decision.keep == [False, True, True]


@pytest.mark.parametrize(
    "implementation, gateway_factory",
    [
        ("llm", lambda: llm_answering({"groundedness": {"grounded": False}})),
        ("jev", lambda: jev_answering(lambda qid, q: {"type": "noul", "noul": 0.1})),
    ],
)
def test_groundedness_flags_unsupported_claims(config, implementation, gateway_factory):
    decider = deciders_for(gateway_factory(), config)[implementation]

    decision = decider.groundedness("She led 12 engineers.", ["Built Kafka pipelines."])

    assert decision.grounded is False
    assert decision.trace.outcome == "ungrounded"


def test_trace_carries_the_gateways_latency_and_cost(config):
    class Timed(FakeModelGateway):
        def decide(self, model, state, questions):
            answers = super().decide(model, state, questions).answers
            return Decisions(answers, latency_s=0.21, cost_usd=0.00002)

        def complete(self, model, prompt, **kwargs):
            return Completion(json.dumps({"verdict": "allowed"}), latency_s=1.4, cost_usd=0.0003)

    deciders = deciders_for(Timed(), config)

    jev = deciders["jev"].input_guardrail("Backend engineers in Lisbon").trace
    llm = deciders["llm"].input_guardrail("Backend engineers in Lisbon").trace

    assert (jev.latency_s, jev.cost_usd) == (0.21, 0.00002)
    assert (llm.latency_s, llm.cost_usd) == (1.4, 0.0003)
