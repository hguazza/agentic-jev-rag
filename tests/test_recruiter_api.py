"""Recruiter seam: the Recruiter API and LangGraph graph with the fake gateway and the in-process RAG API."""

import re

import pytest
from fastapi.testclient import TestClient

from agentic_jev_rag.gateway import ToolCall
from agentic_jev_rag.recruiter.api import create_recruiter_app
from tests.rag_scripts import by_agent, scripted_gateway

ADVANCE = {
    "decision": "advance",
    "scores": [{"competency": "Technical depth", "score": 5, "evidence_chunk_ids": ["C-001#1"], "rationale": "Kafka at scale."}],
    "summary": "Strong Kafka engineer.",
}
REJECT = {
    "decision": "reject",
    "scores": [{"competency": "Technical depth", "score": 2, "evidence_chunk_ids": [], "rationale": "Little backend work."}],
    "summary": "Mostly frontend.",
}
ASSESSMENT = {
    "scores": [{"competency": "System design", "score": 4, "evidence_chunk_ids": [], "rationale": "Partitioned topics by order id."}],
    "recommendation": "next_round",
    "summary": "Good interview.",
}


def supervisor_following_progress(model, messages, tools):
    status = messages[-1]["content"]
    number = lambda label: int(re.search(rf"{label}: (\d+)", status).group(1))
    if "Sourced: False" in status:
        return [ToolCall("s", "to_sourcer", {})]
    if number("Shortlisted") > number("Screened"):
        return [ToolCall("s", "to_screener", {})]
    if "interviews enabled: True" in status and number("Advanced by the Screener") > number(r"Interviewed"):
        return [ToolCall("s", "to_interviewer", {})]
    if "Compliance review done: False" in status:
        return [ToolCall("s", "to_compliance", {})]
    return [ToolCall("s", "finish", {})]


def sourcer_submitting(*candidate_ids):
    def respond(model, messages, tools):
        entries = [{"candidate_id": c, "name": c, "reason": "Job-related evidence."} for c in candidate_ids]
        return [ToolCall("src", "submit_shortlist", {"candidates": entries})]

    return respond


def screener(model, messages, tools):
    candidate = re.search(r"Candidate (C-\d+)", messages[1]["content"]).group(1)
    return [ToolCall("scr", "submit_verdict", ADVANCE if candidate == "C-001" else REJECT)]


def interview_question(prompt):
    asked = int(re.search(r"Questions asked: (\d+)", prompt).group(1))
    return {"done": asked >= 2, "question": f"Tell me about question {asked + 1}."}


def recruiter_gateway(*, shortlist=("C-001", "C-003"), verdict="allowed", grounded=True):
    return scripted_gateway(
        verdict=verdict,
        grounded=grounded,
        answer="A human recruiter must review the rejection first [1].",
        chat_responder=by_agent(
            Supervisor=supervisor_following_progress, Sourcer=sourcer_submitting(*shortlist), Screener=screener
        ),
        structured={"interview_question": interview_question, "interview_assessment": ASSESSMENT},
    )


@pytest.fixture
def client(recruiter_context):
    def make(gateway, **kwargs):
        return TestClient(create_recruiter_app(recruiter_context(gateway, **kwargs)))

    return make


def test_a_discriminatory_goal_is_refused_before_any_agent_runs(client):
    gateway = recruiter_gateway(verdict="discriminatory")

    report = client(gateway).post("/runs", json={"goal": "Backend engineers, men under 35 only", "job_id": "J-001"}).json()

    assert report["refused"] is True
    assert report["shortlist"] is None
    [finding] = report["findings"]
    assert finding["check"] == "input_guardrail" and finding["passed"] is False
    assert not [kind for kind, _, _ in gateway.calls if kind == "chat"]


def test_a_full_run_produces_a_recruitment_report(client):
    response = client(recruiter_gateway()).post("/runs", json={"goal": "Fill the senior backend role", "job_id": "J-001"})

    assert response.status_code == 200
    report = response.json()
    assert report["refused"] is False
    assert [c["candidate_id"] for c in report["shortlist"]["candidates"]] == ["C-001", "C-003"]
    assert {v["candidate_id"]: v["decision"] for v in report["verdicts"]} == {"C-001": "advance", "C-003": "reject"}
    [assessment] = report["assessments"]
    assert assessment["candidate_id"] == "C-001" and len(assessment["turns"]) == 2
    checks = {(f["check"], f["subject"]): f["passed"] for f in report["findings"]}
    assert checks[("input_guardrail", "goal")] is True
    assert checks[("consent", "C-001")] is True
    assert checks[("groundedness", "C-001")] is True
    assert ("policy", "rejections") in checks
    decisions = [t["decision"] for t in report["system_1_trace"]]
    assert decisions[0] == "input_guardrail" and "groundedness" in decisions
    assert {n["agent"] for n in report["notes"]} >= {"sourcer", "screener"}


def test_candidates_without_valid_consent_never_reach_the_shortlist(client):
    report = client(recruiter_gateway(shortlist=("C-004", "C-001"))).post(
        "/runs", json={"goal": "Fill the senior backend role", "job_id": "J-001"}
    ).json()

    assert [c["candidate_id"] for c in report["shortlist"]["candidates"]] == ["C-001"]
    assert {"check": "consent", "subject": "C-004", "passed": False} == {
        k: v for k, v in next(f for f in report["findings"] if f["subject"] == "C-004").items() if k != "detail"
    }
    assert "C-004" not in [v["candidate_id"] for v in report["verdicts"]]


def test_ungrounded_verdicts_are_flagged_in_the_report(client):
    report = client(recruiter_gateway(grounded=False)).post(
        "/runs", json={"goal": "Fill the senior backend role", "job_id": "J-001"}
    ).json()

    assert any(f["check"] == "groundedness" and not f["passed"] for f in report["findings"])


def test_runs_can_switch_to_the_jev_decider(client):
    report = client(recruiter_gateway()).post(
        "/runs", json={"goal": "Fill the senior backend role", "job_id": "J-001", "system_1": "jev", "simulate_interviews": False}
    ).json()

    assert {t["implementation"] for t in report["system_1_trace"]} == {"jev"}
    assert report["assessments"] == []


def test_a_run_for_an_unknown_job_is_404(client):
    assert client(recruiter_gateway()).post("/runs", json={"goal": "Fill it", "job_id": "J-404"}).status_code == 404


def test_a_live_interview_asks_questions_until_it_can_assess(client):
    api = client(recruiter_gateway())

    started = api.post("/interviews", json={"candidate_id": "C-001", "job_id": "J-001"}).json()
    assert started["question"] == "Tell me about question 1."
    second = api.post(f"/interviews/{started['interview_id']}/reply", json={"message": "I partitioned Kafka topics."}).json()
    assert second["question"] == "Tell me about question 2."
    done = api.post(f"/interviews/{started['interview_id']}/reply", json={"message": "I owned on-call."}).json()

    assert done["question"] is None
    assert done["assessment"]["recommendation"] == "next_round"
    assert [t["answer"] for t in done["assessment"]["turns"]] == ["I partitioned Kafka topics.", "I owned on-call."]
    again = api.post(f"/interviews/{started['interview_id']}/reply", json={"message": "Hello?"})
    assert again.status_code == 409


def test_interviews_need_a_known_id_and_a_candidate_with_valid_consent(client):
    api = client(recruiter_gateway())

    assert api.post("/interviews/nope/reply", json={"message": "hi"}).status_code == 404
    assert api.post("/interviews", json={"candidate_id": "C-004", "job_id": "J-001"}).status_code == 409
    assert api.get("/health").json() == {"status": "ok"}
