"""Recruiter seam: the Recruiter API and LangGraph graph with the fake gateway and the in-process RAG API."""

import re

import pytest
from fastapi.testclient import TestClient

from agentic_jev_rag.gateway import ToolCall
from agentic_jev_rag.recruiter.api import create_recruiter_app
from tests.interview_scripts import interview_gateway, interview_question, prompts_for
from tests.rag_scripts import TURN_OUTCOMES, by_agent, scripted_gateway

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


def run(api, body):
    """Starts a run and reads its result; TestClient finishes background tasks before returning the response."""
    accepted = api.post("/runs", json=body)
    assert accepted.status_code == 202, accepted.text
    assert accepted.json()["status"] == "running"
    status = api.get(f"/runs/{accepted.json()['run_id']}").json()
    assert status["status"] == "complete", status
    return status["report"]


@pytest.fixture
def client(recruiter_context):
    def make(gateway, **kwargs):
        return TestClient(create_recruiter_app(recruiter_context(gateway, **kwargs)))

    return make


def test_a_discriminatory_goal_is_refused_before_any_agent_runs(client):
    gateway = recruiter_gateway(verdict="discriminatory")

    report = run(client(gateway), {"goal": "Backend engineers, men under 35 only", "job_id": "J-001"})

    assert report["refused"] is True
    assert report["shortlist"] is None
    [finding] = report["findings"]
    assert finding["check"] == "input_guardrail" and finding["passed"] is False
    assert not [kind for kind, _, _ in gateway.calls if kind == "chat"]


def test_a_full_run_produces_a_recruitment_report(client):
    report = run(client(recruiter_gateway()), {"goal": "Fill the senior backend role", "job_id": "J-001"})

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
    report = run(client(recruiter_gateway(shortlist=("C-004", "C-001"))), {"goal": "Fill the senior backend role", "job_id": "J-001"})

    assert [c["candidate_id"] for c in report["shortlist"]["candidates"]] == ["C-001"]
    assert {"check": "consent", "subject": "C-004", "passed": False} == {
        k: v for k, v in next(f for f in report["findings"] if f["subject"] == "C-004").items() if k != "detail"
    }
    assert "C-004" not in [v["candidate_id"] for v in report["verdicts"]]


def test_ungrounded_verdicts_are_flagged_in_the_report(client):
    report = run(client(recruiter_gateway(grounded=False)), {"goal": "Fill the senior backend role", "job_id": "J-001"})

    assert any(f["check"] == "groundedness" and not f["passed"] for f in report["findings"])


def test_runs_can_switch_to_the_jev_decider(client):
    report = run(
        client(recruiter_gateway()),
        {"goal": "Fill the senior backend role", "job_id": "J-001", "system_1": "jev", "simulate_interviews": False},
    )

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


def test_a_duplicated_candidate_in_the_shortlist_still_gets_a_compliance_review(client):
    report = run(client(recruiter_gateway(shortlist=("C-001", "C-001", "C-003"))), {"goal": "Fill the senior backend role", "job_id": "J-001"})

    assert [c["candidate_id"] for c in report["shortlist"]["candidates"]] == ["C-001", "C-003"]
    assert {f["check"] for f in report["findings"]} >= {"consent", "groundedness"}


def test_an_out_of_range_interview_assessment_is_retried_then_reported_as_a_502(client):
    bad = {**ASSESSMENT, "scores": [{**ASSESSMENT["scores"][0], "score": 7}]}
    calls = {"n": 0}

    def flaky(prompt):
        calls["n"] += 1
        return bad if calls["n"] == 1 else ASSESSMENT

    api = client(scripted_gateway(structured={"interview_question": interview_question, "interview_assessment": flaky}))
    started = api.post("/interviews", json={"candidate_id": "C-001", "job_id": "J-001"}).json()
    api.post(f"/interviews/{started['interview_id']}/reply", json={"message": "Kafka."})
    done = api.post(f"/interviews/{started['interview_id']}/reply", json={"message": "On-call."})
    assert done.status_code == 200 and done.json()["assessment"]["scores"][0]["score"] == 4

    always_bad = client(scripted_gateway(structured={"interview_question": interview_question, "interview_assessment": bad}))
    started = always_bad.post("/interviews", json={"candidate_id": "C-001", "job_id": "J-001"}).json()
    always_bad.post(f"/interviews/{started['interview_id']}/reply", json={"message": "Kafka."})
    failed = always_bad.post(f"/interviews/{started['interview_id']}/reply", json={"message": "On-call."})
    assert failed.status_code == 502


def test_a_reply_retried_after_a_502_does_not_record_the_retry_message_as_another_answer(client):
    calls = {"n": 0}

    def flaky(prompt):
        calls["n"] += 1
        if calls["n"] in (2, 3):  # the question after the first answer is invalid twice: the first reply is a 502
            return {"done": False, "question": ""}
        return interview_question(prompt)

    api = client(scripted_gateway(structured={"interview_question": flaky, "interview_assessment": ASSESSMENT}))
    started = api.post("/interviews", json={"candidate_id": "C-001", "job_id": "J-001"}).json()
    reply = f"/interviews/{started['interview_id']}/reply"

    failed = api.post(reply, json={"message": "I partitioned Kafka topics."})
    retried = api.post(reply, json={"message": "RETRY MESSAGE, must not be recorded"})
    done = api.post(reply, json={"message": "I owned on-call."})

    assert failed.status_code == 502
    assert retried.status_code == 200 and retried.json()["question"] == "Tell me about question 2."
    assert [t["answer"] for t in done.json()["assessment"]["turns"]] == ["I partitioned Kafka topics.", "I owned on-call."]


def test_a_failed_turn_check_is_a_502_and_replying_again_continues_without_duplicating(client):
    failures = iter([None])  # the first Turn check returns invalid output

    api = client(interview_gateway(turn=lambda answer: next(failures, TURN_OUTCOMES["substantive"])))
    started = api.post("/interviews", json={"candidate_id": "C-001", "job_id": "J-001"}).json()
    url = f"/interviews/{started['interview_id']}/reply"

    assert api.post(url, json={"message": "I partitioned Kafka topics."}).status_code == 502
    retried = api.post(url, json={"message": "I partitioned Kafka topics."}).json()
    done = api.post(url, json={"message": "I owned on-call."}).json()

    assert retried["question"] == "Tell me about question 2." and retried["question_kind"] == "question"
    assert [t["answer"] for t in done["assessment"]["turns"]] == ["I partitioned Kafka topics.", "I owned on-call."]


def test_reply_timing_is_kept_per_turn_reported_as_advisory_and_never_shown_to_the_assessor(client):
    gateway = interview_gateway()
    api = client(gateway)
    started = api.post("/interviews", json={"candidate_id": "C-001", "job_id": "J-001"}).json()
    url = f"/interviews/{started['interview_id']}/reply"
    timing = {"response_latency_s": 12.5, "answer_duration_s": 40.0, "words_per_minute": 195.0, "filler_count": 0}

    api.post(url, json={"message": "I partitioned Kafka topics.", "timing": timing})
    done = api.post(url, json={"message": "I owned on-call."}).json()

    assessment = done["assessment"]
    assert assessment["turns"][0]["timing"] == timing and assessment["turns"][1]["timing"] is None
    [observed] = assessment["integrity_signals"]["turns"]
    assert observed["turn"] == 1 and observed["observation"]
    assert "never affect" in assessment["integrity_signals"]["note"]
    assert [f["check"] for f in done["findings"]] == ["integrity"] and done["findings"][0]["passed"]
    prompt = prompts_for(gateway, "interview_assessment")[-1]
    assert "12.5" not in prompt and "response_latency" not in prompt and "filler" not in prompt.lower()


def test_an_interview_continues_on_another_instance_when_checkpoints_are_in_dynamodb(
    recruiter_context, dynamodb_tables, caplog
):
    from dataclasses import replace

    gateway = recruiter_gateway()

    def instance():
        ctx = recruiter_context(gateway)
        ctx.config = replace(ctx.config, **dynamodb_tables)
        return TestClient(create_recruiter_app(ctx))

    first, second = instance(), instance()

    started = first.post("/interviews", json={"candidate_id": "C-001", "job_id": "J-001"}).json()
    next_question = second.post(f"/interviews/{started['interview_id']}/reply", json={"message": "I partitioned Kafka topics."}).json()
    assert next_question["question"] == "Tell me about question 2."
    done = first.post(f"/interviews/{started['interview_id']}/reply", json={"message": "I owned on-call."}).json()

    assert done["assessment"]["recommendation"] == "next_round"
    assert [t["answer"] for t in done["assessment"]["turns"]] == ["I partitioned Kafka topics.", "I owned on-call."]
    # LangGraph will refuse checkpointed types it was not told about; every type the interview stores is allowed
    assert not [r for r in caplog.records if "unregistered type" in r.getMessage()]


def test_a_run_whose_models_fail_is_reported_as_failed(client):
    from agentic_jev_rag.openrouter import OpenRouterError

    def down(model, messages, tools):
        raise OpenRouterError("OpenRouter is down")

    gateway = scripted_gateway(verdict="allowed", chat_responder=by_agent(Supervisor=down))
    api = client(gateway)

    run_id = api.post("/runs", json={"goal": "Fill the senior backend role", "job_id": "J-001"}).json()["run_id"]
    status = api.get(f"/runs/{run_id}").json()

    assert status["status"] == "failed"
    assert "OpenRouter is down" in status["error"] and status["report"] is None


def test_a_run_left_running_too_long_is_reported_as_interrupted(recruiter_context):
    from agentic_jev_rag.recruiter.persistence import MemoryRuns, RunRecord

    runs = MemoryRuns()
    runs.put(RunRecord(run_id="old", status="running", started_at=0.0))
    api = TestClient(create_recruiter_app(recruiter_context(recruiter_gateway()), runs=runs))

    status = api.get("/runs/old").json()

    assert status["status"] == "failed" and "interrupted" in status["error"]


def test_an_unknown_run_is_404(client):
    assert client(recruiter_gateway()).get("/runs/nope").status_code == 404


@pytest.fixture
def keyed(recruiter_context):
    from dataclasses import replace

    ctx = recruiter_context(recruiter_gateway())
    ctx.config = replace(ctx.config, api_key="s3cret")
    return TestClient(create_recruiter_app(ctx))


def test_with_an_api_key_set_every_route_but_health_and_docs_needs_it(keyed):
    start = {"candidate_id": "C-001", "job_id": "J-001"}

    assert keyed.post("/interviews", json=start).status_code == 401
    assert keyed.post("/interviews", json=start, headers={"X-API-Key": "wrong"}).status_code == 401
    assert keyed.post("/runs", json={"goal": "Fill it", "job_id": "J-001"}).status_code == 401
    assert keyed.get("/runs/any").status_code == 401
    assert keyed.post("/interviews/any/reply", json={"message": "hi"}).status_code == 401
    assert keyed.get("/health").status_code == 200
    assert keyed.get("/docs").status_code == 200 and keyed.get("/openapi.json").status_code == 200

    started = keyed.post("/interviews", json=start, headers={"X-API-Key": "s3cret"})
    assert started.status_code == 200 and started.json()["question"]


def test_without_an_api_key_set_auth_is_off(client):
    assert client(recruiter_gateway()).post("/interviews", json={"candidate_id": "C-001", "job_id": "J-001"}).status_code == 200
