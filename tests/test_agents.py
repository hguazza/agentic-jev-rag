"""The five agents, each driven by a scripted fake gateway against the in-process RAG API."""

from agentic_jev_rag.gateway import ToolCall
from agentic_jev_rag.recruiter import compliance, interviewer, screener, sourcer, supervisor
from agentic_jev_rag.recruiter.schemas import CompetencyScore, InterviewTurn, ScreeningVerdict, Shortlist, ShortlistEntry
from tests.rag_scripts import by_agent, scripted_gateway, turns

VERDICT = {
    "decision": "advance",
    "scores": [
        {"competency": "Technical depth", "score": 5, "evidence_chunk_ids": ["C-001#1"], "rationale": "Built Kafka pipelines at 40k events/s."},
        {"competency": "Delivery", "score": 4, "evidence_chunk_ids": ["C-001#1"], "rationale": "Owned on-call."},
    ],
    "summary": "Strong Python and Kafka backend engineer.",
}


def test_sourcer_submits_a_shortlist_from_match_results(recruiter_context):
    gateway = scripted_gateway(chat_responder=by_agent(Sourcer=turns(
        [ToolCall("1", "match_candidates", {"job_id": "J-001", "top_k": 5})],
        [ToolCall("2", "submit_shortlist", {"candidates": [
            {"candidate_id": "C-001", "name": "Ana Ribeiro", "reason": "Kafka at scale"},
            {"candidate_id": "C-002", "name": "Tomás Verhoeven", "reason": "Debezium and Kafka"},
        ]})],
    )))

    shortlist, run = sourcer.source(recruiter_context(gateway), "J-001", "Fill the backend role")

    assert [c.candidate_id for c in shortlist.candidates] == ["C-001", "C-002"]
    assert shortlist.job_id == "J-001"
    tool_result = next(m for m in run.transcript if m["role"] == "tool")
    assert "C-001" in tool_result["content"]


def test_sourcer_that_fails_to_submit_returns_an_empty_shortlist_and_says_why(recruiter_context):
    gateway = scripted_gateway(chat_responder=by_agent(Sourcer=lambda m, msgs, t: "Hmm."))

    shortlist, run = sourcer.source(recruiter_context(gateway), "J-001", "Fill the backend role")

    assert shortlist.candidates == []
    assert run.stopped_reason == "no_tool_call"


def test_screener_returns_a_verdict_whose_scores_cite_evidence(recruiter_context):
    gateway = scripted_gateway(chat_responder=by_agent(Screener=turns(
        [ToolCall("1", "get_document", {"document_id": "C-001"})],
        [ToolCall("2", "submit_verdict", VERDICT)],
    )))

    verdict, _ = screener.screen(recruiter_context(gateway), "C-001", "J-001")

    assert verdict.decision == "advance"
    assert verdict.candidate_id == "C-001" and verdict.job_id == "J-001"
    assert verdict.scores[0].evidence_chunk_ids == ["C-001#1"]


def test_screener_scores_outside_one_to_five_are_sent_back(recruiter_context):
    bad = {**VERDICT, "scores": [{**VERDICT["scores"][0], "score": 7}]}
    gateway = scripted_gateway(chat_responder=by_agent(Screener=turns(
        [ToolCall("1", "submit_verdict", bad)],
        [ToolCall("2", "submit_verdict", VERDICT)],
    )))

    verdict, run = screener.screen(recruiter_context(gateway), "C-001", "J-001")

    assert verdict.scores[0].score == 5
    assert run.steps == 2


def test_interviewer_asks_until_the_limit_then_assesses(recruiter_context):
    asked = []

    def question(prompt):
        asked.append(prompt)
        return {"done": False, "question": f"Question {len(asked)}?"}

    gateway = scripted_gateway(structured={
        "interview_question": question,
        "interview_assessment": {
            "scores": [{"competency": "Technical depth", "score": 4, "evidence_chunk_ids": [], "rationale": "Explained partitioning."}],
            "recommendation": "next_round",
            "summary": "Solid answers.",
        },
    })
    ctx = recruiter_context(gateway)
    candidate, job = ctx.rag.document("C-001"), ctx.rag.document("J-001")
    turns_so_far: list[InterviewTurn] = []

    while (q := interviewer.next_question(ctx, candidate, job, turns_so_far, max_questions=3)) is not None:
        turns_so_far.append(InterviewTurn(question=q, answer=interviewer.simulated_answer(ctx, candidate, q)))
    assessment = interviewer.assess(ctx, candidate, job, turns_so_far)

    assert [t.question for t in turns_so_far] == ["Question 1?", "Question 2?", "Question 3?"]
    assert assessment.recommendation == "next_round"
    assert assessment.turns == turns_so_far
    assert "Interview Guide" in asked[0]  # the guide is fetched from the RAG, not hard-coded


def test_compliance_intake_refuses_a_discriminatory_goal(recruiter_context):
    finding, trace = compliance.intake(recruiter_context(scripted_gateway(verdict="discriminatory")), "Only men under 35")

    assert finding.check == "input_guardrail"
    assert finding.passed is False
    assert [t.decision for t in trace] == ["input_guardrail"]


def test_compliance_review_flags_expired_consent_and_ungrounded_verdicts(recruiter_context):
    ctx = recruiter_context(scripted_gateway(grounded=False))
    shortlist = Shortlist(job_id="J-001", candidates=[
        ShortlistEntry(candidate_id="C-001", name="Ana Ribeiro", reason="r"),
        ShortlistEntry(candidate_id="C-004", name="Mei Tanaka", reason="r"),
    ])
    verdict = ScreeningVerdict(candidate_id="C-001", job_id="J-001", **VERDICT)

    findings, trace = compliance.review(ctx, shortlist, [verdict])

    consent = {f.subject: f.passed for f in findings if f.check == "consent"}
    assert consent == {"C-001": True, "C-004": False}
    [grounded] = [f for f in findings if f.check == "groundedness"]
    assert grounded.subject == "C-001" and grounded.passed is False
    assert "groundedness" in [t.decision for t in trace]


def test_supervisor_corrects_a_handoff_whose_prerequisites_are_missing(recruiter_context):
    gateway = scripted_gateway(chat_responder=by_agent(Supervisor=lambda m, msgs, t: [ToolCall("1", "to_screener", {})]))

    decision = supervisor.supervise(recruiter_context(gateway), supervisor.Progress(goal="g", job_id="J-001"))

    assert decision.choice == "sourcer"
    assert decision.corrected_from == "screener"


def test_supervisor_follows_a_valid_handoff(recruiter_context):
    gateway = scripted_gateway(chat_responder=by_agent(Supervisor=lambda m, msgs, t: [ToolCall("1", "to_compliance", {})]))
    progress = supervisor.Progress(goal="g", job_id="J-001", sourced=True, shortlisted=2, screened=2, advanced=1, interviewed=1)

    decision = supervisor.supervise(recruiter_context(gateway), progress)

    assert decision.choice == "compliance"
    assert decision.corrected_from is None


def test_supervisor_cannot_finish_before_compliance_review(recruiter_context):
    gateway = scripted_gateway(chat_responder=by_agent(Supervisor=lambda m, msgs, t: [ToolCall("1", "finish", {})]))
    progress = supervisor.Progress(goal="g", job_id="J-001", sourced=True, shortlisted=1, screened=1, advanced=0)

    assert supervisor.supervise(recruiter_context(gateway), progress).choice == "compliance"


def test_agent_schemas_are_flat_and_strict_for_structured_output():
    from agentic_jev_rag.recruiter.schemas import json_schema

    schema = json_schema(ScreeningVerdict, exclude=("candidate_id", "job_id"))

    text = str(schema)
    assert "$ref" not in text and "$defs" not in text and "minimum" not in text and "default" not in text
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"decision", "scores", "summary"}
    item = schema["properties"]["scores"]["items"]
    assert item["additionalProperties"] is False
    assert set(item["required"]) == {"competency", "score", "evidence_chunk_ids", "rationale"}
