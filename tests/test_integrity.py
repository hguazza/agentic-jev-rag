"""Integrity signals are advisory observations for a human; the Compliance check keeps them out of the scores."""

from agentic_jev_rag.recruiter import compliance
from agentic_jev_rag.recruiter.integrity import integrity_signals
from agentic_jev_rag.recruiter.schemas import INTEGRITY_NOTE, CompetencyScore, InterviewAssessment, InterviewTurn, TurnTiming


def turn(**timing):
    return InterviewTurn(question="Q?", answer="A.", timing=TurnTiming(**timing) if timing else None)


def test_only_a_long_pause_before_a_fast_filler_free_answer_gets_an_observation():
    signals = integrity_signals([
        turn(response_latency_s=9.5, answer_duration_s=30, words_per_minute=190, filler_count=0),
        turn(response_latency_s=9.5, answer_duration_s=30, words_per_minute=120, filler_count=3),
        turn(response_latency_s=1.2, answer_duration_s=20, words_per_minute=200, filler_count=0),
        turn(),
    ])

    assert signals.note == INTEGRITY_NOTE
    assert [t.turn for t in signals.turns] == [1, 2, 3]  # turns without timing are left out
    assert signals.turns[0].observation and "not evidence" in signals.turns[0].observation
    assert signals.turns[1].observation is None and signals.turns[2].observation is None


def assessment(rationale, summary="Solid answers."):
    return InterviewAssessment(
        candidate_id="C-001", job_id="J-001", turns=[], recommendation="next_round", summary=summary,
        scores=[CompetencyScore(competency="Delivery", score=3, rationale=rationale)],
    )


def test_the_integrity_check_passes_when_no_timing_is_cited():
    finding = compliance.integrity_finding(assessment("Shipped the ERP cutover with a dry run."))

    assert finding.check == "integrity" and finding.passed


def test_the_integrity_check_fails_when_a_score_cites_delivery_timing():
    finding = compliance.integrity_finding(assessment("Long pauses and no filler words suggest a scripted answer."))

    assert not finding.passed and "pauses" in finding.detail and "filler words" in finding.detail


def test_the_integrity_check_ignores_technical_words_that_are_not_about_delivery():
    finding = compliance.integrity_finding(
        assessment("Reduced p99 latency from 800 ms to 120 ms and paused the migration to fix response time regressions.")
    )

    assert finding.passed
