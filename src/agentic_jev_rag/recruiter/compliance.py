"""Compliance: the Input guardrail on the goal, then Consent, grounded verdicts and policy on the results.

An integrity check on interview assessments keeps delivery timing out of scores (ADR 0013).
"""

import re
from datetime import date

from agentic_jev_rag.recruiter.context import RecruiterContext
from agentic_jev_rag.recruiter.rag_client import RagError
from agentic_jev_rag.recruiter.schemas import ComplianceFinding, InterviewAssessment, ScreeningVerdict, Shortlist
from agentic_jev_rag.system1 import System1TraceEntry

# Phrases about how the Candidate delivered an answer; plain technical words (latency, pause) must not match.
TIMING_TERMS = re.compile(
    r"""\b(?:
        long\s+pauses?
        | pause[sd]?\s+before\s+(?:answering|responding|the\s+answer)
        | filler\s+words?
        | fillers
        | speaking\s+(?:rate|speed)
        | words\s+per\s+minute
        | response\s+latency
        | (?:hesitated|hesitation)\s+before\s+(?:answering|responding)
        | took\s+(?:a\s+)?long(?:er)?\s+to\s+(?:answer|respond)
    )\b""",
    re.I | re.X,
)
REJECTION_QUESTION = "What must happen before a candidate rejected with the help of an AI system is told?"


def intake(ctx: RecruiterContext, goal: str) -> tuple[ComplianceFinding, list[System1TraceEntry]]:
    decision = ctx.decider.input_guardrail(goal)
    finding = ComplianceFinding(
        check="input_guardrail",
        subject="goal",
        passed=decision.allowed,
        detail=f"Input guardrail verdict: {decision.verdict}"
        + ("" if decision.allowed else ". Refused under the Fair Hiring Policy; restate the goal with job-related criteria."),
    )
    return finding, [decision.trace]


def review(
    ctx: RecruiterContext, shortlist: Shortlist, verdicts: list[ScreeningVerdict]
) -> tuple[list[ComplianceFinding], list[System1TraceEntry]]:
    findings = [consent_finding(ctx, entry.candidate_id) for entry in shortlist.candidates]
    trace: list[System1TraceEntry] = []
    for verdict in verdicts:
        finding, entry = _groundedness(ctx, verdict)
        findings.append(finding)
        trace.append(entry)
    if any(v.decision == "reject" for v in verdicts):
        answer = ctx.rag.ask(REJECTION_QUESTION, collections=["policies"], system_1=ctx.system_1)
        trace += [System1TraceEntry(**t) for t in answer["system_1_trace"]]
        findings.append(
            ComplianceFinding(
                check="policy",
                subject="rejections",
                passed=not answer["refused"],
                detail=f"Before telling rejected candidates: {answer['answer']}",
            )
        )
    return findings, trace


def consent_finding(ctx: RecruiterContext, candidate_id: str) -> ComplianceFinding:
    try:
        until = ctx.rag.document(candidate_id)["metadata"].get("consent_until")
    except RagError:
        return ComplianceFinding(check="consent", subject=candidate_id, passed=False, detail="Unknown candidate.")
    valid = bool(until) and date.fromisoformat(until) >= ctx.today()
    detail = f"Consent valid until {until}." if valid else f"Consent expired or missing ({until}); must not be shortlisted."
    return ComplianceFinding(check="consent", subject=candidate_id, passed=valid, detail=detail)


def _groundedness(ctx: RecruiterContext, verdict: ScreeningVerdict) -> tuple[ComplianceFinding, System1TraceEntry]:
    chunks = ctx.rag.search(
        verdict.summary, collections=["candidates", "interviews"], filters={"candidate_id": verdict.candidate_id}, top_k=20
    )["chunks"]
    by_id = {c["chunk_id"]: c["text"] for c in chunks}
    cited = [by_id[i] for s in verdict.scores for i in s.evidence_chunk_ids if i in by_id]
    evidence = list(dict.fromkeys(cited)) or list(by_id.values())
    claim = verdict.summary + "\n" + "\n".join(f"{s.competency} ({s.score}/5): {s.rationale}" for s in verdict.scores)
    decision = ctx.decider.groundedness(claim, evidence)
    detail = (
        "Screening verdict is supported by its evidence."
        if decision.grounded
        else "Screening verdict is not supported by its evidence; a human must review it before it is acted on."
    )
    return ComplianceFinding(check="groundedness", subject=verdict.candidate_id, passed=decision.grounded, detail=detail), decision.trace


def integrity_finding(assessment: InterviewAssessment) -> ComplianceFinding:
    """Integrity signals are advisory (ADR 0013): no score rationale or summary may cite delivery timing."""
    texts = [assessment.summary, *(s.rationale for s in assessment.scores)]
    cited = sorted({m.group(0).lower() for text in texts for m in TIMING_TERMS.finditer(text)})
    if not cited:
        return ComplianceFinding(
            check="integrity", subject=assessment.candidate_id, passed=True,
            detail="No delivery timing was cited in the scores or the recommendation.",
        )
    return ComplianceFinding(
        check="integrity", subject=assessment.candidate_id, passed=False,
        detail=f"The assessment cites delivery timing ({', '.join(cited)}). Integrity signals are advisory only and must "
        "not affect scores; a human must review it.",
    )
