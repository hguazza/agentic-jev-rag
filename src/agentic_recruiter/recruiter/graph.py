"""The Recruiter as a LangGraph StateGraph: Compliance intake, then the Supervisor hands work to the agents."""

import time
from dataclasses import replace
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from agentic_recruiter.gateway import CostMeter
from agentic_recruiter.rag.models import TraceOut
from agentic_recruiter.recruiter import compliance, interviewer, screener, sourcer, supervisor
from agentic_recruiter.recruiter.agent_loop import AgentResult
from agentic_recruiter.recruiter.context import RecruiterContext
from agentic_recruiter.recruiter.report import RecruitmentReport
from agentic_recruiter.recruiter.schemas import (
    AgentRunNote,
    ComplianceFinding,
    InterviewAssessment,
    InterviewTurn,
    ScreeningVerdict,
    Shortlist,
)
from agentic_recruiter.system1 import System1TraceEntry

MAX_SUPERVISOR_STEPS = 10


class RecruiterState(TypedDict, total=False):
    goal: str
    job_id: str
    refused: bool
    sourced: bool
    shortlist: Shortlist | None
    screened_ids: list[str]
    verdicts: list[ScreeningVerdict]
    interviewed_ids: list[str]
    assessments: list[InterviewAssessment]
    findings: list[ComplianceFinding]
    trace: list[System1TraceEntry]
    notes: list[AgentRunNote]
    supervisor_steps: int
    compliance_done: bool
    next: str


def build_recruiter_graph(ctx: RecruiterContext, *, simulate_interviews: bool = True, max_interviews: int = 2):
    def intake(state: RecruiterState) -> dict:
        finding, trace = compliance.intake(ctx, state["goal"])
        return {"refused": not finding.passed, "findings": [finding], "trace": trace}

    def supervise(state: RecruiterState) -> dict:
        steps = state.get("supervisor_steps", 0)
        if steps >= MAX_SUPERVISOR_STEPS:
            note = AgentRunNote(agent="supervisor", subject="step limit reached", stopped_reason="step_limit", steps=steps, cost_usd=0.0)
            # The Compliance review always runs before the report, even when the Supervisor ran out of steps.
            next_step = "finish" if state.get("compliance_done") else "compliance"
            return {"next": next_step, "notes": state.get("notes", []) + [note]}
        decision = supervisor.supervise(ctx, _progress(state, simulate_interviews, max_interviews))
        notes = state.get("notes", [])
        if decision.corrected_from:
            notes = notes + [
                AgentRunNote(
                    agent="supervisor", subject=f"proposed {decision.corrected_from}, ran {decision.choice}",
                    stopped_reason="corrected", steps=1, cost_usd=decision.cost_usd,
                )
            ]
        return {"next": decision.choice, "supervisor_steps": steps + 1, "notes": notes}

    def source(state: RecruiterState) -> dict:
        shortlist, run = sourcer.source(ctx, state["job_id"], state["goal"])
        kept, findings = [], list(state.get("findings", []))
        unique = {entry.candidate_id: entry for entry in reversed(shortlist.candidates)}
        for entry in [e for e in shortlist.candidates if unique.get(e.candidate_id) is e]:
            consent = compliance.consent_finding(ctx, entry.candidate_id)
            if consent.passed:
                kept.append(entry)
            else:  # Candidates without valid Consent never reach the shortlist, whatever the Sourcer did
                findings.append(consent.model_copy(update={"detail": consent.detail + " Removed from the shortlist."}))
        return {
            "sourced": True,
            "shortlist": Shortlist(job_id=state["job_id"], candidates=kept),
            "findings": findings,
            "notes": state.get("notes", []) + [_note("sourcer", state["job_id"], run)],
        }

    def screen(state: RecruiterState) -> dict:
        screened = list(state.get("screened_ids", []))
        verdicts, notes = list(state.get("verdicts", [])), list(state.get("notes", []))
        for entry in state["shortlist"].candidates:
            if entry.candidate_id in screened:
                continue
            verdict, run = screener.screen(ctx, entry.candidate_id, state["job_id"])
            screened.append(entry.candidate_id)
            notes.append(_note("screener", entry.candidate_id, run))
            if verdict:
                verdicts.append(verdict)
        return {"screened_ids": screened, "verdicts": verdicts, "notes": notes}

    def interview(state: RecruiterState) -> dict:
        done = list(state.get("interviewed_ids", []))
        assessments = list(state.get("assessments", []))
        job = ctx.rag.document(state["job_id"])
        for candidate_id in _advanced(state, max_interviews):
            if candidate_id in done:
                continue
            candidate = ctx.rag.document(candidate_id)
            briefing = interviewer.brief(ctx, candidate, job)
            turns: list[InterviewTurn] = []
            while (question := interviewer.next_question(ctx, briefing.text, turns)) is not None:
                answer = interviewer.simulated_answer(ctx, candidate, question.text)
                turns.append(InterviewTurn(question=question.text, answer=answer, competency=question.competency))
            assessments.append(interviewer.assess(ctx, briefing.text, candidate["id"], job["id"], turns))
            done.append(candidate_id)
        return {"interviewed_ids": done, "assessments": assessments}

    def review(state: RecruiterState) -> dict:
        findings, trace = compliance.review(ctx, state.get("shortlist") or Shortlist(job_id=state["job_id"]), state.get("verdicts", []))
        return {
            "findings": state.get("findings", []) + findings,
            "trace": state.get("trace", []) + trace,
            "compliance_done": True,
        }

    graph = StateGraph(RecruiterState)
    graph.add_node("compliance_intake", intake)
    graph.add_node("supervisor", supervise)
    graph.add_node("sourcer", source)
    graph.add_node("screener", screen)
    graph.add_node("interviewer", interview)
    graph.add_node("compliance_review", review)
    graph.add_edge(START, "compliance_intake")
    graph.add_conditional_edges("compliance_intake", lambda s: END if s["refused"] else "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        lambda s: s["next"],
        {"sourcer": "sourcer", "screener": "screener", "interviewer": "interviewer", "compliance": "compliance_review", "finish": END},
    )
    for node in ("sourcer", "screener", "interviewer", "compliance_review"):
        graph.add_edge(node, "supervisor")
    return graph.compile()


def run_recruitment(
    ctx: RecruiterContext, goal: str, job_id: str, *, simulate_interviews: bool = True, max_interviews: int = 2
) -> RecruitmentReport:
    started = time.perf_counter()
    meter = CostMeter(ctx.gateway)
    metered = replace(ctx, gateway=meter)
    graph = build_recruiter_graph(metered, simulate_interviews=simulate_interviews, max_interviews=max_interviews)
    state = graph.invoke({"goal": goal, "job_id": job_id}, {"recursion_limit": 4 * MAX_SUPERVISOR_STEPS})
    return RecruitmentReport(
        goal=goal,
        job_id=job_id,
        system_1=ctx.system_1,
        refused=state.get("refused", False),
        shortlist=state.get("shortlist"),
        verdicts=state.get("verdicts", []),
        assessments=state.get("assessments", []),
        findings=state.get("findings", []),
        system_1_trace=[TraceOut(**vars(t)) for t in state.get("trace", [])],
        notes=state.get("notes", []),
        cost_usd=meter.cost_usd,
        latency_s=time.perf_counter() - started,
    )


def _advanced(state: RecruiterState, limit: int) -> list[str]:
    return [v.candidate_id for v in state.get("verdicts", []) if v.decision == "advance"][:limit]


def _progress(state: RecruiterState, interviews: bool, limit: int) -> supervisor.Progress:
    shortlist = state.get("shortlist")
    return supervisor.Progress(
        goal=state["goal"],
        job_id=state["job_id"],
        sourced=state.get("sourced", False),
        shortlisted=len(shortlist.candidates) if shortlist else 0,
        screened=len(state.get("screened_ids", [])),
        advanced=len(_advanced(state, limit)),
        interviewed=len(state.get("interviewed_ids", [])),
        compliance_done=state.get("compliance_done", False),
        interviews_enabled=interviews,
    )


def _note(agent: str, subject: str, run: AgentResult) -> AgentRunNote:
    return AgentRunNote(agent=agent, subject=subject, stopped_reason=run.stopped_reason, steps=run.steps, cost_usd=run.cost_usd)
