"""The Screener: assesses one Candidate against a Job's Scorecard and gives a Screening verdict."""

from agentic_jev_rag.recruiter import prompts
from agentic_jev_rag.recruiter.agent_loop import AgentResult, Tool, run_agent
from agentic_jev_rag.recruiter.context import RecruiterContext
from agentic_jev_rag.recruiter.schemas import ScreeningVerdict, json_schema


def screen(ctx: RecruiterContext, candidate_id: str, job_id: str) -> tuple[ScreeningVerdict | None, AgentResult]:
    submit = Tool(
        "submit_verdict",
        "Submit the Screening verdict: a 1-5 score per Scorecard competency with cited chunk ids, and the decision.",
        json_schema(ScreeningVerdict, exclude=("candidate_id", "job_id")),
        lambda **verdict: ScreeningVerdict(candidate_id=candidate_id, job_id=job_id, **verdict).model_dump(),
    )
    run = run_agent(
        ctx.gateway,
        ctx.config.models.system_2_model,
        prompts.SCREENER,
        f"Screen Candidate {candidate_id} for Job {job_id}.",
        ctx.tools("get_document", "search", "ask"),
        submit,
    )
    verdict = ScreeningVerdict.model_validate(run.submitted) if run.submitted else None
    return verdict, run
