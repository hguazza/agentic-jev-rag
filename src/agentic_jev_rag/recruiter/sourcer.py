"""The Sourcer: finds and ranks Candidates for a Job with Match and Search."""

from agentic_jev_rag.recruiter import prompts
from agentic_jev_rag.recruiter.agent_loop import AgentResult, Tool, run_agent
from agentic_jev_rag.recruiter.context import RecruiterContext
from agentic_jev_rag.recruiter.schemas import Shortlist, json_schema


def source(ctx: RecruiterContext, job_id: str, goal: str) -> tuple[Shortlist, AgentResult]:
    submit = Tool(
        "submit_shortlist",
        "Submit the shortlist: at most 8 Candidates, best first, each with a job-related reason.",
        json_schema(Shortlist, exclude=("job_id",)),
        lambda candidates: Shortlist(job_id=job_id, candidates=candidates).model_dump(),
    )
    run = run_agent(
        ctx.gateway,
        ctx.config.models.system_2_model,
        prompts.SOURCER,
        f"Goal: {goal}\nJob id: {job_id}",
        ctx.tools("match_candidates", "search", "get_document"),
        submit,
    )
    shortlist = Shortlist.model_validate(run.submitted) if run.submitted else Shortlist(job_id=job_id)
    return shortlist, run
