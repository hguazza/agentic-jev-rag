"""Request and response models of the Recruiter API (agent outputs live in `schemas.py`)."""

from typing import Literal

from pydantic import BaseModel, Field

from agentic_jev_rag.recruiter.report import RecruitmentReport
from agentic_jev_rag.recruiter.schemas import ComplianceFinding, InterviewAssessment, TurnTiming


class RunRequest(BaseModel):
    """A hiring goal for the Recruiter to carry out for one Job."""

    goal: str = Field(
        min_length=1,
        description="What to achieve, in plain English. It goes through the Input guardrail first: goals with "
        "discriminatory criteria are refused.",
        examples=["Fill the Senior Linux Administrator role"],
    )
    job_id: str = Field(description="The Job to recruit for. Unknown ids return 404.", examples=["J-037"])
    system_1: Literal["llm", "jev"] | None = Field(
        default=None,
        description="Which System 1 decider runs the Input guardrail and the Groundedness checks for this run, "
        "including inside RAG Ask calls. Omit to use the service default.",
    )
    simulate_interviews: bool = Field(
        default=True,
        description="Interview advanced Candidates with a Simulated candidate (an LLM playing the Candidate from "
        "their CV). Set false to stop after screening.",
    )
    max_interviews: int = Field(default=2, ge=0, le=8, description="At most this many advanced Candidates are interviewed.")


class InterviewStart(BaseModel):
    """Start a live interview with a real person replying."""

    candidate_id: str = Field(
        description="The Candidate to interview. They must exist (404) and have valid Consent (409).", examples=["C-0040"]
    )
    job_id: str = Field(description="The Job the interview is for. Unknown ids return 404.", examples=["J-037"])


class InterviewReply(BaseModel):
    """The Candidate's answer to the current question."""

    message: str = Field(min_length=1, description="The Candidate's answer, in plain text.")
    timing: TurnTiming | None = Field(
        default=None, description="Delivery timing measured by a voice client. Advisory only: reported for human review, never scored."
    )


class InterviewState(BaseModel):
    """Where an interview stands: the next question, or the final assessment."""

    interview_id: str = Field(description="Id to reply to; keep it for the whole interview.", examples=["47898414c2a94bf3a1d0f5e2b6c7d8e9"])
    question: str | None = Field(description="The next question for the Candidate; null when the interview is over.")
    question_kind: Literal["question", "probe", "reprompt", "redirect"] | None = Field(
        default=None,
        description="How to put the question: a new `question`, a `probe` (follow-up on a vague answer), a `reprompt` "
        "(ask again: the reply was not understood) or a `redirect` (the reply tried to instruct the AI). Null when the "
        "interview is over.",
    )
    assessment: InterviewAssessment | None = Field(description="The Interviewer's assessment; set once the interview is over.")
    findings: list[ComplianceFinding] = Field(
        default_factory=list, description="Compliance findings on the assessment, once the interview is over."
    )


class RunStatus(BaseModel):
    """Where a recruitment run stands. Poll `GET /runs/{run_id}` until `status` is `complete` or `failed`."""

    run_id: str = Field(description="Id to poll with `GET /runs/{run_id}`.", examples=["5f0c3b7e9a8d4c2e8b1a6f4d3c2b1a09"])
    status: Literal["running", "complete", "failed"] = Field(
        description="`running` until the Recruiter finishes; then `complete` with the report, or `failed` with the error."
    )
    report: RecruitmentReport | None = Field(default=None, description="The Recruitment report, once `complete`.")
    error: str | None = Field(
        default=None,
        description="Why the run failed: an upstream failure (OpenRouter, the RAG API, invalid model output) or an "
        "interrupted run.",
    )
