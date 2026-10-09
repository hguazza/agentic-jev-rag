"""Request and response models of the Recruiter API (agent outputs live in `schemas.py`)."""

from typing import Literal

from pydantic import BaseModel, Field

from agentic_jev_rag.recruiter.schemas import InterviewAssessment


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


class InterviewState(BaseModel):
    """Where an interview stands: the next question, or the final assessment."""

    interview_id: str = Field(description="Id to reply to; keep it for the whole interview.", examples=["47898414c2a94bf3a1d0f5e2b6c7d8e9"])
    question: str | None = Field(description="The next question for the Candidate; null when the interview is over.")
    assessment: InterviewAssessment | None = Field(description="The Interviewer's assessment; set once the interview is over.")
