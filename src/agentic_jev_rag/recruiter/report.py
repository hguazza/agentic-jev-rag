"""The Recruitment report: the Recruiter's final output for a goal."""

from typing import Literal

from pydantic import BaseModel, Field

from agentic_jev_rag.rag.models import TraceOut
from agentic_jev_rag.recruiter.schemas import (
    AgentRunNote,
    ComplianceFinding,
    InterviewAssessment,
    ScreeningVerdict,
    Shortlist,
)


class RecruitmentReport(BaseModel):
    """Everything the Recruiter did for a goal: shortlist, verdicts, interviews, Compliance findings and the
    System 1 trace. Recommendations only: a human recruiter approves every rejection and offer."""

    goal: str = Field(description="The hiring goal as given.")
    job_id: str = Field(description="The Job recruited for.", examples=["J-037"])
    system_1: Literal["llm", "jev"] = Field(description="The System 1 decider used for this run.")
    refused: bool = Field(description="True when the Input guardrail refused the goal; no agent ran.")
    shortlist: Shortlist | None = Field(description="The Sourcer's shortlist; null when the goal was refused.")
    verdicts: list[ScreeningVerdict] = Field(description="One Screening verdict per shortlisted Candidate.")
    assessments: list[InterviewAssessment] = Field(description="Interview assessments for advanced Candidates (with a Simulated candidate).")
    findings: list[ComplianceFinding] = Field(description="Compliance findings: the goal, Consent, grounded verdicts and policy.")
    system_1_trace: list[TraceOut] = Field(description="Every System 1 decision made during the run, including inside RAG Ask calls.")
    notes: list[AgentRunNote] = Field(description="How each agent run ended, and any Supervisor corrections or step limits.")
    cost_usd: float = Field(description="Model cost of the Recruiter's own calls (agents and System 1), in US dollars. RAG-side costs are not included.")
    latency_s: float = Field(description="Wall-clock duration of the run, in seconds.")
