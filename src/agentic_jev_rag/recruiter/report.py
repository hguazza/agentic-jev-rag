"""The Recruitment report: the Recruiter's final output for a goal."""

from pydantic import BaseModel

from agentic_jev_rag.rag.models import TraceOut
from agentic_jev_rag.recruiter.schemas import (
    AgentRunNote,
    ComplianceFinding,
    InterviewAssessment,
    ScreeningVerdict,
    Shortlist,
)


class RecruitmentReport(BaseModel):
    goal: str
    job_id: str
    system_1: str
    refused: bool
    shortlist: Shortlist | None
    verdicts: list[ScreeningVerdict]
    assessments: list[InterviewAssessment]
    findings: list[ComplianceFinding]
    system_1_trace: list[TraceOut]
    notes: list[AgentRunNote]
    cost_usd: float  # the Recruiter's own model calls; RAG-side costs are reported by the RAG API
    latency_s: float
