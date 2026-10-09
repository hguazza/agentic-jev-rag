"""The Recruiter API: run a recruitment for a goal, or hold a live interview."""

import uuid
from dataclasses import replace
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from langgraph.types import Command
from pydantic import BaseModel, Field

from agentic_jev_rag.config import load_config
from agentic_jev_rag.openrouter import OpenRouterError, OpenRouterGateway
from agentic_jev_rag.recruiter import compliance
from agentic_jev_rag.recruiter.context import RecruiterContext
from agentic_jev_rag.recruiter.graph import run_recruitment
from agentic_jev_rag.recruiter.interview_graph import build_interview_graph
from agentic_jev_rag.recruiter.interviewer import ModelOutputError
from agentic_jev_rag.recruiter.rag_client import RagClient, RagError
from agentic_jev_rag.recruiter.report import RecruitmentReport
from agentic_jev_rag.recruiter.schemas import InterviewAssessment
from agentic_jev_rag.system1 import System1Error


class RunRequest(BaseModel):
    goal: str = Field(min_length=1)
    job_id: str
    system_1: Literal["llm", "jev"] | None = None
    simulate_interviews: bool = True
    max_interviews: int = Field(default=2, ge=0, le=8)


class InterviewStart(BaseModel):
    candidate_id: str
    job_id: str


class InterviewReply(BaseModel):
    message: str = Field(min_length=1)


class InterviewState(BaseModel):
    interview_id: str
    question: str | None
    assessment: InterviewAssessment | None


def create_recruiter_app(ctx: RecruiterContext) -> FastAPI:
    app = FastAPI(title="SwiftAI Recruiter", description="A LangGraph multi-agent recruiter over the SwiftAI RAG API.")
    interviews = build_interview_graph(ctx)

    @app.exception_handler(System1Error)
    @app.exception_handler(OpenRouterError)
    @app.exception_handler(RagError)
    @app.exception_handler(ModelOutputError)
    def upstream(request: Request, error: Exception):
        return JSONResponse(status_code=502, content={"detail": str(error)})

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/runs")
    def run(request: RunRequest) -> RecruitmentReport:
        _require(ctx, request.job_id, "jobs")
        run_ctx = replace(ctx, system_1=request.system_1 or ctx.system_1)
        return run_recruitment(
            run_ctx, request.goal, request.job_id,
            simulate_interviews=request.simulate_interviews, max_interviews=request.max_interviews,
        )

    @app.post("/interviews")
    def start(request: InterviewStart) -> InterviewState:
        candidate = _require(ctx, request.candidate_id, "candidates")
        job = _require(ctx, request.job_id, "jobs")
        consent = compliance.consent_finding(ctx, request.candidate_id)
        if not consent.passed:
            raise HTTPException(409, consent.detail)
        interview_id = uuid.uuid4().hex
        state = interviews.invoke({"candidate": candidate, "job": job, "turns": []}, _thread(interview_id))
        return _state(interview_id, state)

    @app.post("/interviews/{interview_id}/reply")
    def reply(interview_id: str, request: InterviewReply) -> InterviewState:
        snapshot = interviews.get_state(_thread(interview_id))
        if not snapshot.values:
            raise HTTPException(404, f"unknown interview {interview_id}")
        if not snapshot.next:
            raise HTTPException(409, "this interview is already complete")
        state = interviews.invoke(Command(resume=request.message), _thread(interview_id))
        return _state(interview_id, state)

    return app


def create_default_recruiter_app() -> FastAPI:
    config = load_config()
    ctx = RecruiterContext(config=config, gateway=OpenRouterGateway.from_env(), rag=RagClient(config.rag_url), system_1=config.system_1)
    return create_recruiter_app(ctx)


def _require(ctx: RecruiterContext, document_id: str, collection: str) -> dict:
    try:
        document = ctx.rag.document(document_id)
    except RagError as error:
        if "404" in str(error):
            raise HTTPException(404, f"unknown {collection[:-1]} {document_id}") from error
        raise
    if document["collection"] != collection:
        raise HTTPException(404, f"{document_id} is not in {collection}")
    return document


def _thread(interview_id: str) -> dict:
    return {"configurable": {"thread_id": interview_id}}


def _state(interview_id: str, state: dict) -> InterviewState:
    pending = state.get("__interrupt__")
    question = pending[0].value["question"] if pending else None
    return InterviewState(interview_id=interview_id, question=question, assessment=state.get("assessment"))
