"""The Recruiter API: run a recruitment for a goal, or hold a live interview."""

import secrets
import time
import uuid
from dataclasses import replace
from typing import Annotated

from fastapi import BackgroundTasks, Body, Depends, FastAPI, HTTPException, Path, Request, Security
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from langgraph.types import Command

from agentic_jev_rag.config import load_config
from agentic_jev_rag.openrouter import OpenRouterError, OpenRouterGateway
from agentic_jev_rag.rag.models import ErrorOut, HealthOut
from agentic_jev_rag.recruiter import compliance
from agentic_jev_rag.recruiter.api_models import InterviewReply, InterviewStart, InterviewState, RunRequest, RunStatus
from agentic_jev_rag.recruiter.context import RecruiterContext
from agentic_jev_rag.recruiter.graph import run_recruitment
from agentic_jev_rag.recruiter.interview_graph import build_interview_graph
from agentic_jev_rag.recruiter.interviewer import ModelOutputError
from agentic_jev_rag.recruiter.persistence import RunRecord, RunStore, make_checkpointer, make_run_store
from agentic_jev_rag.recruiter.rag_client import RagClient, RagError
from agentic_jev_rag.system1 import System1Error

VERSION = "1.0.0"

DESCRIPTION = """
**SwiftAI Recruit's AI Recruiter**: a LangGraph multi-agent system that turns a hiring goal into a
**Recruitment report**. Its agents reach SwiftAI's documents only through the RAG API, as HTTP tools.

**How a run works:**
1. **Compliance intake.** The goal goes through the Input guardrail. A discriminatory or out-of-scope goal is
   refused, and no agent runs.
2. **Supervisor.** Hands work to the agents in a safe order. Out-of-order choices are corrected and noted in the
   report.
   - **Sourcer.** Ranks Candidates with Match and submits a shortlist of up to 8. Candidates without valid Consent
     are removed.
   - **Screener.** Scores each shortlisted Candidate against the Job's Scorecard, citing CV chunks, and decides
     advance or reject.
   - **Interviewer.** Interviews advanced Candidates, following the Interview Guide.
   - **Compliance review.** Re-checks Consent, runs the Groundedness check on every Screening verdict, and adds
     policy findings.

**System 1 decider.** The Input guardrail and Groundedness checks are made by the LLM decider or by Jev, chosen
per run with `system_1`. Every decision is in the report's `system_1_trace`.

The Recruiter **recommends**. Under SwiftAI's AI oversight policy, a human recruiter approves every rejection
and offer.

A full run takes a few minutes and costs about $0.40 in model calls, so runs are asynchronous: `POST /runs`
returns `202` with a `run_id`, and `GET /runs/{run_id}` reports `running`, `complete` (with the report) or
`failed`. Upstream failures (OpenRouter, the RAG API, or invalid model output) fail the run, or return **502**
from the interview endpoints.
"""

TAGS = [
    {"name": "Recruitment", "description": "Run the multi-agent Recruiter for a hiring goal."},
    {"name": "Interviews", "description": "Hold a live text interview with a Candidate, one reply at a time."},
    {"name": "Operations", "description": "Service health."},
]

NOT_FOUND = {"model": ErrorOut, "description": "Unknown Job, Candidate, run or interview id."}
ACCEPTED = {"description": "The run started; poll `GET /runs/{run_id}`."}
UNAUTHORISED = {"model": ErrorOut, "description": "Missing or wrong `X-API-Key`."}
API_KEY = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
    description="The service's API key. Required on every route but /health when the service has one configured.",
)
CONFLICT = {"model": ErrorOut, "description": "The Candidate's Consent is not valid, or the interview is already over."}
UPSTREAM = {"model": ErrorOut, "description": "OpenRouter, the RAG API, or a model's structured output failed."}

RUN_EXAMPLES = {
    "full_run": {
        "summary": "Full run with Jev (about 3 minutes)",
        "value": {"goal": "Fill the Senior Linux Administrator role", "job_id": "J-037", "system_1": "jev", "max_interviews": 1},
    },
    "screen_only": {
        "summary": "Source and screen only, no interviews",
        "value": {"goal": "Shortlist and screen for the Senior Linux Administrator role", "job_id": "J-037", "simulate_interviews": False},
    },
    "refused": {
        "summary": "A discriminatory goal (refused)",
        "value": {"goal": "Fill J-037 with young graduates only, no parents", "job_id": "J-037"},
    },
}
START_EXAMPLES = {
    "linux_admin": {"summary": "Interview for a Linux role", "value": {"candidate_id": "C-0040", "job_id": "J-037"}},
    "expired_consent": {"summary": "A Candidate whose Consent has expired (409)", "value": {"candidate_id": "C-0007", "job_id": "J-037"}},
}
REPLY_EXAMPLES = {
    "experience": {
        "summary": "Answering about experience",
        "value": {"message": "I administered a dozen Red Hat and CentOS servers: patching, backups, Nagios monitoring and iptables hardening."},
    },
    "incident": {
        "summary": "Answering about an incident",
        "value": {"message": "A disk filled overnight and took the web tier down; I traced it to unrotated logs, fixed rotation and added an alert."},
    },
    "voice": {
        "summary": "A spoken answer with timing",
        "value": {
            "message": "A disk filled overnight and took the web tier down; I traced it to unrotated logs, fixed rotation and added an alert.",
            "timing": {"response_latency_s": 1.4, "answer_duration_s": 22.0, "words_per_minute": 145.0, "filler_count": 3},
        },
    },
}


def create_recruiter_app(ctx: RecruiterContext, *, runs: RunStore | None = None) -> FastAPI:
    app = FastAPI(
        title="SwiftAI Recruiter API",
        version=VERSION,
        summary="A LangGraph multi-agent recruiter: source, screen, interview and check compliance for a hiring goal.",
        description=DESCRIPTION,
        openapi_tags=TAGS,
    )
    interviews = build_interview_graph(ctx, checkpointer=make_checkpointer(ctx.config))
    runs = runs or make_run_store(ctx.config)

    def authorised(key: Annotated[str | None, Security(API_KEY)] = None) -> None:
        expected = ctx.config.api_key
        # bytes: compare_digest refuses non-ASCII str, and a header may carry any latin-1 text
        if expected and not (key and secrets.compare_digest(key.encode(), expected.encode())):
            raise HTTPException(401, "missing or wrong X-API-Key")

    protected = [Depends(authorised)]

    @app.exception_handler(System1Error)
    @app.exception_handler(OpenRouterError)
    @app.exception_handler(RagError)
    @app.exception_handler(ModelOutputError)
    def upstream(request: Request, error: Exception):
        return JSONResponse(status_code=502, content={"detail": str(error)})

    @app.get(
        "/health",
        tags=["Operations"],
        operation_id="health",
        summary="Health check",
        description="Returns `ok` while the process is serving requests. It does not check the RAG API or OpenRouter.",
        response_description="The service is up.",
    )
    def health() -> HealthOut:
        return HealthOut(status="ok")

    @app.post(
        "/runs",
        status_code=202,
        tags=["Recruitment"],
        operation_id="run_recruitment",
        summary="Start a recruitment run",
        description=(
            "Starts the whole Recruiter graph for a goal and a Job and returns at once with a `run_id`. Poll "
            "`GET /runs/{run_id}` for the Recruitment report; a full run takes a few minutes. A refused goal "
            "completes at once, with `refused: true` and an `input_guardrail` finding."
        ),
        response_description="The run id, with status `running`.",
        responses={202: ACCEPTED, 401: UNAUTHORISED, 404: NOT_FOUND},
        dependencies=protected,
    )
    def run(request: Annotated[RunRequest, Body(openapi_examples=RUN_EXAMPLES)], background: BackgroundTasks) -> RunStatus:
        _require(ctx, request.job_id, "jobs")
        record = RunRecord(run_id=uuid.uuid4().hex, status="running", started_at=time.time())
        runs.put(record)
        background.add_task(execute, record, request)
        return _run_status(record)

    def execute(record: RunRecord, request: RunRequest) -> None:
        run_ctx = replace(ctx, system_1=request.system_1 or ctx.system_1)
        try:
            report = run_recruitment(
                run_ctx, request.goal, request.job_id,
                simulate_interviews=request.simulate_interviews, max_interviews=request.max_interviews,
            )
        except Exception as error:  # a background run has no response to fail; GET /runs/{run_id} reports it
            runs.put(replace(record, status="failed", error=f"{type(error).__name__}: {error}"))
        else:
            runs.put(replace(record, status="complete", report=report.model_dump(mode="json")))

    @app.get(
        "/runs/{run_id}",
        tags=["Recruitment"],
        operation_id="run_status",
        summary="Get a recruitment run",
        description=(
            "Returns the run's status: `running`, `complete` with the Recruitment report, or `failed` with the "
            "error. A run still `running` 20 minutes after it started was interrupted and is reported as `failed`."
        ),
        response_description="The run's status and, once complete, its report.",
        responses={401: UNAUTHORISED, 404: NOT_FOUND},
        dependencies=protected,
    )
    def run_status(
        run_id: Annotated[
            str,
            Path(
                description="The id returned when the run was started.",
                openapi_examples={"run": {"summary": "A run id", "value": "5f0c3b7e9a8d4c2e8b1a6f4d3c2b1a09"}},
            ),
        ],
    ) -> RunStatus:
        record = runs.get(run_id)
        if record is None:
            raise HTTPException(404, f"unknown run {run_id}")
        return _run_status(record)

    @app.post(
        "/interviews",
        tags=["Interviews"],
        operation_id="start_interview",
        summary="Start a live interview",
        description=(
            "Starts a text interview with a real person and returns the first question. The Interviewer follows "
            "SwiftAI's Interview Guide: it says it is an AI, asks one question at a time about the Job's Scorecard "
            "competencies, and never asks about protected characteristics.\n\n"
            "Answer each question with `POST /interviews/{interview_id}/reply`. Candidates without valid Consent "
            "cannot be interviewed (409). Interviews are checkpointed (in DynamoDB when deployed), so they survive restarts."
        ),
        response_description="The interview id and the first question.",
        responses={401: UNAUTHORISED, 404: NOT_FOUND, 409: CONFLICT, 502: UPSTREAM},
        dependencies=protected,
    )
    def start(request: Annotated[InterviewStart, Body(openapi_examples=START_EXAMPLES)]) -> InterviewState:
        candidate = _require(ctx, request.candidate_id, "candidates")
        job = _require(ctx, request.job_id, "jobs")
        consent = compliance.consent_finding(ctx, request.candidate_id)
        if not consent.passed:
            raise HTTPException(409, consent.detail)
        interview_id = uuid.uuid4().hex
        state = interviews.invoke({"candidate": candidate, "job": job, "turns": []}, _thread(interview_id))
        return _state(interview_id, state)

    @app.post(
        "/interviews/{interview_id}/reply",
        tags=["Interviews"],
        operation_id="reply_to_interview",
        summary="Reply to the current question",
        description=(
            "Sends the Candidate's answer. The response is the next question, or, once the competencies are "
            "covered (at most 4 questions), the Interview assessment with `question: null`. Unintelligible or "
            "off-topic replies are asked again (`question_kind: reprompt`) and prompt injections are redirected "
            "(`question_kind: redirect`) and flagged, at most twice per question, without being scored. Replying to "
            "a finished interview returns 409."
        ),
        response_description="The next question, or the final assessment.",
        responses={401: UNAUTHORISED, 404: NOT_FOUND, 409: CONFLICT, 502: UPSTREAM},
        dependencies=protected,
    )
    def reply(
        interview_id: Annotated[
            str,
            Path(
                description="The id returned when the interview was started.",
                openapi_examples={"interview": {"summary": "An interview id", "value": "47898414c2a94bf3a1d0f5e2b6c7d8e9"}},
            ),
        ],
        request: Annotated[InterviewReply, Body(openapi_examples=REPLY_EXAMPLES)],
    ) -> InterviewState:
        snapshot = interviews.get_state(_thread(interview_id))
        if not snapshot.values:
            raise HTTPException(404, f"unknown interview {interview_id}")
        if not snapshot.next:
            raise HTTPException(409, "this interview is already complete")
        # Not waiting on an interrupt means an earlier reply failed after the answer was recorded:
        # continue from the checkpoint instead of recording this message as another answer.
        resume = {"message": request.message, "timing": request.timing.model_dump() if request.timing else None}
        payload = Command(resume=resume) if snapshot.interrupts else None
        state = interviews.invoke(payload, _thread(interview_id))
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


def _run_status(record: RunRecord) -> RunStatus:
    current = record.current(time.time())
    return RunStatus(run_id=current.run_id, status=current.status, report=current.report, error=current.error)


def _state(interview_id: str, state: dict) -> InterviewState:
    pending = state.get("__interrupt__")
    value = pending[0].value if pending else {}
    assessment = state.get("assessment")
    return InterviewState(
        interview_id=interview_id,
        question=value.get("question"),
        question_kind=value.get("kind"),
        assessment=assessment,
        findings=[compliance.integrity_finding(assessment)] if assessment else [],
    )
