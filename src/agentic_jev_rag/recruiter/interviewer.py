"""The Interviewer: a text interview built from the Job, the CV and SwiftAI's Interview Guide."""

import json

from agentic_jev_rag.recruiter import prompts
from agentic_jev_rag.recruiter.context import RecruiterContext
from agentic_jev_rag.recruiter.schemas import CompetencyScore, InterviewAssessment, InterviewTurn, json_schema

QUESTION_SCHEMA = {
    "type": "object",
    "properties": {"done": {"type": "boolean"}, "question": {"type": "string"}},
    "required": ["done", "question"],
    "additionalProperties": False,
}
ASSESSMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {"type": "array", "items": json_schema(CompetencyScore)},
        "recommendation": {"type": "string", "enum": ["hire", "no_hire", "next_round"]},
        "summary": {"type": "string"},
    },
    "required": ["scores", "recommendation", "summary"],
    "additionalProperties": False,
}


def next_question(
    ctx: RecruiterContext, candidate: dict, job: dict, turns: list[InterviewTurn], max_questions: int = 4
) -> str | None:
    """The next question to ask, or None when the interview is complete."""
    if len(turns) >= max_questions:
        return None
    prompt = (
        f"{_briefing(ctx, candidate, job)}\n\nInterview so far:\n{_transcript(turns) or '(not started)'}\n\n"
        f"Questions asked: {len(turns)} of at most {max_questions}. Give the next question, or done=true "
        "if every competency has been covered."
    )
    completion = ctx.gateway.complete(
        ctx.config.models.system_2_model, prompt, system=prompts.INTERVIEWER,
        response_schema=("interview_question", QUESTION_SCHEMA),
    )
    answer = json.loads(completion.text)
    if answer.get("done") and turns:
        return None
    return answer["question"]


def assess(ctx: RecruiterContext, candidate: dict, job: dict, turns: list[InterviewTurn]) -> InterviewAssessment:
    prompt = (
        f"{_briefing(ctx, candidate, job)}\n\nInterview transcript:\n{_transcript(turns)}\n\n"
        "Score each Scorecard competency for this Job from 1 to 5 with a rationale quoting the candidate, and "
        "recommend hire, next_round or no_hire following the Interview Guide's scoring rule."
    )
    completion = ctx.gateway.complete(
        ctx.config.models.system_2_model, prompt, system=prompts.INTERVIEWER,
        response_schema=("interview_assessment", ASSESSMENT_SCHEMA),
    )
    return InterviewAssessment(
        candidate_id=candidate["id"], job_id=job["id"], turns=turns, **json.loads(completion.text)
    )


def simulated_answer(ctx: RecruiterContext, candidate: dict, question: str) -> str:
    """A Simulated candidate's answer, played by the System 1 baseline from the CV."""
    completion = ctx.gateway.complete(
        ctx.config.models.system_1_baseline,
        f"CV:\n{candidate['text']}\n\nInterview question: {question}",
        system=prompts.SIMULATED_CANDIDATE,
    )
    return completion.text


def _briefing(ctx: RecruiterContext, candidate: dict, job: dict) -> str:
    guide = ctx.rag.document("policy:interview-guide")["text"]
    scorecards = ctx.rag.document("policy:scorecards")["text"]
    return (
        f"{guide}\n\n{scorecards}\n\nJob ({job['id']}, occupation {job['metadata'].get('occupation')}):\n"
        f"{job['title']}\n{job['text']}\n\nCandidate CV ({candidate['id']}):\n{candidate['text']}"
    )


def _transcript(turns: list[InterviewTurn]) -> str:
    return "\n".join(f"Q{n}: {t.question}\nA{n}: {t.answer}" for n, t in enumerate(turns, start=1))
