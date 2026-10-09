"""The Interviewer: a text interview built from the Job, the CV and SwiftAI's Interview Guide."""

import json
import re
from dataclasses import dataclass

from agentic_jev_rag.recruiter import prompts
from agentic_jev_rag.recruiter.context import RecruiterContext
from agentic_jev_rag.recruiter.schemas import CompetencyScore, InterviewAssessment, InterviewTurn, RedFlag, json_schema

QUESTION_SCHEMA = {
    "type": "object",
    "properties": {"done": {"type": "boolean"}, "question": {"type": "string"}, "competency": {"type": "string"}},
    "required": ["done", "question", "competency"],
    "additionalProperties": False,
}
PROBE_SCHEMA = {
    "type": "object",
    "properties": {"question": {"type": "string"}},
    "required": ["question"],
    "additionalProperties": False,
}
ASSESSMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {"type": "array", "items": json_schema(CompetencyScore)},
        "recommendation": {"type": "string", "enum": ["hire", "no_hire", "next_round"]},
        "summary": {"type": "string"},
        "contradictions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"quote": {"type": "string"}, "detail": {"type": "string"}},
                "required": ["quote", "detail"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["scores", "recommendation", "summary", "contradictions"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Question:
    text: str
    competency: str


@dataclass(frozen=True)
class Briefing:
    """What the Interviewer reads before an interview, fetched once: the guide, the Scorecards, the Job and the Dossier."""

    text: str
    competencies: tuple[str, ...]
    dossier: str


def brief(ctx: RecruiterContext, candidate: dict, job: dict) -> Briefing:
    guide = ctx.rag.document("policy:interview-guide")["text"]
    scorecards = ctx.rag.document("policy:scorecards")["text"]
    dossier = dossier_text(candidate)
    text = (
        f"{guide}\n\n{scorecards}\n\nJob ({job['id']}, occupation {job['metadata'].get('occupation')}):\n"
        f"{job['title']}\n{job['text']}\n\nCandidate Dossier ({candidate['id']}):\n{dossier}"
    )
    return Briefing(text, competencies(scorecards, job["metadata"].get("occupation")), dossier)


def competencies(scorecards: str, occupation: str | None) -> tuple[str, ...]:
    """The Scorecard competencies for an occupation (`## Information technology` <-> `information_technology`)."""
    section, names = None, []
    for line in scorecards.splitlines():
        if line.startswith("## "):
            section = line[3:].strip().lower().replace(" ", "_")
        elif section == occupation and (match := re.match(r"- \*\*(.+?):\*\*", line.strip())):
            names.append(match.group(1))
    return tuple(names)


def dossier_text(candidate: dict) -> str:
    """The Candidate's Dossier as text."""
    return f"CV:\n{candidate['text']}"


def next_question(
    ctx: RecruiterContext, briefing: str, turns: list[InterviewTurn], max_questions: int = 4, *, asked: int | None = None
) -> Question | None:
    """The next question to ask, or None when the interview is complete. `asked` counts new questions when `turns` has more."""
    asked = len(turns) if asked is None else asked
    if asked >= max_questions:
        return None
    prompt = (
        f"{briefing}\n\nInterview so far:\n{_transcript(turns) or '(not started)'}\n\n"
        f"Questions asked: {asked} of at most {max_questions}. Give the next question and the Scorecard "
        "competency it assesses (empty for the opening question), or done=true if every competency has been covered."
    )
    answer = _structured(ctx, prompt, "interview_question", QUESTION_SCHEMA, _question)
    if answer["done"] and asked:
        return None
    return Question(answer["question"], answer["competency"])


def next_probe(ctx: RecruiterContext, briefing: str, turns: list[InterviewTurn]) -> str:
    """A forensic follow-up on the last answer, which the Turn check found shallow."""
    prompt = (
        f"{briefing}\n\nInterview so far:\n{_transcript(turns)}\n\nThe candidate's last answer was vague. Ask one short "
        "follow-up about that answer only: the specific detail it left out, such as what they personally did, how, "
        "why they chose it, what went wrong, or the measurable result. Do not move to a new topic."
    )
    return _structured(ctx, prompt, "interview_probe", PROBE_SCHEMA, _probe)["question"]


def probes_sustained(turns: list[InterviewTurn]) -> dict[str, int]:
    """Per probed competency, the probes answered substantively."""
    counts: dict[str, int] = {}
    for turn in turns:
        if turn.kind == "probe":
            key = turn.competency or "General"
            counts[key] = counts.get(key, 0) + (turn.check == "substantive")
    return counts


def assess(
    ctx: RecruiterContext, briefing: str, candidate_id: str, job_id: str, turns: list[InterviewTurn],
    *, red_flags: list[RedFlag] = (), signals: list[dict] = (),
) -> InterviewAssessment:
    prompt = (
        f"{briefing}\n\nInterview transcript:\n{_transcript(turns)}\n\n{_signals_text(signals)}"
        "Score each Scorecard competency for this Job from 1 to 5 with a rationale quoting the candidate, and "
        "recommend hire, next_round or no_hire following the Interview Guide's scoring rule. In `contradictions`, list "
        "only the flagged possible contradictions where the answer truly conflicts with the Dossier, quoting the "
        "candidate and saying what the Dossier says; otherwise return an empty list."
    )

    def build(answer: dict) -> InterviewAssessment:
        found = [RedFlag(kind="contradiction", **c) for c in answer.pop("contradictions", [])]
        return InterviewAssessment(candidate_id=candidate_id, job_id=job_id, turns=turns, red_flags=[*red_flags, *found], **answer)

    return _structured(ctx, prompt, "interview_assessment", ASSESSMENT_SCHEMA, build)


def _signals_text(signals: list[dict]) -> str:
    """Interview signals for the assessment: advisory evidence per answer, and the possible contradictions to check."""
    if not signals:
        return ""
    lines = ["Evidence signals per answer (from a fast classifier; advisory, check them against the transcript):"]
    lines += [
        f"- Answer {s['turn']}: " + ", ".join(f"{name} {score:.1f}" for name, score in s["evidence"].items())
        for s in signals
    ]
    flagged = [s for s in signals if s["consistency"] == "contradicts"]
    if flagged:
        lines.append("Possible contradictions with the Dossier:")
        lines += [f"- Answer {s['turn']}: {s['answer']}" for s in flagged]
    return "\n".join(lines) + "\n\n"


def simulated_answer(ctx: RecruiterContext, candidate: dict, question: str) -> str:
    """A Simulated candidate's answer, played by the System 1 baseline from the CV."""
    completion = ctx.gateway.complete(
        ctx.config.models.system_1_baseline,
        f"CV:\n{candidate['text']}\n\nInterview question: {question}",
        system=prompts.SIMULATED_CANDIDATE,
    )
    return completion.text


class ModelOutputError(Exception):
    """The model's structured output stayed invalid after a retry."""


def _structured(ctx: RecruiterContext, prompt: str, name: str, schema: dict, build):
    """Ask for structured output and build it; retry once with the validation error fed back."""
    feedback, error = "", None
    for _ in range(2):
        completion = ctx.gateway.complete(
            ctx.config.models.system_2_model, prompt + feedback, system=prompts.INTERVIEWER, response_schema=(name, schema)
        )
        try:
            return build(json.loads(completion.text or ""))
        except (ValueError, TypeError, KeyError) as invalid:  # includes JSON and Pydantic validation errors
            error = invalid
            feedback = f"\n\nYour previous answer was invalid ({invalid}). Answer again, following the schema and its ranges."
    raise ModelOutputError(f"{name}: invalid model output after a retry: {error}")


def _question(answer: dict) -> dict:
    if not isinstance(answer.get("done"), bool) or not isinstance(answer.get("question"), str) or not isinstance(answer.get("competency"), str):
        raise ValueError("expected a boolean `done`, a string `question` and a string `competency`")
    if not answer["done"] and not answer["question"].strip():
        raise ValueError("`question` must not be empty when `done` is false")
    return answer


def _probe(answer: dict) -> dict:
    if not isinstance(answer.get("question"), str) or not answer["question"].strip():
        raise ValueError("expected a non-empty string `question`")
    return answer


def _transcript(turns: list[InterviewTurn]) -> str:
    return "\n".join(
        f"Q{n}{' (follow-up)' if t.kind == 'probe' else ''}: {t.question}\nA{n}: {t.answer}" for n, t in enumerate(turns, start=1)
    )
