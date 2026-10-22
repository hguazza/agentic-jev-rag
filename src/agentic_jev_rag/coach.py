"""The Interview Coach: a debrief of one finished voice interview, for the candidate. Showcase only (ADR 0015).

Run: uv run python -m agentic_jev_rag.coach results/voice-interviews/<id>.json [--out results/ai-engineer-showcase]
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from agentic_jev_rag.corpus.documents import Document
from agentic_jev_rag.recruiter.interviewer import ModelOutputError
from agentic_jev_rag.system1 import System1Decider, System1TraceEntry

OUT_DIR = Path("results") / "ai-engineer-showcase"
FOLDED_KINDS = ("reprompt", "redirect")  # these repeat the previous question


class CoachRefused(Exception):
    """The interview cannot be coached. Raised before any model call."""


@dataclass
class QuestionGroup:
    """One question as the candidate heard it, with every answer given to it (reprompts and redirects included)."""

    number: int
    question: str
    competency: str  # empty for the opening question
    answers: list[str]
    timings: list[dict]


def group_turns(turns: list[dict]) -> list[QuestionGroup]:
    """Questions in interview order. Injected and empty answers are left out; a question left with none is dropped."""
    groups: list[QuestionGroup] = []
    for turn in turns:
        if turn.get("kind") not in FOLDED_KINDS or not groups:
            groups.append(QuestionGroup(0, turn["question"], turn.get("competency", ""), [], []))
        if turn.get("check") != "injection" and turn["answer"].strip():
            groups[-1].answers.append(turn["answer"])
            groups[-1].timings.append(turn.get("timing") or {})
    kept = [group for group in groups if group.answers]
    for number, group in enumerate(kept, start=1):
        group.number = number
    return kept


def check_interview(interview: dict) -> None:
    interview_id = interview.get("interview_id")
    if interview.get("error"):
        raise CoachRefused(f"interview {interview_id} ended with an error: {interview['error']}")
    if not interview.get("finished") or not interview.get("assessment"):
        raise CoachRefused(f"interview {interview_id} is not finished: there is no assessment to coach")


def find_candidate(documents: list[Document], candidate_id: str) -> dict:
    for document in documents:
        if document.collection == "candidates" and document.id == candidate_id:
            if not document.metadata.get("dossier"):
                raise CoachRefused(f"{candidate_id} has no Dossier: model answers need the CV and Dossier to check against")
            return asdict(document)
    raise CoachRefused(f"{candidate_id} is not in the corpus")


def scorecard_section(scorecards: str, occupation: str | None) -> str:
    """The competency lines of one occupation's Scorecard (`## AI engineering` <-> `ai_engineering`)."""
    section, lines = None, []
    for line in scorecards.splitlines():
        if line.startswith("## "):
            section = line[3:].strip().lower().replace(" ", "_")
        elif section == occupation and line.strip().startswith("- **"):
            lines.append(line.strip())
    return "\n".join(lines)


COACH = (
    "You are an interview coach. You debrief a candidate on an interview they have finished, so they do better in the "
    "next one. Be direct and specific, quote their words, and be kind. Every claim in a model answer must come from "
    "the candidate's CV or Dossier: never invent experience, numbers or employers, and never repeat a claim the "
    "Dossier contradicts."
)
DEBRIEF_TASK = (
    "Debrief the candidate on this question. Say what the question was testing, in terms of its competency; what in "
    "their answer worked and what cost points, each point with a short quote from the answer; then write a tighter "
    "model answer, in the first person, about 60 to 90 seconds of speech (150 to 220 words), using only facts from "
    "the CV and Dossier. List in facts_used the CV and Dossier lines the model answer relies on. Where a red flag "
    "quotes this answer, the model answer says what the Dossier supports instead."
)
POINT_SCHEMA = {
    "type": "object",
    "properties": {"point": {"type": "string"}, "quote": {"type": "string"}},
    "required": ["point", "quote"],
    "additionalProperties": False,
}
DEBRIEF_SCHEMA = {
    "type": "object",
    "properties": {
        "looking_for": {"type": "string"},
        "what_worked": {"type": "array", "items": POINT_SCHEMA},
        "what_cost_points": {"type": "array", "items": POINT_SCHEMA},
        "model_answer": {"type": "string"},
        "facts_used": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["looking_for", "what_worked", "what_cost_points", "model_answer", "facts_used"],
    "additionalProperties": False,
}
REWRITE = (
    "\n\nA groundedness check found your previous model answer not supported by the CV and Dossier "
    "(verdict: ungrounded):\n{model_answer}\n\nWrite the debrief again. Keep every claim in the model answer to facts "
    "in the CV and Dossier, and list them in facts_used."
)


@dataclass
class Debrief:
    number: int
    question: str
    competency: str
    answers: list[str]
    looking_for: str
    what_worked: list[dict]  # {"point", "quote"}
    what_cost_points: list[dict]
    model_answer: str
    facts_used: list[str]
    grounded: bool  # False: the System 1 check rejected the model answer twice


def debrief_question(
    gateway, model: str, decider: System1Decider, group: QuestionGroup, *, dossier: str, scorecard: str, scores: list[dict], red_flags: list[dict]
) -> tuple[Debrief, list[System1TraceEntry]]:
    """One structured debrief, its model answer checked by System 1 and rewritten once if ungrounded."""
    score = next((s for s in scores if s["competency"] == group.competency), None)
    scored = f"{score['score']}/5. {score['rationale']}" if score else "not scored on its own (it informs every competency)."
    flags = [f for f in red_flags if any(f["quote"] in answer for answer in group.answers)]
    flagged = "\n".join(f"- You said: \"{f['quote']}\". {f['detail']}" for f in flags) or "None."
    answers = "\n\n".join(f"Answer {n}: {a}" for n, a in enumerate(group.answers, start=1))
    prompt = (
        f"{DEBRIEF_TASK}\n\nScorecard for this role:\n{scorecard}\n\n"
        f"Question {group.number} ({group.competency or 'opening question'}): {group.question}\n\n"
        f"The candidate's answers (speech-to-text):\n{answers}\n\nHow the interviewer scored {group.competency or 'it'}: {scored}\n\n"
        f"Red flags on this answer:\n{flagged}\n\nThe candidate's CV and Dossier:\n{dossier}"
    )
    data = _structured(gateway, model, prompt, "debrief", DEBRIEF_SCHEMA, _debrief)
    trace = []
    for attempt in range(2):
        decision = decider.groundedness(data["model_answer"], data["facts_used"] + [dossier])
        trace.append(decision.trace)
        if decision.grounded or attempt == 1:
            break
        data = _structured(gateway, model, prompt + REWRITE.format(model_answer=data["model_answer"]), "debrief", DEBRIEF_SCHEMA, _debrief)
    return Debrief(group.number, group.question, group.competency, group.answers, grounded=decision.grounded, **data), trace


def _structured(gateway, model: str, prompt: str, name: str, schema: dict, build):
    """Ask for structured output and build it; retry once with the validation error fed back."""
    feedback, error = "", None
    for _ in range(2):
        completion = gateway.complete(model, prompt + feedback, system=COACH, response_schema=(name, schema))
        try:
            return build(json.loads(completion.text or ""))
        except (ValueError, TypeError, KeyError) as invalid:
            error = invalid
            feedback = f"\n\nYour previous answer was invalid ({invalid}). Answer again, following the schema."
    raise ModelOutputError(f"{name}: invalid model output after a retry: {error}")


def _debrief(answer: dict) -> dict:
    points = answer["what_worked"] + answer["what_cost_points"]
    if not isinstance(answer["looking_for"], str) or not all(isinstance(p["point"], str) and isinstance(p["quote"], str) for p in points):
        raise ValueError("expected a string `looking_for` and a string `point` and `quote` in every point")
    if not isinstance(answer["model_answer"], str) or not answer["model_answer"].strip():
        raise ValueError("`model_answer` must be a non-empty string")
    if not all(isinstance(fact, str) for fact in answer["facts_used"]):
        raise ValueError("`facts_used` must be a list of strings")
    return {key: answer[key] for key in DEBRIEF_SCHEMA["required"]}
