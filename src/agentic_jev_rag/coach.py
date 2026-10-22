"""The Interview Coach: a debrief of one finished voice interview, for the candidate. Showcase only (ADR 0015).

Run: uv run python -m agentic_jev_rag.coach results/voice-interviews/<id>.json [--out results/ai-engineer-showcase]
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from agentic_jev_rag.corpus.documents import Document
from agentic_jev_rag.gateway import CostMeter
from agentic_jev_rag.openrouter import OpenRouterError
from agentic_jev_rag.recruiter.interviewer import ModelOutputError, dossier_text
from agentic_jev_rag.recruiter.schemas import INTEGRITY_NOTE
from agentic_jev_rag.system1 import System1Decider, System1Error, System1TraceEntry

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


HABITS_TASK = (
    "Coach the candidate on delivery across the whole interview. Give at most three habits to fix, most costly first, "
    "each with the question numbers that show it and one concrete fix. Then, for every red flag, in the same order, "
    "say what the candidate should say instead, using only what the CV and Dossier support. "
    f"Timing never affected the score: {INTEGRITY_NOTE} Use it only to coach delivery."
)
HABITS_SCHEMA = {
    "type": "object",
    "properties": {
        "habits": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"habit": {"type": "string"}, "turns": {"type": "array", "items": {"type": "integer"}}, "fix": {"type": "string"}},
                "required": ["habit", "turns", "fix"],
                "additionalProperties": False,
            },
        },
        "red_flags": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"quote": {"type": "string"}, "say_instead": {"type": "string"}},
                "required": ["quote", "say_instead"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["habits", "red_flags"],
    "additionalProperties": False,
}
MODEL_ERRORS = (OpenRouterError, ModelOutputError, System1Error)


class CoachFailed(Exception):
    """A model call failed mid-run. Nothing is written."""


def delivery_habits(gateway, model: str, groups: list[QuestionGroup], red_flags: list[dict]) -> dict:
    """Up to three delivery habits, and what to say instead for each red flag, in the red flags' order."""
    questions = "\n".join(
        f"Question {g.number} ({g.competency or 'opening'}), reprompted {len(g.answers) - 1} time(s): "
        + "; ".join(f"{t.get('answer_duration_s')} s at {t.get('words_per_minute')} words per minute" for t in g.timings)
        for g in groups
    )
    flags = "\n".join(f"{n}. You said: \"{f['quote']}\". {f['detail']}" for n, f in enumerate(red_flags, start=1)) or "None."
    prompt = f"{HABITS_TASK}\n\nAnswers, with their timing:\n{questions}\n\nRed flags, in order:\n{flags}"

    def build(answer: dict) -> dict:
        if len(answer["habits"]) > 3:
            raise ValueError(f"at most three habits, got {len(answer['habits'])}")
        if len(answer["red_flags"]) != len(red_flags):
            raise ValueError(f"one say_instead per red flag: expected {len(red_flags)}, got {len(answer['red_flags'])}")
        if not all(isinstance(t, int) for h in answer["habits"] for t in h["turns"]):
            raise ValueError("habit turns must be question numbers")
        return {"habits": answer["habits"], "red_flags": answer["red_flags"]}

    return _structured(gateway, model, prompt, "delivery_habits", HABITS_SCHEMA, build)


@dataclass
class Coaching:
    interview_id: str
    candidate_id: str
    job_id: str
    recommendation: str
    scores: list[dict]
    summary: str
    habits: list[dict]
    red_flags: list[dict]  # the assessment's red flags, each with "say_instead"
    debriefs: list[Debrief]
    system_1: str
    cost_usd: float
    trace: list[dict]  # System 1 decisions


def coach(gateway, model: str, decider: System1Decider, interview: dict, candidate: dict, scorecards: str) -> Coaching:
    """Coach one finished interview. Raises CoachRefused before any model call, CoachFailed if a model call fails."""
    check_interview(interview)
    assessment = interview["assessment"]
    groups = group_turns(assessment["turns"])
    meter = CostMeter(gateway)  # System 2 calls; System 1 costs come from the trace
    base = dict(
        interview_id=interview["interview_id"], candidate_id=interview["candidate_id"], job_id=interview["job_id"],
        recommendation=assessment["recommendation"], scores=assessment["scores"], summary=assessment["summary"], system_1=decider.name,
    )
    if not groups:
        return Coaching(**base, habits=[], red_flags=[], debriefs=[], cost_usd=0.0, trace=[])
    dossier = dossier_text(candidate)
    scorecard = scorecard_section(scorecards, candidate["metadata"].get("occupation"))
    red_flags = assessment.get("red_flags", [])
    debriefs, trace = [], []
    for group in groups:
        try:
            debrief, checks = debrief_question(
                meter, model, decider, group, dossier=dossier, scorecard=scorecard, scores=assessment["scores"], red_flags=red_flags
            )
        except MODEL_ERRORS as error:
            raise CoachFailed(f"question {group.number}: {error}") from error
        debriefs.append(debrief)
        trace.extend(checks)
    try:
        delivery = delivery_habits(meter, model, groups, red_flags)
    except MODEL_ERRORS as error:
        raise CoachFailed(f"delivery habits: {error}") from error
    flags = [flag | {"say_instead": fix["say_instead"]} for flag, fix in zip(red_flags, delivery["red_flags"])]
    return Coaching(
        **base, habits=delivery["habits"], red_flags=flags, debriefs=debriefs,
        cost_usd=meter.cost_usd + sum(t.cost_usd for t in trace), trace=[asdict(t) for t in trace],
    )


def render(coaching: Coaching) -> str:
    scores = ", ".join(f"{s['competency']} {s['score']}" for s in coaching.scores)
    lines = [
        f"# Interview coaching: {coaching.candidate_id} for {coaching.job_id}",
        "",
        f"Recommendation: `{coaching.recommendation}`. Scores: {scores}.",
        "",
        coaching.summary,
        "",
    ]
    if not coaching.debriefs:
        lines += ["There was no answered question to coach.", ""]
    else:
        lines += ["## Delivery habits to fix", ""]
        for n, habit in enumerate(coaching.habits, start=1):
            questions = ", ".join(str(t) for t in habit["turns"])
            label = "questions" if len(habit["turns"]) > 1 else "question"
            lines.append(f"{n}. **{habit['habit']}** ({label} {questions}). {habit['fix']}")
        if coaching.red_flags:
            lines += ["", "## Red flags: what to say instead", ""]
            for flag in coaching.red_flags:
                lines.append(f"- You said: \"{flag['quote']}\". {flag['detail']} **Say instead:** {flag['say_instead']}")
        for debrief in coaching.debriefs:
            lines += _debrief_lines(debrief)
    lines += [
        "",
        "---",
        "",
        f"System 1 decider: `{coaching.system_1}`. Cost of this run: ${coaching.cost_usd:.4f}. Interview `{coaching.interview_id}`. "
        "This coaching is for the candidate only: it never reached a recruiter or a score.",
    ]
    return "\n".join(lines) + "\n"


def _debrief_lines(debrief: Debrief) -> list[str]:
    lines = ["", f"## Question {debrief.number}: {debrief.competency or 'Opening'}", "", f"> {debrief.question}", ""]
    lines += [f"**What it was looking for:** {debrief.looking_for}", "", "<details><summary>What you said</summary>", ""]
    lines += [f"> {answer}\n" for answer in debrief.answers] + ["</details>", "", "**What worked:**", ""]
    lines += [f"- {p['point']}: \"{p['quote']}\"" for p in debrief.what_worked] + ["", "**What cost points:**", ""]
    lines += [f"- {p['point']}: \"{p['quote']}\"" for p in debrief.what_cost_points] + ["", "**A tighter answer:**", "", debrief.model_answer]
    if not debrief.grounded:
        lines += ["", "*Not fully supported by your CV: check every claim before you use it.*"]
    return lines


def write_outputs(coaching: Coaching, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "coaching.json").write_text(json.dumps(asdict(coaching), indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "coaching.md").write_text(render(coaching), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("interview", type=Path, help="a voice interview results file (results/voice-interviews/<id>.json)")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    from agentic_jev_rag.config import load_config
    from agentic_jev_rag.corpus.documents import load_corpus
    from agentic_jev_rag.openrouter import OpenRouterGateway
    from agentic_jev_rag.system1 import make_deciders

    config = load_config()
    interview = json.loads(args.interview.read_text(encoding="utf-8"))
    try:
        check_interview(interview)
        candidate = find_candidate(load_corpus(config.corpus_dir), interview["candidate_id"])
        scorecards = (Path(config.corpus_dir) / "policies" / "scorecards.md").read_text(encoding="utf-8")
        gateway = OpenRouterGateway.from_env()
        decider = make_deciders(config, gateway)[config.system_1]
        coaching = coach(gateway, config.models.system_2_model, decider, interview, candidate, scorecards)
    except (CoachRefused, CoachFailed) as error:
        raise SystemExit(f"coach: {error}") from error
    write_outputs(coaching, args.out)
    print(f"wrote {args.out / 'coaching.md'} (${coaching.cost_usd:.4f})")


if __name__ == "__main__":
    main()
