"""The Interview Coach: a debrief of one finished voice interview, for the candidate. Showcase only (ADR 0015).

Run: uv run python -m agentic_jev_rag.coach results/voice-interviews/<id>.json [--out results/ai-engineer-showcase]
"""

from dataclasses import asdict, dataclass
from pathlib import Path

from agentic_jev_rag.corpus.documents import Document

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
