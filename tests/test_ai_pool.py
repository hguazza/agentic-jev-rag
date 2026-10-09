"""The AI engineer showcase pool: Scorecard, Persona grid, generation and replacement."""

import re
from pathlib import Path

CORPUS = Path(__file__).parents[1] / "corpus"
AI_COMPETENCIES = ["LLM systems", "Retrieval and evaluation", "Production engineering", "Cloud delivery"]


def test_the_ai_engineering_scorecard_has_four_competencies():
    text = (CORPUS / "policies" / "scorecards.md").read_text(encoding="utf-8")
    section = text.split("## AI engineering\n", 1)[1].split("\n## ", 1)[0]

    assert re.findall(r"- \*\*(.+?):\*\*", section) == AI_COMPETENCIES
