"""The AI engineer showcase pool: Scorecard, Persona grid, generation and replacement."""

import json
import re
from pathlib import Path

import pytest

from agentic_jev_rag.corpus.ai_pool import FAKE_IDS, TIERS, fake_candidate, load_personas
from agentic_jev_rag.gateway import FakeModelGateway

CORPUS = Path(__file__).parents[1] / "corpus"
AI_COMPETENCIES = ["LLM systems", "Retrieval and evaluation", "Production engineering", "Cloud delivery"]
PERSONAS = CORPUS / "ai_pool" / "personas.toml"
PROFILE = {"headline": "LLM platform engineer", "years_experience": 8, "skills": ["Python", "LangGraph", "FastAPI"]}


def test_the_ai_engineering_scorecard_has_four_competencies():
    text = (CORPUS / "policies" / "scorecards.md").read_text(encoding="utf-8")
    section = text.split("## AI engineering\n", 1)[1].split("\n## ", 1)[0]

    assert re.findall(r"- \*\*(.+?):\*\*", section) == AI_COMPETENCIES



def pool_gateway():
    """CVs are plain text; profiles and jobs are structured."""

    def complete(model, prompt, system, schema):
        if schema is None:
            return "SENIOR AI ENGINEER\nSummary\nBuilds LLM agents.\n"
        if schema[0] == "candidate_profile":
            return json.dumps(PROFILE)
        if schema[0] == "job_posting":
            return json.dumps({
                "title": "AI/ML Engineer", "about": "micro1 builds AI recruiters.", "responsibilities": "Build agents.",
                "requirements": "Python, LangGraph.", "benefits": "Remote.", "required_skills": ["Python"], "seniority": "senior",
            })
        raise AssertionError(f"unexpected schema {schema[0]}")

    return FakeModelGateway(complete_responder=complete)


def test_the_committed_persona_grid_has_the_agreed_tiers():
    personas = load_personas(PERSONAS)

    assert [p.id for p in personas] == FAKE_IDS
    assert {t: sum(p.tier == t for p in personas) for t in TIERS} == TIERS
    assert [p.id for p in personas if p.honesty == "bluffer"] == ["C-0211", "C-0212"]


def test_a_grid_with_the_wrong_tier_counts_is_rejected(tmp_path):
    text = PERSONAS.read_text(encoding="utf-8").replace('tier = "bluffer"', 'tier = "strong"', 1)
    (tmp_path / "personas.toml").write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="tiers"):
        load_personas(tmp_path / "personas.toml")


def test_a_fake_candidate_has_profile_metadata_and_no_persona():
    persona = load_personas(PERSONAS)[10]  # C-0211, a bluffer
    gateway = pool_gateway()

    candidate = fake_candidate(
        gateway, writer="fake/s2", extractor="fake/s1", persona=persona, name="Ana Costa", consent_until="2027-10-09"
    )

    assert candidate.id == "C-0211" and candidate.collection == "candidates" and candidate.title == "Ana Costa — CV"
    assert candidate.metadata == {
        "name": "Ana Costa", "occupation": "ai_engineering", "headline": "LLM platform engineer",
        "location": persona.location, "years_experience": persona.years_experience,
        "skills": ["Python", "LangGraph", "FastAPI"], "consent_until": "2027-10-09", "source": "ai_pool",
    }
    cv_prompt = gateway.calls[0][2]["prompt"]
    assert persona.background in cv_prompt and "states only the claims" in cv_prompt
    assert "bluffer" not in candidate.text and "tier" not in json.dumps(candidate.metadata)
