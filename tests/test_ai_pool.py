"""The AI engineer showcase pool: Scorecard, Persona grid, generation and replacement."""

import json
import re
import shutil
from datetime import date
from pathlib import Path

import pytest

from agentic_jev_rag.corpus.ai_pool import AUTHOR_ID, FAKE_IDS, POOL_IDS, TIERS, build_pool, fake_candidate, load_personas, replace_pool
from agentic_jev_rag.corpus.documents import load_corpus
from agentic_jev_rag.gateway import FakeModelGateway

CORPUS = Path(__file__).parents[1] / "corpus"
AI_COMPETENCIES = ["LLM systems", "Retrieval and evaluation", "Production engineering", "Cloud delivery"]
PERSONAS = CORPUS / "ai_pool" / "personas.toml"
FIXTURE_CORPUS = Path(__file__).parent / "fixtures" / "corpus"
TODAY = date(2026, 10, 9)
LINKEDIN = {"headline": "AI Engineer", "about": "I build agents.", "positions": [{"title": "AI Engineer", "company": "Acme", "years": "2023-2026"}]}
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


def author_dir(tmp_path, *, skip=()):
    directory = tmp_path / "henrique"
    directory.mkdir()
    files = {
        "profile.toml": 'name = "Henrique Guazzelli"\nlocation = "Porto Alegre, RS, BR"\n',
        "cv.md": "AI ENGINEER\nSummary\nBuilt a LangGraph recruiter.\n",
        "cover_letter.md": "I am applying for the AI/ML Engineer role.",
        "linkedin.json": json.dumps(LINKEDIN),
    }
    for name, text in files.items():
        if name not in skip:
            (directory / name).write_text(text, encoding="utf-8")
    return directory


def pool(tmp_path, gateway=None):
    return build_pool(
        gateway or pool_gateway(), writer="fake/s2", extractor="fake/s1", personas=load_personas(PERSONAS),
        author_dir=author_dir(tmp_path), description="micro1 AI/ML Engineer.", seed=7, today=TODAY,
    )


def test_the_pool_has_twelve_fakes_the_author_and_the_micro1_job(tmp_path):
    candidates, job = pool(tmp_path)

    assert [c.id for c in candidates] == POOL_IDS
    assert len({c.metadata["name"] for c in candidates}) == 13
    assert all(c.consent_valid_on(TODAY) and c.metadata["source"] == "ai_pool" for c in candidates)
    author = candidates[-1]
    assert author.id == AUTHOR_ID and author.title == "Henrique Guazzelli — CV"
    assert author.metadata["dossier"] == {"cover_letter": "I am applying for the AI/ML Engineer role.", "linkedin": LINKEDIN}
    assert author.metadata["location"] == "Porto Alegre, RS, BR" and author.metadata["consent_until"] == "2027-10-09"
    assert job.id == "J-061" and job.title == "AI/ML Engineer, Internal Platforms — micro1"
    assert job.metadata["client"] == "micro1" and job.metadata["occupation"] == "ai_engineering"
    assert job.metadata["seniority"] == "senior" and "LangGraph" in job.metadata["required_skills"]
    assert job.text.startswith("About\nmicro1 builds AI recruiters.")


def test_missing_author_files_fail_before_any_model_call(tmp_path):
    gateway = pool_gateway()
    directory = author_dir(tmp_path, skip=("cv.md", "linkedin.json"))

    with pytest.raises(FileNotFoundError, match="cv.md, linkedin.json"):
        build_pool(
            gateway, writer="fake/s2", extractor="fake/s1", personas=load_personas(PERSONAS),
            author_dir=directory, description="x", seed=7, today=TODAY,
        )
    assert gateway.calls == []


def test_replacing_the_pool_twice_keeps_one_copy_and_leaves_other_rows_untouched(tmp_path):
    corpus = tmp_path / "corpus"
    shutil.copytree(FIXTURE_CORPUS, corpus)
    before = (corpus / "candidates.jsonl").read_text(encoding="utf-8").splitlines()
    candidates, job = pool(tmp_path)

    replace_pool(corpus, candidates, job)
    replace_pool(corpus, candidates, job)

    after = (corpus / "candidates.jsonl").read_text(encoding="utf-8").splitlines()
    assert after[: len(before)] == before and len(after) == len(before) + 13
    jobs = [d for d in load_corpus(corpus) if d.collection == "jobs"]
    assert [d.id for d in jobs].count("J-061") == 1
