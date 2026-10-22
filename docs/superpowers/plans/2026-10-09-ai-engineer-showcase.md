# AI Engineer Showcase Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The Recruiter ranks, screens and interviews 12 synthetic AI engineer Candidates and the author for micro1's AI/ML Engineer role (J-061), and a committed report shows how the author compares, graded by the same judge.

**Architecture:** `corpus/ai_pool.py` appends J-061, C-0201..C-0212 (from a Persona grid) and the author C-0213 to the committed corpus without touching the other documents. `showcase.py` runs Match and the Screener over the pool, interviews the advanced fakes through the Recruiter API in-process (the same endpoints the voice worker calls) with a Persona-driven Simulated candidate, imports the author's voice interview after checking it was graded under the same config, and writes `results/ai-engineer-showcase/`.

**Tech Stack:** Python 3.12, uv, FastAPI, LangGraph, Qdrant, OpenRouter via the Model gateway, pytest with `FakeModelGateway`.

**Spec:** `docs/superpowers/specs/2026-10-09-ai-engineer-showcase-design.md`

## Global Constraints

- Ids: fakes `C-0201`..`C-0212`, author `C-0213`, Job `J-061`. Occupation code `ai_engineering`; Scorecard heading `## AI engineering`.
- Tier counts: `strong` 3, `gaps` 4, `adjacent` 3, `bluffer` 2. Bluffers are C-0211 and C-0212.
- J-061's Client is `micro1`, the only real Client; every other Client stays fictional.
- Personas never enter the corpus: Qdrant stores each document's metadata in every chunk and Search returns it to the Screener. Personas live only in `corpus/ai_pool/personas.toml`.
- All 13 pool Candidates have `source: "ai_pool"` (the author too), so nothing in the metadata tells the Screener who wrote which CV.
- All 13 have valid Consent on the run date.
- `FakeModelGateway` is the only fake. Tests need Qdrant up (`docker compose up -d qdrant`). Live tests are `-m live`.
- Commits end with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.
- Part A (Tasks 1–6) builds on `main` now. Part B (Tasks 7–10) starts only after voice-interview slices 2–5 (Tasks 6–11 of `docs/superpowers/plans/2026-10-09-voice-interview.md`) are merged to `main` and this branch is rebased on it.

## Review Focus

1. **Match misses a pool Candidate.** Match reranks only the top `rerank_candidates` (30) chunks, so a Candidate can be absent. Expected: the runner still screens and lists them, with Match rank "not retrieved". Test: Task 5.
2. **Re-running the pool generator.** Expected: it replaces the pool (13 + 1 rows, never 26 + 2) and leaves every other row byte-identical. Test: Task 3.
3. **The author's files are missing or incomplete.** Expected: the generator fails before any model call and names the missing files. Test: Task 3.
4. **The Screener ends without a verdict** (step limit or no tool call). Expected: the row says "no verdict", the Candidate is not interviewed, and the report still renders. Test: Task 5.
5. **The voice results file is from an unfinished session** (`finished: false`, `assessment: null`, or `error` set). Expected: the import refuses with a message that says so, instead of a `KeyError`. Test: Task 9.

---

# Part A: buildable now from `main`

### Task 1: The AI engineering Scorecard

**Files:**
- Modify: `corpus/policies/scorecards.md` (insert before `## Rule for advancing`)
- Test: `tests/test_ai_pool.py` (create)

**Interfaces:**
- Produces: the `## AI engineering` section with four competencies, read by the Screener and (after the voice merge) `interviewer.competencies(scorecards, "ai_engineering")`.

- [ ] **Step 1: Write the failing test** (`tests/test_ai_pool.py`)

```python
"""The AI engineer showcase pool: Scorecard, Persona grid, generation and replacement."""

import re
from pathlib import Path

CORPUS = Path(__file__).parents[1] / "corpus"
AI_COMPETENCIES = ["LLM systems", "Retrieval and evaluation", "Production engineering", "Cloud delivery"]


def test_the_ai_engineering_scorecard_has_four_competencies():
    text = (CORPUS / "policies" / "scorecards.md").read_text(encoding="utf-8")
    section = text.split("## AI engineering\n", 1)[1].split("\n## ", 1)[0]

    assert re.findall(r"- \*\*(.+?):\*\*", section) == AI_COMPETENCIES
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_ai_pool.py -v`
Expected: FAIL with `IndexError: list index out of range`.

- [ ] **Step 3: Add the section** to `corpus/policies/scorecards.md`, directly before `## Rule for advancing`:

```markdown
## AI engineering

- **LLM systems:** agents, tool calling, prompt chains and orchestration (LangGraph, LangChain or LlamaIndex).
- **Retrieval and evaluation:** RAG architecture, vector databases and deterministic evaluation harnesses.
- **Production engineering:** Python and FastAPI services that are low-latency, tested and observable.
- **Cloud delivery:** deploys, monitors and scales on AWS; ships customer-facing features fast.

```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/test_ai_pool.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add corpus/policies/scorecards.md tests/test_ai_pool.py
git commit -m "AI engineering Scorecard for the showcase role

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Persona grid and fake Candidates

**Files:**
- Create: `corpus/ai_pool/personas.toml`
- Create: `src/agentic_jev_rag/corpus/ai_pool.py`
- Modify: `src/agentic_jev_rag/corpus/build.py` (rename `_profile` → `extract_profile`, two call sites)
- Test: `tests/test_ai_pool.py`

**Interfaces:**
- Consumes: `build.JOB_SCHEMA`, `build.extract_profile(gateway, model, text) -> dict` (`headline`, `years_experience`, `skills`), `synthetic.person_name(rng)`, `Document`.
- Produces in `corpus.ai_pool`: constants `OCCUPATION = "ai_engineering"`, `SOURCE = "ai_pool"`, `JOB_ID = "J-061"`, `AUTHOR_ID = "C-0213"`, `FAKE_IDS: list[str]` (C-0201..C-0212), `POOL_IDS = FAKE_IDS + [AUTHOR_ID]`, `TIERS = {"strong": 3, "gaps": 4, "adjacent": 3, "bluffer": 2}`, `POOL_DIR = Path("corpus/ai_pool")`; `@dataclass(frozen=True) Persona(id, tier, background, years_experience: int, location, speaking_style, honesty)`; `load_personas(path=POOL_DIR / "personas.toml") -> list[Persona]`; `fake_cv(gateway, model, persona) -> str`; `fake_candidate(gateway, *, writer, extractor, persona, name, consent_until) -> Document`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_ai_pool.py`)

```python
import json

import pytest

from agentic_jev_rag.corpus.ai_pool import FAKE_IDS, TIERS, Persona, fake_candidate, load_personas
from agentic_jev_rag.gateway import FakeModelGateway

PERSONAS = CORPUS / "ai_pool" / "personas.toml"
PROFILE = {"headline": "LLM platform engineer", "years_experience": 8, "skills": ["Python", "LangGraph", "FastAPI"]}


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
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_ai_pool.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agentic_jev_rag.corpus.ai_pool'`.

- [ ] **Step 3: Rename `_profile` to `extract_profile`** in `src/agentic_jev_rag/corpus/build.py`: the definition and its one call in `build_corpus` (`lambda r: extract_profile(gateway, extraction_model, r["text"])`). It is now shared with `ai_pool`.

- [ ] **Step 4: Write the Persona grid** (`corpus/ai_pool/personas.toml`)

```toml
# Personas for the AI engineer showcase (spec 2026-10-09-ai-engineer-showcase-design.md).
# Read by corpus.ai_pool (to write each CV) and the Simulated candidate (to play each interview).
# Never copied into the corpus: chunk payloads carry document metadata to the Screener.

[[persona]]
id = "C-0201"
tier = "strong"
background = "Senior LLM platform engineer at a fintech: LangGraph agents with tool calling in production, RAG over policy documents, an evaluation harness in CI, FastAPI services on AWS ECS"
years_experience = 8
location = "Austin, TX, US"
speaking_style = "concise and precise"
honesty = "truthful"

[[persona]]
id = "C-0202"
tier = "strong"
background = "Staff ML engineer who moved into LLM applications: built a retrieval and reranking platform on Qdrant, offline and online evaluation, Python services on AWS Lambda and EKS"
years_experience = 11
location = "Toronto, ON, CA"
speaking_style = "calm and structured, thinks aloud"
honesty = "truthful"

[[persona]]
id = "C-0203"
tier = "strong"
background = "AI engineer at a hiring-tech startup: voice and chat screening agents, LlamaIndex RAG, prompt optimisation with regression evals, FastAPI on AWS"
years_experience = 5
location = "Lisbon, LIS, PT"
speaking_style = "energetic and fast, occasionally rambling"
honesty = "truthful"

[[persona]]
id = "C-0204"
tier = "gaps"
background = "Senior Python backend engineer: FastAPI and AWS at scale, RAG only in side projects, no agents in production"
years_experience = 9
location = "Berlin, BE, DE"
speaking_style = "precise, a little terse"
honesty = "truthful"

[[persona]]
id = "C-0205"
tier = "gaps"
background = "ML researcher turned engineer: strong on evaluation and fine-tuning, little production backend or cloud ownership"
years_experience = 6
location = "Bangalore, KA, IN"
speaking_style = "academic, long answers"
honesty = "truthful"

[[persona]]
id = "C-0206"
tier = "gaps"
background = "Full-stack engineer who shipped a LangChain chatbot feature to customers; weak on evaluation and retrieval quality"
years_experience = 4
location = "São Paulo, SP, BR"
speaking_style = "friendly, many fillers"
honesty = "truthful"

[[persona]]
id = "C-0207"
tier = "gaps"
background = "Platform engineer on AWS who built an internal RAG search tool; light on agent design and prompt work"
years_experience = 7
location = "Dublin, D, IE"
speaking_style = "matter-of-fact"
honesty = "truthful"

[[persona]]
id = "C-0208"
tier = "adjacent"
background = "Data scientist: classical ML, forecasting, notebooks and dashboards; has called an LLM API in prototypes only"
years_experience = 6
location = "Chicago, IL, US"
speaking_style = "careful, hedges a lot"
honesty = "truthful"

[[persona]]
id = "C-0209"
tier = "adjacent"
background = "MLOps engineer: Kubernetes, model serving and feature stores for tabular models; no LLM applications"
years_experience = 8
location = "Warsaw, MZ, PL"
speaking_style = "nervous, self-correcting"
honesty = "truthful"

[[persona]]
id = "C-0210"
tier = "adjacent"
background = "Junior developer with a portfolio of LangChain tutorial projects; no production systems"
years_experience = 1
location = "Manila, NCR, PH"
speaking_style = "eager and rambling, many fillers"
honesty = "truthful"

[[persona]]
id = "C-0211"
tier = "bluffer"
background = "Claims to have led an autonomous agent platform at a startup; in truth integrated a vendor chatbot widget and wrote its prompts"
years_experience = 6
location = "London, LND, GB"
speaking_style = "confident, buzzword-heavy"
honesty = "bluffer"

[[persona]]
id = "C-0212"
tier = "bluffer"
background = "Claims senior RAG and evaluation expertise; in truth wrote prompt templates for a marketing content tool"
years_experience = 5
location = "Sydney, NSW, AU"
speaking_style = "smooth and salesy, deflects detail"
honesty = "bluffer"
```

- [ ] **Step 5: Implement `src/agentic_jev_rag/corpus/ai_pool.py`** (Personas and fake Candidates; Task 3 adds the rest)

```python
"""The AI engineer showcase pool: micro1's AI/ML Engineer role (J-061), 12 synthetic Candidates written from a
Persona grid (C-0201..C-0212), and the author (C-0213). Appended to the committed corpus; nothing else changes.

Run: uv run python -m agentic_jev_rag.corpus.ai_pool [--seed 7]
then re-index: docker compose --profile setup run --rm ingest
Re-running replaces the pool and drops the fakes' Dossiers (add them again with corpus.dossier). A corpus rebuild
(`corpus.build`) drops the pool: run this again afterwards.
"""

import tomllib
from dataclasses import dataclass
from pathlib import Path

from agentic_jev_rag.corpus.build import extract_profile
from agentic_jev_rag.corpus.documents import Document
from agentic_jev_rag.gateway import ModelGateway

OCCUPATION = "ai_engineering"
SOURCE = "ai_pool"  # the same for every pool Candidate, the author included, so the metadata does not say who is who
JOB_ID = "J-061"
AUTHOR_ID = "C-0213"
FAKE_IDS = [f"C-{n:04d}" for n in range(201, 213)]
POOL_IDS = FAKE_IDS + [AUTHOR_ID]
TIERS = {"strong": 3, "gaps": 4, "adjacent": 3, "bluffer": 2}
POOL_DIR = Path("corpus") / "ai_pool"
TIER_BRIEF = {
    "strong": "a strong fit: production LLM agents with tool calling, RAG and evaluation, Python and FastAPI, AWS.",
    "gaps": "a partial fit: clearly strong in some of the role's areas and clearly light in others.",
    "adjacent": "an adjacent profile: related engineering or data work, little or no production LLM experience.",
    "bluffer": (
        "a CV that reads as a strong fit, with impressive but vague claims: buzzwords and outcomes without specifics. "
        "The CV states only the claims, never the truth behind them given in the background."
    ),
}


@dataclass(frozen=True)
class Persona:
    """Who a fake Candidate is and how they interview. Kept out of the corpus (see personas.toml)."""

    id: str
    tier: str
    background: str
    years_experience: int
    location: str
    speaking_style: str
    honesty: str


def load_personas(path: Path = POOL_DIR / "personas.toml") -> list[Persona]:
    personas = [Persona(**entry) for entry in tomllib.loads(Path(path).read_text(encoding="utf-8"))["persona"]]
    counts = {tier: sum(p.tier == tier for p in personas) for tier in TIERS}
    if [p.id for p in personas] != FAKE_IDS or counts != TIERS:
        raise ValueError(f"personas must be {FAKE_IDS[0]}..{FAKE_IDS[-1]} in order with tiers {TIERS}; got {counts}")
    for persona in personas:
        if (persona.honesty == "bluffer") != (persona.tier == "bluffer"):
            raise ValueError(f"{persona.id}: honesty 'bluffer' goes with tier 'bluffer' only")
    return personas


def fake_cv(gateway: ModelGateway, model: str, persona: Persona) -> str:
    prompt = (
        "Write the plain-text CV of a fictional job seeker. Layout: a title line in capitals with their current role, "
        "then Summary, Highlights, Experience (each role: MM/YYYY to MM/YYYY, title, a fictional employer, city, and 2 "
        "to 4 lines of achievements), Education and Skills. Use fictional employers only. No personal name, contact "
        "details, age, gender, nationality or other personal characteristics.\n\n"
        f"Background: {persona.background}\nYears of experience: {persona.years_experience}\n"
        f"Location: {persona.location}\nFit for an AI/ML engineering role: {TIER_BRIEF[persona.tier]}"
    )
    return gateway.complete(model, prompt).text.strip()


def fake_candidate(
    gateway: ModelGateway, *, writer: str, extractor: str, persona: Persona, name: str, consent_until: str
) -> Document:
    text = fake_cv(gateway, writer, persona)
    profile = extract_profile(gateway, extractor, text)
    return Document(
        persona.id,
        "candidates",
        f"{name} — CV",
        text,
        {
            "name": name,
            "occupation": OCCUPATION,
            "headline": profile["headline"],
            "location": persona.location,
            "years_experience": persona.years_experience,
            "skills": profile["skills"][:12],
            "consent_until": consent_until,
            "source": SOURCE,
        },
    )
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_ai_pool.py tests/test_corpus_build.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add corpus/ai_pool/personas.toml src/agentic_jev_rag/corpus/ai_pool.py src/agentic_jev_rag/corpus/build.py tests/test_ai_pool.py
git commit -m "Persona grid and fake AI engineer Candidates for the showcase pool

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: J-061, the author, and replacing the pool in the corpus

**Files:**
- Modify: `src/agentic_jev_rag/corpus/ai_pool.py`
- Test: `tests/test_ai_pool.py`

**Interfaces:**
- Consumes: Task 2's module.
- Produces in `corpus.ai_pool`: `REQUIRED_SKILLS: list[str]`; `AUTHOR_FILES = ("profile.toml", "cv.md", "cover_letter.md", "linkedin.json")`; `micro1_job(gateway, model, description: str) -> Document`; `author_candidate(gateway, extractor, directory: Path, today: date) -> Document`; `build_pool(gateway, *, writer, extractor, personas, author_dir, description, seed, today) -> tuple[list[Document], Document]` (13 Candidates in id order, J-061); `replace_pool(corpus_dir: Path, candidates: list[Document], job: Document) -> None`; `main()`.
- The author's directory `corpus/ai_pool/henrique/` holds `profile.toml` (`name`, `location`), `cv.md`, `cover_letter.md`, and `linkedin.json` (`{"headline", "about", "positions": [{"title", "company", "years"}]}`, the voice spec's Dossier shape).

- [ ] **Step 1: Write the failing tests** (append to `tests/test_ai_pool.py`)

```python
import shutil
from datetime import date

from agentic_jev_rag.corpus.ai_pool import AUTHOR_ID, POOL_IDS, build_pool, replace_pool
from agentic_jev_rag.corpus.documents import Document, load_corpus

FIXTURE_CORPUS = Path(__file__).parent / "fixtures" / "corpus"
TODAY = date(2026, 10, 9)
LINKEDIN = {"headline": "AI Engineer", "about": "I build agents.", "positions": [{"title": "AI Engineer", "company": "Acme", "years": "2023-2026"}]}


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
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_ai_pool.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_pool'`.

- [ ] **Step 3: Implement** (add to `src/agentic_jev_rag/corpus/ai_pool.py`)

New imports at the top:

```python
import argparse
import json
import random
import tomllib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

from agentic_jev_rag.config import load_config
from agentic_jev_rag.corpus import synthetic
from agentic_jev_rag.corpus.build import JOB_SCHEMA, WORKERS, extract_profile
from agentic_jev_rag.corpus.documents import Document
from agentic_jev_rag.gateway import ModelGateway
```

New constants after `TIER_BRIEF`:

```python
JOB_TITLE = "AI/ML Engineer, Internal Platforms"
REQUIRED_SKILLS = [
    "Python", "FastAPI", "LLMs", "LangGraph", "LangChain", "LlamaIndex", "RAG", "Vector databases",
    "Evaluation harnesses", "AWS",
]
AUTHOR_FILES = ("profile.toml", "cv.md", "cover_letter.md", "linkedin.json")
```

Functions after `fake_candidate`:

```python
def micro1_job(gateway: ModelGateway, model: str, description: str) -> Document:
    prompt = (
        "Restructure this job description into the sections about, responsibilities, requirements and benefits. Keep "
        "micro1 as the company and keep every fact; add nothing. Remove any requirement based on a protected "
        f"characteristic.\n\n{description}"
    )
    job = json.loads(gateway.complete(model, prompt, response_schema=("job_posting", JOB_SCHEMA)).text)
    return Document(
        JOB_ID,
        "jobs",
        f"{JOB_TITLE} — micro1",
        (
            f"About\n{job['about']}\n\nResponsibilities\n{job['responsibilities']}\n\n"
            f"Requirements\n{job['requirements']}\n\nBenefits\n{job['benefits']}\n"
        ),
        {
            "client": "micro1",
            "occupation": OCCUPATION,
            "seniority": "senior",
            "location": "Remote",
            "required_skills": REQUIRED_SKILLS,
            "source": "docs/job-description.md",
        },
    )


def check_author_files(directory: Path) -> None:
    missing = [name for name in AUTHOR_FILES if not (Path(directory) / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{directory} is missing {', '.join(missing)}")


def author_candidate(gateway: ModelGateway, extractor: str, directory: Path, today: date) -> Document:
    """The author, from hand-written files: CV, cover letter and LinkedIn profile (the Dossier)."""
    check_author_files(directory)
    directory = Path(directory)
    profile = tomllib.loads((directory / "profile.toml").read_text(encoding="utf-8"))
    text = (directory / "cv.md").read_text(encoding="utf-8").strip()
    extracted = extract_profile(gateway, extractor, text)
    return Document(
        AUTHOR_ID,
        "candidates",
        f"{profile['name']} — CV",
        text,
        {
            "name": profile["name"],
            "occupation": OCCUPATION,
            "headline": extracted["headline"],
            "location": profile["location"],
            "years_experience": extracted["years_experience"],
            "skills": extracted["skills"][:12],
            "consent_until": (today + timedelta(days=365)).isoformat(),
            "source": SOURCE,
            "dossier": {
                "cover_letter": (directory / "cover_letter.md").read_text(encoding="utf-8").strip(),
                "linkedin": json.loads((directory / "linkedin.json").read_text(encoding="utf-8")),
            },
        },
    )


def build_pool(
    gateway: ModelGateway,
    *,
    writer: str,
    extractor: str,
    personas: list[Persona],
    author_dir: Path,
    description: str,
    seed: int,
    today: date,
) -> tuple[list[Document], Document]:
    """The 13 pool Candidates in id order, and J-061. Every Consent is valid for at least six months."""
    check_author_files(author_dir)  # before any model call
    rng = random.Random(seed)
    names: list[str] = []
    while len(names) < len(personas):
        if (name := synthetic.person_name(rng)) not in names:
            names.append(name)
    consents = [(today + timedelta(days=rng.randint(180, 720))).isoformat() for _ in personas]
    with ThreadPoolExecutor(WORKERS) as workers:
        fakes = list(workers.map(
            lambda args: fake_candidate(gateway, writer=writer, extractor=extractor, persona=args[0], name=args[1], consent_until=args[2]),
            zip(personas, names, consents),
        ))
    author = author_candidate(gateway, extractor, author_dir, today)
    return fakes + [author], micro1_job(gateway, writer, description)


def replace_pool(corpus_dir: Path, candidates: list[Document], job: Document) -> None:
    """Swap the pool's rows in candidates.jsonl and jobs.jsonl. Other lines are kept byte for byte."""
    for filename, new in (("candidates.jsonl", candidates), ("jobs.jsonl", [job])):
        path = Path(corpus_dir) / filename
        ids = {d.id for d in new}
        kept = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and json.loads(line)["id"] not in ids]
        added = [json.dumps(asdict(d), ensure_ascii=False) for d in new]  # the same serialisation as write_jsonl
        path.write_text("\n".join(kept + added) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--corpus", type=Path, default=Path("corpus"))
    parser.add_argument("--author", type=Path, default=POOL_DIR / "henrique")
    args = parser.parse_args()

    from agentic_jev_rag.openrouter import OpenRouterGateway

    config = load_config()
    candidates, job = build_pool(
        OpenRouterGateway.from_env(),
        writer=config.models.system_2_model,
        extractor=config.models.system_1_baseline,
        personas=load_personas(),
        author_dir=args.author,
        description=Path("docs/job-description.md").read_text(encoding="utf-8"),
        seed=args.seed,
        today=date.today(),
    )
    replace_pool(args.corpus, candidates, job)
    print(f"wrote {len(candidates)} Candidates and {job.id} to {args.corpus}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_ai_pool.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/agentic_jev_rag/corpus/ai_pool.py tests/test_ai_pool.py
git commit -m "Pool generator: micro1's J-061, the author's Candidate, and in-place pool replacement

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Generate the pool and index it (needs the author's files and OpenRouter)

**Files:**
- Create: `corpus/ai_pool/henrique/{profile.toml,cv.md,cover_letter.md,linkedin.json}` (from the author)
- Modify: `corpus/candidates.jsonl`, `corpus/jobs.jsonl` (generated)

- [ ] **Step 1: Get the author's files.** Ask your human partner for their CV, cover letter and LinkedIn profile in any format. Convert them: CV and cover letter to plain Markdown text; LinkedIn to `linkedin.json` in the shape `{"headline", "about", "positions": [{"title", "company", "years"}]}`; `profile.toml` with `name` and `location` (`City, Region, CC`). Show them the converted files and wait for their confirmation: this content becomes public.

- [ ] **Step 2: Generate**

Run: `uv run python -m agentic_jev_rag.corpus.ai_pool --seed 7`
Expected: `wrote 13 Candidates and J-061 to corpus`. `git diff --stat corpus/` shows 13 added lines in `candidates.jsonl` and 1 in `jobs.jsonl`, and nothing removed.

- [ ] **Step 3: Read the output.** Read three generated CVs (one strong, one adjacent, one bluffer) and J-061. Check: the strong CV is concrete; the adjacent one is plainly not an LLM engineer; the bluffer's CV sounds strong but vague and does not mention the "in truth" part of the Persona; no real employer names; J-061 names micro1 and adds no requirements beyond `docs/job-description.md`. If one is off, fix `TIER_BRIEF` or the prompt and regenerate. Do not edit the JSONL by hand.

- [ ] **Step 4: Index and run the suite**

Run: `docker compose up -d qdrant && docker compose --profile setup run --rm ingest && uv run pytest`
Expected: ingest completes without errors; `curl -s localhost:8000/documents/J-061` (after `docker compose up -d rag-api`) returns the micro1 Job; tests PASS.

- [ ] **Step 5: Commit**

```bash
git add corpus/ai_pool/henrique corpus/candidates.jsonl corpus/jobs.jsonl
git commit -m "Generate the AI engineer pool: J-061 (micro1), 12 synthetic Candidates and the author

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Showcase runner: Match, screening and the report

**Files:**
- Modify: `src/agentic_jev_rag/recruiter/rag_client.py` (`match` gains `filters`)
- Create: `src/agentic_jev_rag/showcase.py`
- Test: `tests/test_showcase.py`

**Interfaces:**
- Consumes: `RagClient.match(job_id, top_k=5, filters: dict | None = None) -> dict`; `screener.screen(ctx, candidate_id, job_id) -> (ScreeningVerdict | None, AgentResult)`; `ai_pool.load_personas`, `POOL_IDS`, `AUTHOR_ID`, `JOB_ID`, `OCCUPATION`.
- Produces in `showcase`: `@dataclass Row(candidate_id, name, tier, match_rank: int | None = None, match_score: float | None = None, verdict: ScreeningVerdict | None = None, assessment: InterviewAssessment | None = None)`; `@dataclass Showcase(job_id, author_id, system_1, rows: list[Row])`; `match_and_screen(ctx, job_id, candidate_ids, tiers: dict[str, str], *, occupation: str | None = None) -> list[Row]`; `average(scores) -> float | None`; `leaderboard(showcase) -> list[Row]`; `render_report(showcase) -> str`; `write_outputs(showcase, out_dir: Path) -> None`; `main()`.

- [ ] **Step 1: Write the failing tests** (`tests/test_showcase.py`)

```python
"""Showcase runner: Match, screening, interviews and the report, with the fake gateway and the fixture corpus."""

import json
import re

from agentic_jev_rag.gateway import ToolCall
from agentic_jev_rag.showcase import Showcase, leaderboard, match_and_screen, render_report, write_outputs
from tests.rag_scripts import scripted_gateway

ADVANCE = {
    "decision": "advance",
    "scores": [{"competency": "Technical depth", "score": 4, "evidence_chunk_ids": ["C-001#1"], "rationale": "Kafka."}],
    "summary": "Strong.",
}
REJECT = {
    "decision": "reject",
    "scores": [{"competency": "Technical depth", "score": 2, "evidence_chunk_ids": [], "rationale": "Thin."}],
    "summary": "Weak.",
}
TIERS = {"C-001": "strong", "C-002": "bluffer", "C-003": "adjacent", "C-005": "author"}


def screener(decisions):
    """Screener replies by Candidate: a verdict dict, or a string for a run that ends without one."""

    def respond(model, messages, tools):
        candidate = re.search(r"Candidate (C-\d+)", messages[1]["content"]).group(1)
        reply = decisions[candidate]
        return reply if isinstance(reply, str) else [ToolCall("scr", "submit_verdict", reply)]

    return respond


def test_every_candidate_is_screened_even_when_match_does_not_retrieve_them(recruiter_context):
    decisions = {"C-001": ADVANCE, "C-002": REJECT, "C-003": REJECT, "C-005": ADVANCE}
    ctx = recruiter_context(scripted_gateway(chat_responder=screener(decisions)))

    rows = match_and_screen(ctx, "J-001", list(TIERS), TIERS, occupation="software_engineering")

    by_id = {r.candidate_id: r for r in rows}
    assert list(by_id) == list(TIERS)
    assert by_id["C-005"].match_rank is None  # data_science: filtered out of Match, screened anyway
    assert sorted(r.match_rank for r in rows if r.match_rank) == list(range(1, 4))
    assert by_id["C-001"].verdict.decision == "advance" and by_id["C-005"].name == "Priya Raman"


def test_a_screening_without_a_verdict_is_reported_not_dropped(recruiter_context, tmp_path):
    decisions = {"C-001": ADVANCE, "C-002": "I could not decide.", "C-003": REJECT, "C-005": ADVANCE}
    ctx = recruiter_context(scripted_gateway(chat_responder=screener(decisions)))
    rows = match_and_screen(ctx, "J-001", list(TIERS), TIERS)
    showcase = Showcase(job_id="J-001", author_id="C-005", system_1="llm", rows=rows)

    report = render_report(showcase)
    write_outputs(showcase, tmp_path)

    line = next(l for l in report.splitlines() if "Tomás Verhoeven" in l)
    assert "no verdict" in line
    assert "| **Priya Raman** (author) |" in report
    assert json.loads((tmp_path / "run.json").read_text())["rows"][1]["verdict"] is None
    assert (tmp_path / "report.md").read_text() == report


def test_the_leaderboard_sorts_by_interview_then_screening_score(recruiter_context):
    decisions = {"C-001": REJECT, "C-002": ADVANCE, "C-003": "no verdict", "C-005": ADVANCE}
    ctx = recruiter_context(scripted_gateway(chat_responder=screener(decisions)))
    rows = match_and_screen(ctx, "J-001", list(TIERS), TIERS)

    ordered = [r.candidate_id for r in leaderboard(Showcase("J-001", "C-005", "llm", rows))]

    assert ordered[:2] in (["C-002", "C-005"], ["C-005", "C-002"]) and ordered[-1] == "C-003"
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_showcase.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agentic_jev_rag.showcase'`.

- [ ] **Step 3: Let Match take filters** (`src/agentic_jev_rag/recruiter/rag_client.py`)

```python
    def match(self, job_id: str, top_k: int = 5, filters: dict | None = None) -> dict:
        return self._post("/match", {"job_id": job_id, "top_k": top_k, "filters": filters or {}})
```

- [ ] **Step 4: Implement `src/agentic_jev_rag/showcase.py`**

```python
"""The AI engineer showcase: rank, screen and interview the J-061 pool and report how the author compares.

Run (RAG API up, pool indexed): uv run python -m agentic_jev_rag.showcase [--live-interview PATH] [--out DIR]
"""

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean

from agentic_jev_rag.corpus import ai_pool
from agentic_jev_rag.recruiter import screener
from agentic_jev_rag.recruiter.context import RecruiterContext
from agentic_jev_rag.recruiter.schemas import CompetencyScore, InterviewAssessment, ScreeningVerdict

OUT_DIR = Path("results") / "ai-engineer-showcase"


@dataclass
class Row:
    candidate_id: str
    name: str
    tier: str  # the Persona tier, or "author"
    match_rank: int | None = None  # None: Match did not retrieve this Candidate
    match_score: float | None = None
    verdict: ScreeningVerdict | None = None  # None: the Screener ended without a verdict
    assessment: InterviewAssessment | None = None


@dataclass
class Showcase:
    job_id: str
    author_id: str
    system_1: str
    rows: list[Row] = field(default_factory=list)


def match_and_screen(
    ctx: RecruiterContext, job_id: str, candidate_ids: list[str], tiers: dict[str, str], *, occupation: str | None = None
) -> list[Row]:
    """Rank the Candidates with Match, then screen every one of them, whether Match retrieved them or not."""
    filters = {"occupation": occupation} if occupation else None
    ranked = ctx.rag.match(job_id, top_k=len(candidate_ids), filters=filters)["candidates"]
    rank = {m["candidate_id"]: (n, m["score"]) for n, m in enumerate(ranked, start=1)}
    rows = []
    for candidate_id in candidate_ids:
        position, score = rank.get(candidate_id, (None, None))
        verdict, _ = screener.screen(ctx, candidate_id, job_id)
        name = ctx.rag.document(candidate_id)["metadata"].get("name", candidate_id)
        rows.append(Row(candidate_id, name, tiers[candidate_id], position, score, verdict))
    return rows


def average(scores: list[CompetencyScore] | None) -> float | None:
    return mean(s.score for s in scores) if scores else None


def leaderboard(showcase: Showcase) -> list[Row]:
    """Best first: Interview score, then Screening score. Missing scores sort last."""

    def key(row: Row):
        interview = average(row.assessment.scores) if row.assessment else None
        screening = average(row.verdict.scores) if row.verdict else None
        return (interview if interview is not None else -1.0, screening if screening is not None else -1.0)

    return sorted(showcase.rows, key=key, reverse=True)


def render_report(showcase: Showcase) -> str:
    lines = [
        f"# AI engineer showcase: {showcase.job_id}",
        "",
        f"System 1 decider: `{showcase.system_1}`. The author's row is in bold. Fit tiers come from the Persona grid "
        "(`corpus/ai_pool/personas.toml`); the Recruiter never saw them.",
        "",
        "| # | Candidate | Tier | Match rank | Screening | Interview | Recommendation |",
        "|---|---|---|---|---|---|---|",
    ]
    for n, row in enumerate(leaderboard(showcase), start=1):
        name = f"**{row.name}** (author)" if row.candidate_id == showcase.author_id else row.name
        match = str(row.match_rank) if row.match_rank else "not retrieved"
        screening = f"{row.verdict.decision} ({average(row.verdict.scores):.1f})" if row.verdict else "no verdict"
        interview = f"{average(row.assessment.scores):.1f}" if row.assessment else "not interviewed"
        recommendation = row.assessment.recommendation if row.assessment else "—"
        lines.append(f"| {n} | {name} | {row.tier} | {match} | {screening} | {interview} | {recommendation} |")
    return "\n".join(lines) + "\n"


def write_outputs(showcase: Showcase, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run = asdict(showcase)
    for row, data in zip(showcase.rows, run["rows"]):  # asdict leaves Pydantic models as objects
        data["verdict"] = row.verdict.model_dump() if row.verdict else None
        data["assessment"] = row.assessment.model_dump() if row.assessment else None
    (out_dir / "run.json").write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "report.md").write_text(render_report(showcase), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    from agentic_jev_rag.config import load_config
    from agentic_jev_rag.openrouter import OpenRouterGateway
    from agentic_jev_rag.recruiter.rag_client import RagClient

    config = load_config()
    ctx = RecruiterContext(config=config, gateway=OpenRouterGateway.from_env(), rag=RagClient(config.rag_url), system_1=config.system_1)
    tiers = {p.id: p.tier for p in ai_pool.load_personas()} | {ai_pool.AUTHOR_ID: "author"}
    rows = match_and_screen(ctx, ai_pool.JOB_ID, ai_pool.POOL_IDS, tiers, occupation=ai_pool.OCCUPATION)
    showcase = Showcase(ai_pool.JOB_ID, ai_pool.AUTHOR_ID, ctx.system_1, rows)
    write_outputs(showcase, args.out)
    print(render_report(showcase))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_showcase.py tests/test_agents.py tests/test_recruiter_api.py -v`
Expected: PASS. If the first test's Match ranks are not `1..3` because the fixture's fake reranker drops a software engineering Candidate, assert `rank is None or rank >= 1` for those three instead. The point of the test is C-005: screened despite no Match rank.

- [ ] **Step 6: Commit**

```bash
git add src/agentic_jev_rag/recruiter/rag_client.py src/agentic_jev_rag/showcase.py tests/test_showcase.py
git commit -m "Showcase runner: Match and screen the whole pool and write the leaderboard

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Docs and the first committed run (Match and screening)

**Files:**
- Create: `docs/adr/0014-ai-engineer-showcase.md`
- Modify: `CONTEXT.md`, `README.md`, `corpus/SOURCES.md`, `src/agentic_jev_rag/corpus/build.py` (`SOURCES` constant)
- Create: `results/ai-engineer-showcase/{run.json,report.md}` (generated)

- [ ] **Step 1: Write ADR 0014** (`docs/adr/0014-ai-engineer-showcase.md`)

```markdown
# 14. An AI engineer showcase for micro1's role

Date: 2026-10-09. Status: accepted.

## Context

The project was built to show fit for micro1's AI/ML Engineer, Internal Platforms role (`docs/job-description.md`), but its corpus had no AI engineering Candidates or Jobs, and the public resume dataset has none.

## Decision

- Add J-061, micro1's role, as the one real Client in the corpus. Every other Client stays fictional.
- Append 12 synthetic Candidates (C-0201..C-0212) written from a Persona grid (3 strong, 4 with gaps, 3 adjacent, 2 bluffers) and the author (C-0213) with a real name, CV and Dossier. The rest of the corpus is unchanged.
- Personas never enter the corpus, because chunk payloads carry document metadata to the Screener. Every pool Candidate has `source: ai_pool`, so the metadata does not say which CV is the author's.
- The Recruiter ranks and screens all 13. It interviews the fakes the Screener advances through the Recruiter API, with a Simulated candidate playing each Persona. The author holds the same interview by voice.
- Same judge: every Interview assessment records the System 2 model, the System 1 decider and a hash of the Interview Guide and Scorecards. The showcase refuses an author interview graded under a different one.

## Consequences

- The comparison is honest about its limits: simulated text against live speech, Personas written by the same model family that grades, and one run per Candidate. The report states them.
- `corpus.build` drops the pool. Run `corpus.ai_pool` again after a rebuild.
```

- [ ] **Step 2: Update `CONTEXT.md`.** Under **Client**, replace "Clients are fictional." with "Clients are fictional, except micro1 (J-061), the role this project targets (ADR 0014)." Add to the Corpus section:

```markdown
**Persona**:
Who a synthetic showcase Candidate is: fit tier, background, speaking style and honesty. Used to write their CV and to play them as a Simulated candidate; never stored in the corpus.

**Showcase**:
The AI engineer comparison for J-061: 12 Persona Candidates and the author, ranked, screened and interviewed by the Recruiter (`results/ai-engineer-showcase/`).
```

- [ ] **Step 3: Update the sources.** Append to `corpus/SOURCES.md` and to the `SOURCES` string in `build.py`, so a rebuild keeps the note:

```markdown
- **AI engineer showcase pool** (J-061, C-0201..C-0213): J-061 is micro1's public job description, restructured by
  the System 2 model. C-0201..C-0212 are synthetic CVs written by the System 2 model from `ai_pool/personas.toml`.
  C-0213 is the author's own CV. A rebuild drops the pool: run `uv run python -m agentic_jev_rag.corpus.ai_pool`.
```

- [ ] **Step 4: Run the showcase** (RAG API up: `docker compose up -d rag-api`)

Run: `uv run python -m agentic_jev_rag.showcase`
Expected: a 13-row leaderboard printed and written to `results/ai-engineer-showcase/`. Read the table. Strong-tier Candidates should mostly be advanced and adjacent ones rejected. If the tiers come out inverted, stop and report it before committing: it means the CVs or the Scorecard are wrong, not the report.

- [ ] **Step 5: Add "How I compare" to `README.md`**, after the section on Recruiter runs:

```markdown
## How I compare: the AI engineer showcase

The Recruiter screens me against 12 synthetic AI engineers for micro1's AI/ML Engineer role (J-061): 3 strong, 4 with gaps, 3 adjacent and 2 bluffers. The leaderboard and transcripts are in [`results/ai-engineer-showcase/report.md`](results/ai-engineer-showcase/report.md). The interview comparison is pending the voice interview. Rationale and limits: ADR 0014.

Regenerate the pool with `uv run python -m agentic_jev_rag.corpus.ai_pool`, re-index, then run `uv run python -m agentic_jev_rag.showcase`.
```

- [ ] **Step 6: Run the suite and commit**

Run: `uv run pytest`
Expected: PASS.

```bash
git add docs/adr/0014-ai-engineer-showcase.md CONTEXT.md README.md corpus/SOURCES.md src/agentic_jev_rag/corpus/build.py results/ai-engineer-showcase
git commit -m "Showcase docs (ADR 0014) and the first run: Match and screening for J-061

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

# Part B: after voice slices 2–5 are merged

Before Task 7: `git fetch && git rebase origin/main`, then `uv run pytest` passes. Confirm these voice-plan interfaces exist, and stop and report if any are missing: `interviewer.Briefing`, `interviewer.brief`, `interviewer.competencies`, `interviewer.dossier_text`, `corpus.dossier.add_dossiers`, `InterviewAssessment.red_flags`, `InterviewAssessment.probes_sustained`, `InterviewTurn.kind`, and `tests/interview_scripts.py` (`interview_gateway`, `start`, `reply`, `pending`).

### Task 7: Dossiers for the fakes, and `graded_under` on every assessment

**Files:**
- Modify: `src/agentic_jev_rag/recruiter/schemas.py`, `src/agentic_jev_rag/recruiter/interviewer.py`, `src/agentic_jev_rag/recruiter/interview_graph.py`
- Modify: `corpus/candidates.jsonl` (generated Dossiers)
- Test: `tests/test_interview_graph.py`, `tests/test_ai_pool.py`

**Interfaces:**
- Produces in `schemas`: `GradedUnder(BaseModel): system_2_model: str, system_1: str, policy_sha: str`; `InterviewAssessment.graded_under: GradedUnder | None = None`.
- Produces in `interviewer`: `policy_sha(guide: str, scorecards: str) -> str` (SHA-256 hex of `guide + "\n" + scorecards`); `Briefing.policy_sha: str`.
- Produces in the graph state: `policy_sha: str`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_interview_graph.py`:

```python
from agentic_jev_rag.recruiter.interviewer import policy_sha


def test_the_assessment_records_what_it_was_graded_under(recruiter_context):
    ctx = recruiter_context(interview_gateway(), system_1="jev")
    graph = build_interview_graph(ctx)
    start(graph, ctx)
    reply(graph, "I partitioned Kafka topics by order id.")
    done = reply(graph, "I owned on-call for the service.")

    graded = done["assessment"].graded_under
    guide = ctx.rag.document("policy:interview-guide")["text"]
    scorecards = ctx.rag.document("policy:scorecards")["text"]
    assert graded.system_2_model == "fake/system-2" and graded.system_1 == "jev"
    assert graded.policy_sha == policy_sha(guide, scorecards) and len(graded.policy_sha) == 64
```

Append to `tests/test_ai_pool.py`:

```python
def test_the_real_scorecard_parses_into_the_ai_engineering_competencies():
    from agentic_jev_rag.recruiter.interviewer import competencies

    text = (CORPUS / "policies" / "scorecards.md").read_text(encoding="utf-8")

    assert competencies(text, "ai_engineering") == tuple(AI_COMPETENCIES)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_interview_graph.py tests/test_ai_pool.py -v`
Expected: FAIL with `ImportError: cannot import name 'policy_sha'`. The Scorecard test should already PASS. If it fails, the heading-to-code mapping in `competencies` does not give `ai_engineering`: fix the heading, not the parser.

- [ ] **Step 3: Implement**

In `schemas.py`, before `InterviewAssessment`:

```python
class GradedUnder(BaseModel):
    """What an Interview assessment was graded under. Two assessments are comparable only when these match."""

    system_2_model: str = Field(description="The System 2 model that wrote the assessment.")
    system_1: str = Field(description="The System 1 decider: `llm` or `jev`.")
    policy_sha: str = Field(description="SHA-256 of the Interview Guide and Scorecards the Interviewer was briefed with.")
```

and in `InterviewAssessment`:

```python
    graded_under: GradedUnder | None = Field(default=None, description="The models and policies this assessment was graded under.")
```

In `interviewer.py`, add `import hashlib`, the field `policy_sha: str` to `Briefing` (last), and:

```python
def policy_sha(guide: str, scorecards: str) -> str:
    return hashlib.sha256(f"{guide}\n{scorecards}".encode()).hexdigest()
```

`brief` passes `policy_sha(guide, scorecards)` as the new last argument of `Briefing(...)`.

In `interview_graph.py`, add `policy_sha: str` to `InterviewGraphState`. The `brief` node also returns `"policy_sha": briefing.policy_sha`. In the `assess` node, add to the `model_copy` update:

```python
                "graded_under": GradedUnder(
                    system_2_model=ctx.config.models.system_2_model, system_1=ctx.system_1, policy_sha=state["policy_sha"]
                ),
```

and import `GradedUnder` from `schemas`.

- [ ] **Step 4: Run the suite**

Run: `uv run pytest`
Expected: PASS.

- [ ] **Step 5: Add Dossiers to the fakes** (OpenRouter)

Run: `uv run python -m agentic_jev_rag.corpus.dossier C-0201 C-0202 C-0203 C-0204 C-0205 C-0206 C-0207 C-0208 C-0209 C-0210 C-0211 C-0212 --discrepancy C-0211 C-0212`
Expected: `git diff corpus/candidates.jsonl` changes exactly those 12 lines, each gaining `metadata.dossier`. C-0213 keeps the author's own Dossier. Then re-index: `docker compose --profile setup run --rm ingest`.

- [ ] **Step 6: Commit**

```bash
git add src/agentic_jev_rag/recruiter tests/test_interview_graph.py tests/test_ai_pool.py corpus/candidates.jsonl
git commit -m "Record what each Interview assessment was graded under; Dossiers for the showcase fakes

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Simulated candidate with a Persona

**Files:**
- Modify: `src/agentic_jev_rag/config.py`, `configs/recruiter.toml`, `tests/conftest.py`
- Modify: `src/agentic_jev_rag/recruiter/prompts.py`, `src/agentic_jev_rag/recruiter/interviewer.py`
- Test: `tests/test_interview_graph.py`

**Interfaces:**
- Consumes: `ai_pool.Persona`; `interviewer.dossier_text(candidate)`; `interviewer._transcript(turns)`.
- Produces: `ModelConfig.simulated_candidate: str | None = None` (None falls back to the System 2 model); `prompts.simulated_persona(persona) -> str`; `interviewer.simulated_answer(ctx, candidate, question, *, persona: Persona | None = None, turns: list[InterviewTurn] = ()) -> str`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_interview_graph.py`)

```python
from agentic_jev_rag.corpus.ai_pool import Persona
from agentic_jev_rag.recruiter.schemas import InterviewTurn

BLUFFER = Persona("C-001", "bluffer", "Claims an agent platform; in truth a chatbot widget", 6, "London, LND, GB", "confident, buzzword-heavy", "bluffer")


def test_a_persona_simulated_candidate_speaks_in_character_with_the_interview_so_far(recruiter_context):
    gateway = interview_gateway()
    ctx = recruiter_context(gateway)
    candidate = ctx.rag.document("C-001")
    earlier = [InterviewTurn(question="What did you build?", answer="An agent platform, end to end.")]

    interviewer.simulated_answer(ctx, candidate, "How did the tool calls fail over?", persona=BLUFFER, turns=earlier)

    _, model, call = gateway.calls[-1]
    assert model == "fake/system-2"  # simulated_candidate unset: the System 2 model plays the Persona
    assert "confident, buzzword-heavy" in call["system"] and "vague" in call["system"]
    assert "An agent platform, end to end." in call["prompt"] and call["prompt"].startswith("CV:\n")


def test_without_a_persona_the_simulated_candidate_is_unchanged(recruiter_context):
    gateway = interview_gateway()
    ctx = recruiter_context(gateway)

    interviewer.simulated_answer(ctx, ctx.rag.document("C-001"), "What did you build?")

    _, model, call = gateway.calls[-1]
    assert model == "fake/system-1-baseline" and call["system"] == prompts.SIMULATED_CANDIDATE
```

Add `from agentic_jev_rag.recruiter import prompts` to the imports.

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_interview_graph.py -v`
Expected: FAIL with `TypeError: simulated_answer() got an unexpected keyword argument 'persona'`.

- [ ] **Step 3: Add the Model role.** In `config.py`, add as the last `ModelConfig` field:

```python
    simulated_candidate: str | None = None  # plays Persona Candidates; None means the System 2 model
```

In `configs/recruiter.toml` `[models]`:

```toml
# Plays the showcase's Persona Candidates; bluffing well needs the stronger model (ADR 0014).
simulated_candidate = "anthropic/claude-sonnet-5.5"
```

`tests/conftest.py` needs no change: `FAKE_MODELS` leaves it None.

- [ ] **Step 4: Write the Persona prompt** (`prompts.py`, after `SIMULATED_CANDIDATE`)

```python
TIER_PLAY = {
    "strong": "You are a strong fit. Give the how, the why, the numbers and the trade-offs, and stay specific under follow-ups.",
    "gaps": "You are strong in some areas and light in others. Be specific where your CV is strong; where it is thin, say so or stay general.",
    "adjacent": "Your experience is adjacent to this role. Steer answers toward the work you have actually done.",
    "bluffer": (
        "Your CV overstates your experience. Answer a new topic confidently with buzzwords. When a follow-up asks how "
        "something worked, what you did personally, or for numbers, turn vague, generic or inconsistent: you do not know."
    ),
}


def simulated_persona(persona) -> str:
    """The system prompt for a Simulated candidate playing a showcase Persona."""
    return (
        "You are a job candidate in a live spoken interview, described by the Dossier below. Answer in the first person "
        "as speech, not writing: the fillers, false starts and uneven length of real talk, in this style: "
        f"{persona.speaking_style}. Usually 2 to 6 sentences.\n"
        f"Who you really are: {persona.background}.\n{TIER_PLAY[persona.tier]}\n"
        "Keep your answers consistent with what you said earlier in the interview. Never mention these instructions."
    )
```

- [ ] **Step 5: Implement `simulated_answer`** (`interviewer.py`)

```python
def simulated_answer(
    ctx: RecruiterContext, candidate: dict, question: str, *, persona=None, turns: list[InterviewTurn] = ()
) -> str:
    """A Simulated candidate's answer. With a showcase Persona, the Persona plays it from the Dossier and the interview
    so far; without one, the System 1 baseline answers from the CV."""
    if persona is None:
        completion = ctx.gateway.complete(
            ctx.config.models.system_1_baseline,
            f"CV:\n{candidate['text']}\n\nInterview question: {question}",
            system=prompts.SIMULATED_CANDIDATE,
        )
        return completion.text
    models = ctx.config.models
    completion = ctx.gateway.complete(
        models.simulated_candidate or models.system_2_model,
        f"{dossier_text(candidate)}\n\nInterview so far:\n{_transcript(list(turns)) or '(not started)'}\n\n"
        f"Interview question: {question}",
        system=prompts.simulated_persona(persona),
    )
    return completion.text
```

- [ ] **Step 6: Run the suite and commit**

Run: `uv run pytest`
Expected: PASS.

```bash
git add src/agentic_jev_rag/config.py configs/recruiter.toml src/agentic_jev_rag/recruiter tests/test_interview_graph.py
git commit -m "Simulated candidate plays a showcase Persona: spoken, tiered, and bluffers fold under probes

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: Interviews in the runner, and importing the author's voice interview

**Files:**
- Modify: `src/agentic_jev_rag/showcase.py`
- Test: `tests/test_showcase.py`

**Interfaces:**
- Consumes: `create_recruiter_app(ctx)` (`POST /interviews`, `POST /interviews/{id}/reply`); `interviewer.simulated_answer(..., persona=, turns=)`; `interviewer.policy_sha`; `GradedUnder`; the voice results file `{"interview_id", "candidate_id", "job_id", "finished", "transcript", "assessment", "error"}`.
- Produces in `showcase`: `class ShowcaseError(Exception)`; `MAX_REPLIES = 40`; `expected_grading(ctx) -> GradedUnder`; `interview(api, ctx, candidate_id, job_id, persona) -> InterviewAssessment`; `interview_advanced(ctx, rows, job_id, personas: dict[str, Persona]) -> None` (fills `row.assessment` for advanced, non-author rows); `import_live(path: Path, ctx, author_id, job_id) -> InterviewAssessment`; `transcript_markdown(row) -> str`. `write_outputs` also writes `transcripts/<id>.md`. `main()` gains `--live-interview`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_showcase.py`)

```python
import pytest

from agentic_jev_rag.corpus.ai_pool import Persona
from agentic_jev_rag.showcase import ShowcaseError, expected_grading, import_live, interview_advanced
from tests.interview_scripts import ASSESSMENT, interview_gateway

PERSONAS = {
    cid: Persona(cid, tier, "background", 5, "Dublin, D, IE", "concise", "bluffer" if tier == "bluffer" else "truthful")
    for cid, tier in TIERS.items() if tier != "author"
}


def interviewing(decisions):
    return interview_gateway(chat_responder=screener(decisions))


def test_only_advanced_fakes_are_interviewed_through_the_recruiter_api(recruiter_context):
    decisions = {"C-001": ADVANCE, "C-002": REJECT, "C-003": ADVANCE, "C-005": ADVANCE}
    gateway = interviewing(decisions)
    ctx = recruiter_context(gateway)
    rows = match_and_screen(ctx, "J-001", list(TIERS), TIERS)

    interview_advanced(ctx, rows, "J-001", PERSONAS)

    by_id = {r.candidate_id: r for r in rows}
    assert by_id["C-001"].assessment.graded_under == expected_grading(ctx)
    assert by_id["C-003"].assessment is not None
    assert by_id["C-002"].assessment is None  # rejected by the Screener
    assert by_id["C-005"].assessment is None  # the author is interviewed live, not simulated
    assert all(t.answer for t in by_id["C-001"].assessment.turns)


def live_file(tmp_path, ctx, **overrides):
    assessment = {
        **ASSESSMENT, "candidate_id": "C-005", "job_id": "J-001", "turns": [{"question": "Q1", "answer": "A1"}],
        "graded_under": expected_grading(ctx).model_dump(),
    }
    result = {
        "interview_id": "abc", "candidate_id": "C-005", "job_id": "J-001", "finished": True,
        "transcript": [], "assessment": assessment, "error": None,
    } | overrides
    path = tmp_path / "abc.json"
    path.write_text(json.dumps(result))
    return path


def test_a_live_interview_graded_the_same_way_is_imported(recruiter_context, tmp_path):
    ctx = recruiter_context(interviewing({}))

    assessment = import_live(live_file(tmp_path, ctx), ctx, "C-005", "J-001")

    assert assessment.candidate_id == "C-005" and assessment.turns[0].answer == "A1"


def test_a_live_interview_graded_under_another_model_is_refused(recruiter_context, tmp_path):
    ctx = recruiter_context(interviewing({}))
    path = live_file(tmp_path, ctx)
    data = json.loads(path.read_text())
    data["assessment"]["graded_under"]["system_2_model"] = "other/model"
    path.write_text(json.dumps(data))

    with pytest.raises(ShowcaseError, match="system_2_model"):
        import_live(path, ctx, "C-005", "J-001")


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"finished": False}, "not finished"),
        ({"assessment": None}, "no assessment"),
        ({"error": "Recruiter API 502"}, "Recruiter API 502"),
        ({"candidate_id": "C-001"}, "C-001"),
    ],
)
def test_an_unusable_live_interview_is_refused_with_the_reason(recruiter_context, tmp_path, overrides, message):
    ctx = recruiter_context(interviewing({}))

    with pytest.raises(ShowcaseError, match=message):
        import_live(live_file(tmp_path, ctx, **overrides), ctx, "C-005", "J-001")


def test_transcripts_are_written_for_interviewed_candidates(recruiter_context, tmp_path):
    decisions = {"C-001": ADVANCE, "C-002": REJECT, "C-003": REJECT, "C-005": ADVANCE}
    ctx = recruiter_context(interviewing(decisions))
    rows = match_and_screen(ctx, "J-001", list(TIERS), TIERS)
    interview_advanced(ctx, rows, "J-001", PERSONAS)

    write_outputs(Showcase("J-001", "C-005", "llm", rows), tmp_path)

    assert [p.name for p in (tmp_path / "transcripts").iterdir()] == ["C-001.md"]
    assert "**Q1** (question" in (tmp_path / "transcripts" / "C-001.md").read_text()
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_showcase.py -v`
Expected: FAIL with `ImportError: cannot import name 'ShowcaseError'`.

- [ ] **Step 3: Implement** (add to `src/agentic_jev_rag/showcase.py`)

Imports:

```python
from fastapi.testclient import TestClient

from agentic_jev_rag.recruiter import interviewer, screener
from agentic_jev_rag.recruiter.api import create_recruiter_app
from agentic_jev_rag.recruiter.schemas import CompetencyScore, GradedUnder, InterviewAssessment, InterviewTurn, ScreeningVerdict
```

Code:

```python
MAX_REPLIES = 40  # 4 topics x (1 question + 2 probes + 2 reprompts) is 20; twice that means the graph is looping


class ShowcaseError(Exception):
    pass


def expected_grading(ctx: RecruiterContext) -> GradedUnder:
    guide = ctx.rag.document("policy:interview-guide")["text"]
    scorecards = ctx.rag.document("policy:scorecards")["text"]
    return GradedUnder(
        system_2_model=ctx.config.models.system_2_model, system_1=ctx.system_1,
        policy_sha=interviewer.policy_sha(guide, scorecards),
    )


def interview(api: TestClient, ctx: RecruiterContext, candidate_id: str, job_id: str, persona) -> InterviewAssessment:
    """One simulated interview through the Recruiter API: the same endpoints the voice worker calls."""
    candidate = ctx.rag.document(candidate_id)
    state = _ok(api.post("/interviews", json={"candidate_id": candidate_id, "job_id": job_id}))
    turns: list[InterviewTurn] = []
    for _ in range(MAX_REPLIES):
        if state["question"] is None:
            return InterviewAssessment.model_validate(state["assessment"])
        answer = interviewer.simulated_answer(ctx, candidate, state["question"], persona=persona, turns=turns)
        turns.append(InterviewTurn(question=state["question"], answer=answer))
        state = _ok(api.post(f"/interviews/{state['interview_id']}/reply", json={"message": answer}))
    raise ShowcaseError(f"{candidate_id}: the interview did not finish after {MAX_REPLIES} replies")


def interview_advanced(ctx: RecruiterContext, rows: list[Row], job_id: str, personas: dict) -> None:
    """Interview every fake the Screener advanced. The author is interviewed live, by voice."""
    api = TestClient(create_recruiter_app(ctx))
    for row in rows:
        if row.candidate_id in personas and row.verdict and row.verdict.decision == "advance":
            row.assessment = interview(api, ctx, row.candidate_id, job_id, personas[row.candidate_id])


def import_live(path: Path, ctx: RecruiterContext, author_id: str, job_id: str) -> InterviewAssessment:
    """The author's voice interview, only if it is finished and was graded under this run's config."""
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    if (result.get("candidate_id"), result.get("job_id")) != (author_id, job_id):
        raise ShowcaseError(f"{path} is for {result.get('candidate_id')} / {result.get('job_id')}, not {author_id} / {job_id}")
    if result.get("error"):
        raise ShowcaseError(f"{path} ended with an error: {result['error']}")
    if not result.get("finished"):
        raise ShowcaseError(f"{path}: the interview is not finished")
    if not result.get("assessment"):
        raise ShowcaseError(f"{path} has no assessment")
    assessment = InterviewAssessment.model_validate(result["assessment"])
    expected = expected_grading(ctx)
    graded = assessment.graded_under
    if graded is None:
        raise ShowcaseError(f"{path}: the assessment does not say what it was graded under; re-run the interview")
    differing = [name for name in GradedUnder.model_fields if getattr(graded, name) != getattr(expected, name)]
    if differing:
        details = "; ".join(f"{n}: interview {getattr(graded, n)!r}, run {getattr(expected, n)!r}" for n in differing)
        raise ShowcaseError(f"{path} was graded differently from this run ({details})")
    return assessment


def transcript_markdown(row: Row) -> str:
    lines = [f"# {row.name} ({row.candidate_id}), {row.tier}", ""]
    for n, turn in enumerate(row.assessment.turns, start=1):
        mark = "" if turn.scored else ", not scored"
        lines += [f"**Q{n}** ({turn.kind}{mark}): {turn.question}", "", f"> {turn.answer}", ""]
    if row.assessment.red_flags:
        lines += ["## Red flags", ""] + [f"- {f.kind}: \"{f.quote}\" ({f.detail})" for f in row.assessment.red_flags] + [""]
    lines += ["## Assessment", ""] + [f"- {s.competency}: {s.score}. {s.rationale}" for s in row.assessment.scores]
    lines += ["", f"Recommendation: {row.assessment.recommendation}. {row.assessment.summary}"]
    return "\n".join(lines) + "\n"


def _ok(response) -> dict:
    if response.status_code >= 400:
        raise ShowcaseError(f"Recruiter API {response.request.url.path} → {response.status_code}: {response.text[:300]}")
    return response.json()
```

In `write_outputs`, after the report:

```python
    transcripts = out_dir / "transcripts"
    transcripts.mkdir(exist_ok=True)
    for row in showcase.rows:
        if row.assessment:
            (transcripts / f"{row.candidate_id}.md").write_text(transcript_markdown(row), encoding="utf-8")
```

In `render_report`, add two columns, "Probes sustained" (`sum(row.assessment.probes_sustained.values())`, or `—`) and "Red flags" (`len(row.assessment.red_flags)`, or `—`), plus a transcript link in the Interview cell (`[3.5](transcripts/C-0201.md)`). Update the header and separator rows to match.

In `main`, add `--live-interview` (`type=Path`, default None). After `match_and_screen`, call `interview_advanced(ctx, rows, ai_pool.JOB_ID, {p.id: p for p in ai_pool.load_personas()})`. If `--live-interview` was given, set the author row's `assessment = import_live(args.live_interview, ctx, ai_pool.AUTHOR_ID, ai_pool.JOB_ID)`. Do this import first, before any screening, so a mismatched file fails before the run spends money. Without the flag, the report gains the line "The author's voice interview is pending."

- [ ] **Step 4: Run the suite**

Run: `uv run pytest`
Expected: PASS. If `TestClient(create_recruiter_app(ctx))` fails because the app's lifespan needs something the tests' `RecruiterContext` lacks, report it. Do not swap in direct graph calls: the point is that fakes and the author go through the same endpoints.

- [ ] **Step 5: Commit**

```bash
git add src/agentic_jev_rag/showcase.py tests/test_showcase.py
git commit -m "Showcase interviews advanced fakes through the Recruiter API and imports the author's voice interview under the same judge

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: Where the author won and lost points, the live run, and the final report

**Files:**
- Modify: `src/agentic_jev_rag/showcase.py`
- Modify: `tests/test_showcase.py`, `tests/test_live.py`
- Modify: `README.md`
- Create/replace: `results/ai-engineer-showcase/` (generated)

**Interfaces:**
- Produces in `showcase`: `comparison(ctx, showcase) -> str | None`, Markdown written by the System 2 model, or None when the author or every strong-tier Candidate lacks an assessment. `render_report(showcase, analysis: str | None = None)` appends it under `## Where I won and lost points`, followed by a fixed `## Limits` section.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_showcase.py`)

```python
from agentic_jev_rag.showcase import comparison


def test_the_comparison_quotes_the_author_and_the_strong_tier(recruiter_context):
    decisions = {"C-001": ADVANCE, "C-002": REJECT, "C-003": REJECT, "C-005": ADVANCE}
    gateway = interviewing(decisions)
    ctx = recruiter_context(gateway)
    rows = match_and_screen(ctx, "J-001", list(TIERS), TIERS)
    interview_advanced(ctx, rows, "J-001", PERSONAS)
    author = next(r for r in rows if r.candidate_id == "C-005")
    author.assessment = rows[0].assessment.model_copy(update={"candidate_id": "C-005"})
    showcase = Showcase("J-001", "C-005", "llm", rows)

    analysis = comparison(ctx, showcase)
    report = render_report(showcase, analysis)

    prompt = gateway.calls[-1][2]["prompt"]
    assert "Author (C-005" in prompt and "Strong tier: Ana Ribeiro (C-001)" in prompt
    assert "## Where I won and lost points" in report and "## Limits" in report


def test_no_comparison_without_the_authors_interview(recruiter_context):
    ctx = recruiter_context(interviewing({}))
    rows = [Row("C-001", "Ana Ribeiro", "strong"), Row("C-005", "Priya Raman", "author")]

    assert comparison(ctx, Showcase("J-001", "C-005", "llm", rows)) is None
```

Add `Row` to the import from `agentic_jev_rag.showcase`.

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_showcase.py -v`
Expected: FAIL with `ImportError: cannot import name 'comparison'`.

- [ ] **Step 3: Implement** (`showcase.py`)

```python
LIMITS = """## Limits

- The fakes are Simulated candidates answering in text; the author answered live, by voice, through speech-to-text.
- The Personas and their answers were written by the same model family that grades them.
- Integrity signals (timing, fluency) exist only for the author's voice interview and are advisory: they never enter scores.
- One run per Candidate: model sampling alone can move a score by a point.
"""


def comparison(ctx: RecruiterContext, showcase: Showcase) -> str | None:
    """Where the author won and lost points against the strong tier, per competency, citing transcript turns."""
    author = next((r for r in showcase.rows if r.candidate_id == showcase.author_id and r.assessment), None)
    strong = [r for r in showcase.rows if r.tier == "strong" and r.assessment]
    if author is None or not strong:
        return None
    author_block = f"Author ({author.candidate_id}, {author.name})\n{transcript_markdown(author)}"
    strong_blocks = [f"Strong tier: {r.name} ({r.candidate_id})\n{transcript_markdown(r)}" for r in strong]
    prompt = (
        "Compare the author's interview with the strong-tier candidates', competency by competency. For each "
        "competency, say where the author scored higher or lower and why, quoting both sides and citing each quote as "
        "(C-xxxx Qn). Then give three concrete things the author should do better in a real interview. Use only what "
        "the transcripts and assessments say. Markdown, under 400 words, no top-level heading.\n\n"
        + "\n\n".join([author_block, *strong_blocks])
    )
    return ctx.gateway.complete(ctx.config.models.system_2_model, prompt).text.strip()
```

`render_report(showcase, analysis=None)`: after the table, append `\n## Where I won and lost points\n\n{analysis}\n` when `analysis` is set, then `\n` + `LIMITS` always. `write_outputs(showcase, out_dir, analysis=None)` passes it through, and `main` computes `analysis = comparison(ctx, showcase)` before writing.

- [ ] **Step 4: Add the live test** (append to `tests/test_live.py`)

```python
def test_the_showcase_runs_end_to_end_without_the_live_interview(tmp_path, gateway):
    from agentic_jev_rag.corpus import ai_pool
    from agentic_jev_rag.showcase import Showcase, interview_advanced, match_and_screen, write_outputs

    ctx = RecruiterContext(config=CONFIG, gateway=gateway, rag=RagClient(CONFIG.rag_url), system_1=CONFIG.system_1)
    personas = {p.id: p for p in ai_pool.load_personas()}
    ids = ["C-0201", "C-0210", ai_pool.AUTHOR_ID]  # one strong, one adjacent, the author: keeps the cost near $1
    tiers = {cid: personas[cid].tier if cid in personas else "author" for cid in ids}

    rows = match_and_screen(ctx, ai_pool.JOB_ID, ids, tiers, occupation=ai_pool.OCCUPATION)
    interview_advanced(ctx, rows, ai_pool.JOB_ID, personas)
    write_outputs(Showcase(ai_pool.JOB_ID, ai_pool.AUTHOR_ID, ctx.system_1, rows), tmp_path)

    assert all(r.verdict for r in rows)
    assert "| 1 |" in (tmp_path / "report.md").read_text()
```

Run: `uv run pytest -m live tests/test_live.py::test_the_showcase_runs_end_to_end_without_the_live_interview -v` (RAG API up and the pool indexed).
Expected: PASS.

- [ ] **Step 5: Run the suite and commit the code**

Run: `uv run pytest`
Expected: PASS.

```bash
git add src/agentic_jev_rag/showcase.py tests/test_showcase.py tests/test_live.py
git commit -m "Showcase report: where the author won and lost points against the strong tier, and the limits

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 6: The author's voice interview (human).** Ask your human partner to hold the voice interview for C-0213 and J-061. They set the Playground participant attributes `candidate_id=C-0213` and `job_id=J-061` (`docs/voice-demo.md`), with the same `configs/recruiter.toml` and `SYSTEM_1` the showcase will use. Wait for the file in `results/voice-interviews/<id>.json`.

- [ ] **Step 7: The final run**

Run: `uv run python -m agentic_jev_rag.showcase --live-interview results/voice-interviews/<id>.json`
Expected: 13 rows, transcripts for every interviewed Candidate, and the comparison section. Read the comparison and check that every quote it cites exists in the named transcript turn. If one does not, re-run the comparison rather than editing it by hand.

- [ ] **Step 8: Update the README.** In "How I compare", replace "The interview comparison is pending the voice interview." with one or two sentences on where the author placed (rank, recommendation) and the largest gap from the comparison, linking the report.

- [ ] **Step 9: Commit the results**

```bash
git add results/ai-engineer-showcase results/voice-interviews/<id>.json README.md
git commit -m "AI engineer showcase: the full run with the author's voice interview

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```
