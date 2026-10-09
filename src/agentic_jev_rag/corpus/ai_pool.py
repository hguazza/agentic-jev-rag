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
