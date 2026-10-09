"""The AI engineer showcase pool: micro1's AI/ML Engineer role (J-061), 12 synthetic Candidates written from a
Persona grid (C-0201..C-0212), and the author (C-0213). Appended to the committed corpus; nothing else changes.

Run: uv run python -m agentic_jev_rag.corpus.ai_pool [--seed 7]
then re-index: docker compose --profile setup run --rm ingest
Re-running replaces the pool and drops the fakes' Dossiers (add them again with corpus.dossier). A corpus rebuild
(`corpus.build`) drops the pool: run this again afterwards.
"""

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
JOB_TITLE = "AI/ML Engineer, Internal Platforms"
REQUIRED_SKILLS = [
    "Python", "FastAPI", "LLMs", "LangGraph", "LangChain", "LlamaIndex", "RAG", "Vector databases",
    "Evaluation harnesses", "AWS",
]
AUTHOR_FILES = ("profile.toml", "cv.md", "cover_letter.md", "linkedin.json")


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
