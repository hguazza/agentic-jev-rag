"""Build the committed corpus: sample public rows, enrich them, and add synthetic interview notes.

Run: uv run python -m agentic_recruiter.corpus.build [--seed 7] [--out corpus]
"""

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

from agentic_recruiter.config import load_config
from agentic_recruiter.corpus import public, synthetic
from agentic_recruiter.corpus.documents import Document, write_jsonl
from agentic_recruiter.gateway import ModelGateway

WORKERS = 8
CANDIDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string"},
        "years_experience": {"type": "number"},
        "skills": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["headline", "years_experience", "skills"],
    "additionalProperties": False,
}
JOB_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "about": {"type": "string"},
        "responsibilities": {"type": "string"},
        "requirements": {"type": "string"},
        "benefits": {"type": "string"},
        "required_skills": {"type": "array", "items": {"type": "string"}},
        "seniority": {"type": "string", "enum": ["entry", "mid", "senior", "lead"]},
    },
    "required": ["title", "about", "responsibilities", "requirements", "benefits", "required_skills", "seniority"],
    "additionalProperties": False,
}


def build_corpus(
    resume_rows: list[dict],
    job_rows: list[dict],
    gateway: ModelGateway,
    *,
    system_2_model: str,
    extraction_model: str,
    seed: int,
    today: date,
    resumes_per_occupation: int = 40,
    jobs_per_occupation: int = 12,
    notes_per_job: int = 2,
) -> list[Document]:
    rng = random.Random(seed)
    resumes = public.sample(resume_rows, resumes_per_occupation, rng)
    postings = public.sample(job_rows, jobs_per_occupation, rng)
    clients = [rng.choice(list(synthetic.CLIENTS)) for _ in postings]
    locations = sorted({p["location"] for p in postings})

    with ThreadPoolExecutor(WORKERS) as pool:
        profiles = list(pool.map(lambda r: extract_profile(gateway, extraction_model, r["text"]), resumes))
        rewritten = list(pool.map(lambda pc: _rewrite_job(gateway, system_2_model, *pc), zip(postings, clients)))

    candidates = [
        Document(
            id=f"C-{n:04d}",
            collection="candidates",
            title=f"{name} — CV",
            text=row["text"],
            metadata={
                "name": name,
                "occupation": row["occupation"],
                "headline": profile["headline"],
                "location": rng.choice(locations),
                "years_experience": profile["years_experience"],
                "skills": profile["skills"][:12],
                "consent_until": synthetic.consent_until(rng, today),
                "source": f"opensporks/resumes:{row['id']}",
            },
        )
        for n, (row, profile, name) in enumerate(
            zip(resumes, profiles, [synthetic.person_name(rng) for _ in resumes]), start=1
        )
    ]
    jobs = [
        Document(
            id=f"J-{n:03d}",
            collection="jobs",
            title=f"{job['title']} — {client}",
            text=(
                f"About\n{job['about']}\n\nResponsibilities\n{job['responsibilities']}\n\n"
                f"Requirements\n{job['requirements']}\n\nBenefits\n{job['benefits']}\n"
            ),
            metadata={
                "client": client,
                "occupation": posting["occupation"],
                "seniority": job["seniority"],
                "location": posting["location"],
                "required_skills": job["required_skills"][:10],
                "source": f"EMSCAD:{posting['job_id']}",
            },
        )
        for n, (posting, job, client) in enumerate(zip(postings, rewritten, clients), start=1)
    ]

    plans = _interview_plans(candidates, jobs, notes_per_job, rng, today)
    with ThreadPoolExecutor(WORKERS) as pool:
        texts = list(pool.map(lambda plan: _note(gateway, system_2_model, *plan[:2], plan[2]), plans))
    notes = [
        Document(
            id=f"I-{n:04d}",
            collection="interviews",
            title=f"Interview note I-{n:04d}: {candidate.metadata['name']} for {job.title} ({plan['stage']})",
            text=text,
            metadata={
                "candidate_id": candidate.id, "job_id": job.id, "stage": plan["stage"], "outcome": plan["outcome"],
                "date": plan["date"], "occupation_match": candidate.metadata["occupation"] == job.metadata["occupation"],
            },
        )
        for n, ((candidate, job, plan), text) in enumerate(zip(plans, texts), start=1)
    ]
    return candidates + jobs + notes


def extract_profile(gateway: ModelGateway, model: str, text: str) -> dict:
    prompt = (
        "Extract from this CV: a short professional headline, total years of professional experience (estimate "
        "from dates; a number), and up to 12 job-related skills. Never extract personal characteristics.\n\n"
        f"CV:\n{text[:8000]}"
    )
    return json.loads(gateway.complete(model, prompt, response_schema=("candidate_profile", CANDIDATE_SCHEMA)).text)


def _rewrite_job(gateway: ModelGateway, model: str, posting: dict, client: str) -> dict:
    prompt = (
        f"Rewrite this public job posting as a posting by the fictional company {client} "
        f"({synthetic.CLIENTS[client]}). Remove the original company's name, products, URLs and anything that "
        "identifies it. Keep the role, its real responsibilities and requirements. Remove any requirement based on "
        "a protected characteristic. Give up to 10 required skills and the seniority.\n\n"
        f"Title: {posting['title']}\nExperience level: {posting.get('required_experience', '')}\n"
        f"Description: {posting['description'][:4000]}\nRequirements: {posting['requirements'][:3000]}\n"
        f"Benefits: {posting.get('benefits', '')[:1500]}"
    )
    return json.loads(gateway.complete(model, prompt, response_schema=("job_posting", JOB_SCHEMA)).text)


def _interview_plans(candidates, jobs, per_job, rng, today) -> list[tuple[Document, Document, dict]]:
    plans = []
    for job in jobs:
        pool = [c for c in candidates if c.metadata["occupation"] == job.metadata["occupation"]]
        if not pool:
            continue
        ranked = sorted(pool, key=lambda c: synthetic.skill_overlap(c.metadata["skills"], job.metadata["required_skills"]), reverse=True)
        picked = [ranked[0]] + rng.sample(ranked[1:], min(per_job - 1, len(ranked) - 1))
        for candidate in picked[:per_job]:
            overlap = synthetic.skill_overlap(candidate.metadata["skills"], job.metadata["required_skills"])
            result = synthetic.outcome(overlap, rng)
            plans.append((candidate, job, {
                "outcome": result, "stage": synthetic.STAGE_FOR[result],
                "scores": synthetic.scores_for(result, rng), "date": synthetic.interview_date(rng, today),
            }))
    return plans


def _note(gateway: ModelGateway, model: str, candidate: Document, job: Document, plan: dict) -> str:
    prompt = (
        f"Write a recruiter's interview note (120 to 180 words) for a {plan['stage']} with {candidate.metadata['name']} "
        f"for the job '{job.title}'. Describe what the candidate said about relevant past work, consistent with "
        f"their CV, and record these Scorecard scores (1 to 5) for the four competencies of the "
        f"{job.metadata['occupation'].replace('_', ' ')} Scorecard: {plan['scores']}. End with 'Outcome: "
        f"{plan['outcome']}.' Never mention age, gender, family, nationality, religion, health or other protected "
        f"characteristics.\n\nJob requirements:\n{job.text[:1500]}\n\nCV excerpt:\n{candidate.text[:2500]}"
    )
    return gateway.complete(model, prompt).text.strip()


SOURCES = """# Corpus sources

- **candidates.jsonl**: résumé text from `opensporks/resumes` (Hugging Face, CC0-1.0), a mirror of the Kaggle
  LiveCareer "Resume Dataset" (CC0). Names, locations and Consent dates are synthetic. Headlines, years of
  experience and skills were extracted by the System 1 baseline.
- **jobs.jsonl**: postings from EMSCAD via `victor/real-or-fake-fake-jobposting-prediction` (Hugging Face, CC0-1.0),
  rewritten by the System 2 model for fictional Clients, with identifying details removed.
- **interviews.jsonl**: synthetic interview notes written by the System 2 model for seeded (Candidate, Job) pairs
  and Outcomes.
- **policies/**: hand-written policies for the fictional agency.
- **AI engineer showcase pool** (J-061, C-0201..C-0213): J-061 is micro1's public job description, restructured
  into the Job sections. C-0201..C-0212 are synthetic CVs and Dossiers written from `ai_pool/personas.toml` (the committed
  pool was written in Claude Code; re-running the generator has the System 2 model write them). C-0213 is the author's
  own CV. A rebuild drops the pool: run `uv run python -m agentic_recruiter.corpus.ai_pool`.

Rebuild with `uv run python -m agentic_recruiter.corpus.build --seed {seed}` (downloads are cached in `data/raw/`).
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=Path("corpus"))
    parser.add_argument("--cache", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    from agentic_recruiter.openrouter import OpenRouterGateway

    config = load_config()
    documents = build_corpus(
        public.read_resumes(public.download(public.RESUMES_URL, args.cache)),
        public.read_jobs(public.download(public.JOBS_URL, args.cache)),
        OpenRouterGateway.from_env(),
        system_2_model=config.models.system_2_model,
        extraction_model=config.models.system_1_baseline,
        seed=args.seed,
        today=date.today(),
    )
    for collection in ("candidates", "jobs", "interviews"):
        write_jsonl([d for d in documents if d.collection == collection], args.out / f"{collection}.jsonl")
    (args.out / "SOURCES.md").write_text(SOURCES.format(seed=args.seed), encoding="utf-8")
    print({c: sum(d.collection == c for d in documents) for c in ("candidates", "jobs", "interviews")})


if __name__ == "__main__":
    main()
