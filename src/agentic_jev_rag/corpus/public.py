"""Public datasets (CC0): LiveCareer résumés via opensporks/resumes and EMSCAD job postings.

See docs/research/2026-10-08-recruiting-datasets.md for licences and provenance.
"""

import csv
import random
import re
import sys
import urllib.request
from pathlib import Path

RESUMES_URL = "https://huggingface.co/datasets/opensporks/resumes/resolve/main/Resume/Resume.csv"
JOBS_URL = "https://huggingface.co/datasets/victor/real-or-fake-fake-jobposting-prediction/resolve/main/fake_job_postings.csv"

RESUME_OCCUPATIONS = {
    "INFORMATION-TECHNOLOGY": "information_technology",
    "HR": "human_resources",
    "SALES": "sales",
    "DESIGNER": "design",
    "FINANCE": "finance",
    "ACCOUNTANT": "finance",
}
JOB_OCCUPATIONS = {
    "Information Technology": "information_technology",
    "Human Resources": "human_resources",
    "Sales": "sales",
    "Design": "design",
    "Finance": "finance",
    "Accounting/Auditing": "finance",
}
ENGLISH_SPEAKING = ("US", "GB", "CA", "IE", "AU", "NZ")

csv.field_size_limit(sys.maxsize)


def download(url: str, cache_dir: Path) -> Path:
    """Download once into `cache_dir`; later calls reuse the cached file."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / url.rsplit("/", 1)[-1]
    if not path.exists():
        urllib.request.urlretrieve(url, path)
    return path


def read_resumes(path: Path) -> list[dict]:
    """Résumés in the target occupations: {id, text, category, occupation}."""
    with open(path, encoding="utf-8", newline="") as f:
        return [
            {"id": row["ID"], "text": clean_resume_text(row["Resume_str"]), "category": row["Category"],
             "occupation": RESUME_OCCUPATIONS[row["Category"]]}
            for row in csv.DictReader(f)
            if row["Category"] in RESUME_OCCUPATIONS and len(row["Resume_str"].strip()) > 500
        ]


def read_jobs(path: Path) -> list[dict]:
    """Genuine, detailed, English-speaking-market postings in the target occupations."""
    with open(path, encoding="utf-8", newline="") as f:
        rows = []
        for row in csv.DictReader(f):
            occupation = job_occupation(row)
            if (
                occupation
                and row["fraudulent"] == "0"
                and row["requirements"].strip()
                and len(row["description"]) > 400
                and row["location"][:2] in ENGLISH_SPEAKING
            ):
                rows.append({**row, "occupation": occupation, "location": normalise_location(row["location"])})
        return rows


def job_occupation(row: dict) -> str | None:
    return JOB_OCCUPATIONS.get(row.get("function", ""))


def normalise_location(raw: str) -> str:
    """'US, NY, New York' -> 'New York, NY, US'."""
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return ", ".join(reversed(parts)) if parts else "Remote"


def clean_resume_text(text: str) -> str:
    """The CSV flattens résumés into runs of spaces; turn long runs back into line breaks."""
    lines = [re.sub(r"[ \t ]+", " ", line).strip() for line in re.split(r"[ \t ]{3,}|\n", text)]
    return "\n".join(line for line in lines if line)


def sample(rows: list[dict], per_occupation: int, rng: random.Random) -> list[dict]:
    """`per_occupation` rows from each occupation, in a seeded, stable order."""
    by_occupation: dict[str, list[dict]] = {}
    for row in rows:
        by_occupation.setdefault(row["occupation"], []).append(row)
    picked = []
    for occupation in sorted(by_occupation):
        group = by_occupation[occupation]
        picked += rng.sample(group, min(per_occupation, len(group)))
    return picked
