"""Synthetic parts of the corpus: names, Consent dates, Clients and interview Outcomes (seeded)."""

import random
from datetime import date, timedelta

FIRST_NAMES = [
    "Ana", "Amara", "Bilal", "Carmen", "Chen", "Daniel", "Dmitri", "Elena", "Emeka", "Farah", "Grace", "Hiroshi",
    "Ines", "Isaac", "Jonas", "Kavya", "Leila", "Lucas", "Maya", "Mateo", "Nadia", "Noah", "Olu", "Priya",
    "Rafael", "Rosa", "Samir", "Sofia", "Tariq", "Thandiwe", "Tomás", "Uma", "Viktor", "Wen", "Yara", "Zoe",
]
LAST_NAMES = [
    "Adeyemi", "Almeida", "Andersen", "Bauer", "Chowdhury", "Costa", "Dubois", "Eriksen", "Fernández", "Garcia",
    "Haddad", "Ivanova", "Johnson", "Kim", "Kowalski", "Lindqvist", "Mensah", "Moreau", "Nakamura", "Novak",
    "O'Brien", "Okafor", "Patel", "Quinn", "Rahman", "Rossi", "Santos", "Schmidt", "Singh", "Tanaka",
    "Usman", "Van Dijk", "Walsh", "Wright", "Yilmaz", "Zhang",
]
CLIENTS = {
    "Northwind Logistics": "A European parcel and freight network modernising its operations with data and automation.",
    "Brightpath Health": "A digital health company building apps that help patients follow treatment plans.",
    "Cobalt Retail Group": "An omnichannel retailer with 300 stores and a fast-growing online business.",
    "Harbor & Finch Financial": "A mid-sized wealth and corporate finance advisory firm.",
    "Lumen Studio Collective": "A design and brand studio working with consumer and cultural brands.",
    "Ridgeway Manufacturing": "An industrial manufacturer of precision components for energy and transport.",
}
GOOD_OUTCOMES = ("advanced", "advanced", "offered", "hired")
BAD_OUTCOMES = ("rejected", "rejected", "rejected", "withdrew")
STAGE_FOR = {"advanced": "first interview", "offered": "final", "hired": "final", "rejected": "screening call", "withdrew": "first interview"}


def person_name(rng: random.Random) -> str:
    return f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"


def consent_until(rng: random.Random, today: date) -> str:
    """About 15% of Candidates have expired Consent."""
    if rng.random() < 0.15:
        return (today - timedelta(days=rng.randint(1, 240))).isoformat()
    return (today + timedelta(days=rng.randint(30, 720))).isoformat()


def skill_overlap(candidate_skills: list[str], required_skills: list[str]) -> float:
    a = {s.lower().strip() for s in candidate_skills}
    b = {s.lower().strip() for s in required_skills}
    return len(a & b) / len(a | b) if a | b else 0.0


def outcome(overlap: float, rng: random.Random) -> str:
    """Better skill overlap makes a good Outcome likelier, but not certain, so Match ground truth isn't circular."""
    return rng.choice(GOOD_OUTCOMES) if rng.random() < 0.25 + 0.6 * min(overlap * 2, 1.0) else rng.choice(BAD_OUTCOMES)


def scores_for(result: str, rng: random.Random) -> list[int]:
    low, high = (3, 5) if result in ("advanced", "offered", "hired") else (1, 3)
    return [rng.randint(low, high) for _ in range(4)]


def interview_date(rng: random.Random, today: date) -> str:
    return (today - timedelta(days=rng.randint(10, 300))).isoformat()
