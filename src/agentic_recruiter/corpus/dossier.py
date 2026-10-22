"""Synthetic Dossier sections for demo Candidates: a cover letter and a LinkedIn profile consistent with the CV.

Run: uv run python -m agentic_recruiter.corpus.dossier C-0040 C-0150 --discrepancy C-0150
then re-index: docker compose --profile setup run --rm ingest
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from agentic_recruiter.corpus.documents import Document, write_jsonl
from agentic_recruiter.gateway import ModelGateway

DOSSIER_SCHEMA = {
    "type": "object",
    "properties": {
        "cover_letter": {"type": "string"},
        "linkedin": {
            "type": "object",
            "properties": {
                "headline": {"type": "string"},
                "about": {"type": "string"},
                "positions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"title": {"type": "string"}, "company": {"type": "string"}, "years": {"type": "string"}},
                        "required": ["title", "company", "years"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["headline", "about", "positions"],
            "additionalProperties": False,
        },
    },
    "required": ["cover_letter", "linkedin"],
    "additionalProperties": False,
}
INSTRUCTIONS = (
    "Write the cover letter (120 to 180 words) and the LinkedIn profile (headline, a 2 to 3 sentence about, and the "
    "positions list) this candidate would plausibly have. Every fact must be consistent with the CV. Use company "
    "names from the CV; where the CV says 'Company Name', invent a plausible name and use it consistently."
)
DISCREPANCY = (
    " Also include exactly two deliberate discrepancies with the CV in the LinkedIn positions, as real profiles "
    "sometimes have: a different employer name for one role, and a different number of years for another."
)


def generate(gateway: ModelGateway, model: str, candidate: Document, *, discrepancy: bool = False) -> dict:
    prompt = f"{INSTRUCTIONS}{DISCREPANCY if discrepancy else ''}\n\nCV:\n{candidate.text}"
    completion = gateway.complete(model, prompt, response_schema=("dossier", DOSSIER_SCHEMA))
    return json.loads(completion.text)


def add_dossiers(path: Path, ids: list[str], gateway: ModelGateway, model: str, *, discrepancy_ids=()) -> None:
    """Rewrite a candidates JSONL file with a Dossier in the metadata of each named Candidate."""
    rows = [Document(**json.loads(line)) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    wanted = set(ids)
    updated = [
        replace(row, metadata={**row.metadata, "dossier": generate(gateway, model, row, discrepancy=row.id in discrepancy_ids)})
        if row.id in wanted else row
        for row in rows
    ]
    write_jsonl(updated, Path(path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ids", nargs="+")
    parser.add_argument("--discrepancy", nargs="*", default=[])
    parser.add_argument("--path", type=Path, default=Path("corpus") / "candidates.jsonl")
    args = parser.parse_args()

    from agentic_recruiter.config import load_config
    from agentic_recruiter.openrouter import OpenRouterGateway

    add_dossiers(args.path, args.ids, OpenRouterGateway.from_env(), load_config().models.system_2_model, discrepancy_ids=args.discrepancy)


if __name__ == "__main__":
    main()
