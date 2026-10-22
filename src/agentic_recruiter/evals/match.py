"""Match evaluation: recall@k and MRR of Match against synthetic interview Outcomes.

A Candidate who was advanced, offered or hired for a Job (and still has valid Consent) counts as relevant.
Run: uv run python -m agentic_recruiter.evals.match
"""

import argparse
import json
from datetime import date, datetime
from pathlib import Path

from agentic_recruiter.corpus.documents import Document
from agentic_recruiter.evals.metrics import mrr, recall_at_k
from agentic_recruiter.rag.models import MatchRequest
from agentic_recruiter.rag.service import RagService

GOOD_OUTCOMES = {"advanced", "offered", "hired"}


def relevant_candidates(documents: list[Document], today: date) -> dict[str, set[str]]:
    by_id = {d.id: d for d in documents}
    relevant: dict[str, set[str]] = {}
    for note in (d for d in documents if d.collection == "interviews"):
        candidate = by_id.get(note.metadata["candidate_id"])
        if note.metadata["outcome"] in GOOD_OUTCOMES and candidate and candidate.consent_valid_on(today):
            relevant.setdefault(note.metadata["job_id"], set()).add(candidate.id)
    return relevant


def run_match_eval(service: RagService, relevant: dict[str, set[str]], top_k: int = 10) -> dict:
    rows = []
    for job_id, expected in sorted(relevant.items()):
        ranked = [c.candidate_id for c in service.match(MatchRequest(job_id=job_id, top_k=top_k)).candidates]
        rows.append({
            "job_id": job_id, "relevant": sorted(expected), "ranked": ranked,
            "recall@5": recall_at_k(ranked, expected, 5), "recall@10": recall_at_k(ranked, expected, 10),
            "mrr": mrr(ranked, expected),
        })
    n = len(rows) or 1
    return {
        "jobs": len(rows),
        "recall@5": sum(r["recall@5"] for r in rows) / n,
        "recall@10": sum(r["recall@10"] for r in rows) / n,
        "mrr": sum(r["mrr"] for r in rows) / n,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results") / f"match-{datetime.now():%Y%m%d-%H%M%S}")
    args = parser.parse_args()

    from qdrant_client import QdrantClient

    from agentic_recruiter.config import load_config
    from agentic_recruiter.corpus.documents import load_corpus
    from agentic_recruiter.openrouter import OpenRouterGateway

    config = load_config()
    documents = load_corpus(config.corpus_dir)
    qdrant = QdrantClient(url=config.qdrant_url, api_key=config.qdrant_api_key or None)
    service = RagService(config, OpenRouterGateway.from_env(), qdrant, documents)
    result = run_match_eval(service, relevant_candidates(documents, date.today()))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in result.pop("rows")))
    summary = (
        "# Match evaluation\n\n"
        f"Ground truth: Candidates advanced, offered or hired for a Job in the synthetic interview notes ({result['jobs']} Jobs).\n\n"
        "| recall@5 | recall@10 | MRR |\n|---|---|---|\n"
        f"| {result['recall@5']:.2f} | {result['recall@10']:.2f} | {result['mrr']:.2f} |\n"
    )
    (args.out / "summary.md").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
