"""The AI engineer showcase: rank, screen and interview the J-061 pool and report how the author compares.

Run (RAG API up, pool indexed): uv run python -m agentic_recruiter.showcase [--live-interview PATH] [--out DIR]
"""

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean

from agentic_recruiter.corpus import ai_pool
from agentic_recruiter.recruiter import screener
from agentic_recruiter.recruiter.context import RecruiterContext
from agentic_recruiter.recruiter.schemas import CompetencyScore, InterviewAssessment, ScreeningVerdict

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


def render_report(showcase: Showcase, *, coaching: bool = False) -> str:
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
    if coaching:
        lines += ["", "The author's interview coaching: [coaching.md](coaching.md)"]
    return "\n".join(lines) + "\n"


def write_outputs(showcase: Showcase, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run = asdict(showcase)
    for row, data in zip(showcase.rows, run["rows"]):  # asdict leaves Pydantic models as objects
        data["verdict"] = row.verdict.model_dump() if row.verdict else None
        data["assessment"] = row.assessment.model_dump() if row.assessment else None
    (out_dir / "run.json").write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")
    report = render_report(showcase, coaching=(out_dir / "coaching.md").exists())
    (out_dir / "report.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    from agentic_recruiter.config import load_config
    from agentic_recruiter.openrouter import OpenRouterGateway
    from agentic_recruiter.recruiter.rag_client import RagClient

    config = load_config()
    ctx = RecruiterContext(config=config, gateway=OpenRouterGateway.from_env(), rag=RagClient(config.rag_url), system_1=config.system_1)
    tiers = {p.id: p.tier for p in ai_pool.load_personas()} | {ai_pool.AUTHOR_ID: "author"}
    rows = match_and_screen(ctx, ai_pool.JOB_ID, ai_pool.POOL_IDS, tiers, occupation=ai_pool.OCCUPATION)
    showcase = Showcase(ai_pool.JOB_ID, ai_pool.AUTHOR_ID, ctx.system_1, rows)
    write_outputs(showcase, args.out)
    print(render_report(showcase))


if __name__ == "__main__":
    main()
