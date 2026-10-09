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
