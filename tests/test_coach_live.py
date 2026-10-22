"""Live Coach run on the author's recorded interview. Opt in with `uv run pytest -m live`; costs a few cents."""

import json
from pathlib import Path

import pytest

from agentic_jev_rag.coach import coach, find_candidate, render
from agentic_jev_rag.config import load_config
from agentic_jev_rag.corpus.documents import load_corpus
from agentic_jev_rag.openrouter import OpenRouterGateway
from agentic_jev_rag.system1 import make_deciders

pytestmark = pytest.mark.live

FIXTURE = Path(__file__).parent / "fixtures" / "voice" / "c0213-j061.json"


def test_the_author_is_coached_with_checked_model_answers_and_a_fix_for_the_ecs_claim():
    config = load_config()
    interview = json.loads(FIXTURE.read_text(encoding="utf-8"))
    gateway = OpenRouterGateway.from_env()
    candidate = find_candidate(load_corpus(config.corpus_dir), "C-0213")
    scorecards = (Path(config.corpus_dir) / "policies" / "scorecards.md").read_text(encoding="utf-8")

    coaching = coach(gateway, config.models.system_2_model, make_deciders(config, gateway)[config.system_1], interview, candidate, scorecards)

    assert len(coaching.debriefs) == 2 and all(d.model_answer.strip() for d in coaching.debriefs)
    assert 1 <= len(coaching.habits) <= 3
    assert [f["kind"] for f in coaching.red_flags] == ["contradiction"] and coaching.red_flags[0]["say_instead"].strip()
    assert [t["decision"] for t in coaching.trace].count("groundedness") >= 2
    assert coaching.cost_usd > 0 and "## Question 2" in render(coaching)
