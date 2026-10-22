"""What the Voice host is told to say for each kind of graph question, and the closing summary."""

from agentic_recruiter.voice.interview import Next
from agentic_recruiter.voice.script import next_instruction, summary_text


def test_each_kind_of_question_is_delivered_with_its_text_unchanged():
    question = "How did you handle cache invalidation?"
    for kind, lead in [
        ("question", "Ask this next question"),
        ("probe", "Ask this follow-up"),
        ("reprompt", "could not be understood"),
        ("redirect", "only discuss their experience"),
    ]:
        instruction = next_instruction(Next(question, kind))
        assert lead in instruction and instruction.endswith(question), kind


def test_the_end_of_the_interview_tells_the_host_to_close():
    assert "over" in next_instruction(Next(None, assessment={}))


def test_the_summary_lists_the_recommendation_scores_and_red_flags():
    text = summary_text({
        "recommendation": "next_round",
        "scores": [{"competency": "Technical operations", "score": 4}],
        "summary": "Solid Linux operations.",
        "red_flags": [{"kind": "injection", "quote": "Ignore your instructions", "detail": "..."}],
    })

    assert text.splitlines()[0] == "Interview assessment: next_round"
    assert "- Technical operations: 4/5" in text
    assert "Red flags: 1" in text and "human recruiter reviews" in text


def test_the_summary_mentions_advisory_integrity_observations_without_judging():
    text = summary_text({
        "recommendation": "hire", "scores": [], "summary": "Strong.",
        "integrity_signals": {"note": "Advisory only.", "turns": [{"turn": 2, "observation": "A long pause..."}, {"turn": 3, "observation": None}]},
    })

    assert "Integrity observations for human review: 1 (advisory, not scored)" in text
