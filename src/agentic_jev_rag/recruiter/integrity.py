"""Integrity signals: delivery timings of spoken answers, with advisory observations for a human (ADR 0013).

They are computed here, after the assessment, from the turns' timing. The assessment prompt never sees them.
"""

from agentic_jev_rag.recruiter.schemas import IntegritySignals, InterviewTurn, TurnIntegrity

LONG_PAUSE_S = 8.0
FAST_WPM = 170.0
OBSERVATION = (
    "A long pause before a fast answer with no filler words. Worth a follow-up question by a human; not evidence of "
    "anything on its own."
)


def integrity_signals(turns: list[InterviewTurn]) -> IntegritySignals:
    entries = []
    for n, turn in enumerate(turns, start=1):
        if turn.timing is None:
            continue
        t = turn.timing
        flagged = (
            (t.response_latency_s or 0) >= LONG_PAUSE_S and t.filler_count == 0 and (t.words_per_minute or 0) >= FAST_WPM
        )
        entries.append(TurnIntegrity(turn=n, timing=t, observation=OBSERVATION if flagged else None))
    return IntegritySignals(turns=entries)
