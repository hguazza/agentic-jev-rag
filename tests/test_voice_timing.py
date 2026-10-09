"""Timing of one spoken answer, from LiveKit agent and user state changes."""

from agentic_jev_rag.voice.timing import TurnTimer, count_fillers


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def at(clock, t, call, state):
    clock.now = t
    call(state)


def test_latency_runs_from_the_end_of_the_question_to_the_start_of_the_answer():
    clock = Clock()
    timer = TurnTimer(clock)
    at(clock, 0.0, timer.agent_state, "speaking")
    at(clock, 4.0, timer.agent_state, "listening")  # question ends
    at(clock, 6.5, timer.user_state, "speaking")  # answer starts
    at(clock, 9.0, timer.user_state, "listening")
    at(clock, 9.5, timer.agent_state, "speaking")  # acknowledgement: does not move the question's end
    at(clock, 10.0, timer.agent_state, "listening")
    at(clock, 10.5, timer.user_state, "speaking")
    at(clock, 18.5, timer.user_state, "listening")  # answer ends

    timing = timer.take("um I fixed the uh logrotate config and added a disk alert at eighty percent")

    assert timing["response_latency_s"] == 2.5
    assert timing["answer_duration_s"] == 12.0
    assert timing["words_per_minute"] == round(15 / 12 * 60, 1)
    assert timing["filler_count"] == 2


def test_take_resets_for_the_next_question_and_reports_unknowns_as_none():
    timer = TurnTimer(Clock())
    timer.take("")

    assert timer.take("hello") == {"response_latency_s": None, "answer_duration_s": None, "words_per_minute": None, "filler_count": 0}


def test_fillers_are_whole_words_only():
    assert count_fillers("Umm, uh, the summary was hmm... errr fine. Umbrella, hummus, error.") == 4
