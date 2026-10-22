"""Delivery timing of one spoken answer, from the session's agent and user state changes (advisory only, ADR 0013)."""

import re
import time
from typing import Callable

FILLERS = re.compile(r"\b(?:u+m+|u+h+|e+r+m*|h+m+|a+h+|m{2,})\b", re.I)


def count_fillers(text: str) -> int:
    return len(FILLERS.findall(text))


class TurnTimer:
    """Measures from the end of the agent's last speech before the candidate starts, to the candidate's last speech."""

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._reset()

    def agent_state(self, new_state: str) -> None:
        if self._agent_speaking and new_state != "speaking" and self._user_start is None:
            self._agent_end = self._clock()
        self._agent_speaking = new_state == "speaking"

    def user_state(self, new_state: str) -> None:
        now = self._clock()
        if new_state == "speaking":
            if self._user_start is None:
                self._user_start = now
        elif self._user_start is not None:
            self._user_end = now

    def take(self, transcript: str) -> dict:
        latency = duration = wpm = None
        if self._agent_end is not None and self._user_start is not None:
            latency = round(max(self._user_start - self._agent_end, 0.0), 2)
        if self._user_start is not None and self._user_end is not None:
            duration = round(self._user_end - self._user_start, 2)
            if duration > 0:
                wpm = round(len(transcript.split()) / duration * 60, 1)
        self._reset()
        return {"response_latency_s": latency, "answer_duration_s": duration, "words_per_minute": wpm, "filler_count": count_fillers(transcript)}

    def _reset(self) -> None:
        self._agent_speaking = False
        self._agent_end: float | None = None
        self._user_start: float | None = None
        self._user_end: float | None = None
