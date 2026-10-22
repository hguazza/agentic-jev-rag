"""One live voice interview: the answer buffer, the calls to the Recruiter API, and the saved result.

No LiveKit here: the LiveKit agent feeds final transcripts into `buffer` and speaks what `submit()` returns.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

from agentic_recruiter.voice.recruiter_client import RecruiterClient, RecruiterError
from agentic_recruiter.voice.timing import TurnTimer

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Next:
    """What the graph wants next: a question and how to deliver it, or the end of the interview."""

    question: str | None
    kind: str = "question"
    assessment: dict | None = None

    @property
    def finished(self) -> bool:
        return self.question is None


class AnswerBuffer:
    """The candidate's final transcripts since the current question was asked. What is graded is exactly this."""

    def __init__(self):
        self._parts: list[str] = []

    def add(self, transcript: str) -> None:
        if text := transcript.strip():
            self._parts.append(text)

    def take(self) -> str:
        text, self._parts = " ".join(self._parts), []
        return text

    def __bool__(self) -> bool:
        return bool(self._parts)


class EmptyAnswer(Exception):
    """submit() was called before the candidate said anything."""


class InterviewOver(Exception):
    """submit() was called after the interview finished."""


class InterviewFailed(Exception):
    """The Recruiter API failed again after a retry, or could not be retried; the interview cannot continue."""


class VoiceInterview:
    def __init__(
        self,
        client: RecruiterClient,
        candidate_id: str,
        job_id: str,
        results_dir: Path,
        *,
        on_retry: Callable[[], Awaitable[None]] | None = None,
    ):
        self.client = client
        self.candidate_id = candidate_id
        self.job_id = job_id
        self.results_dir = Path(results_dir)
        self.on_retry = on_retry
        self.buffer = AnswerBuffer()
        self.timer = TurnTimer()
        self.interview_id: str | None = None
        self.current: Next | None = None
        self.transcript: list[dict] = []
        self.error: str | None = None

    @property
    def finished(self) -> bool:
        return self.current is not None and self.current.finished

    @property
    def assessment(self) -> dict | None:
        return self.current.assessment if self.current else None

    async def start(self) -> Next:
        state = await self.client.start(self.candidate_id, self.job_id)
        self.interview_id = state["interview_id"]
        return self._advance(state)

    async def submit(self) -> Next:
        """Send everything the candidate said since the question, verbatim, and return what comes next."""
        if self.finished:
            raise InterviewOver("the interview is already over")
        answer = self.buffer.take()
        if not answer:
            raise EmptyAnswer("the candidate has said nothing since the question was asked")
        timing = self.timer.take(answer)
        log.info("Submitting answer: interview=%s kind=%s answer=%r", self.interview_id, self.current.kind, answer)
        self.transcript.append({"question": self.current.question, "kind": self.current.kind, "answer": answer, "timing": timing})
        try:
            state = await self._reply(answer, timing)
        except RecruiterError as error:
            self.error = str(error)
            self.save()
            raise InterviewFailed(str(error)) from error
        return self._advance(state)

    def save(self) -> Path:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        path = self.results_dir / f"{self.interview_id or 'not-started'}.json"
        result = {
            "interview_id": self.interview_id,
            "candidate_id": self.candidate_id,
            "job_id": self.job_id,
            "finished": self.finished,
            "transcript": self.transcript,
            "assessment": self.assessment,
            "error": self.error,
        }
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        return path

    async def _reply(self, answer: str, timing: dict) -> dict:
        try:
            return await self.client.reply(self.interview_id, answer, timing)
        except RecruiterError as error:
            if not error.retryable:
                raise
            if self.on_retry:
                await self.on_retry()
            return await self.client.reply(self.interview_id, answer, timing)

    def _advance(self, state: dict) -> Next:
        self.current = Next(state["question"], state.get("question_kind") or "question", state.get("assessment"))
        if self.finished:
            self.save()
        return self.current
