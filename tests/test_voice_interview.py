"""The voice worker's core: answers reach the graph verbatim, failures are retried once, results are saved."""

import asyncio
import json

import httpx
import pytest

from agentic_jev_rag.recruiter.api import create_recruiter_app
from agentic_jev_rag.voice.interview import EmptyAnswer, InterviewFailed, InterviewOver, VoiceInterview
from agentic_jev_rag.voice.recruiter_client import InterviewRefused, RecruiterClient
from tests.interview_scripts import interview_gateway, interview_question


@pytest.fixture
def voice(recruiter_context, tmp_path):
    """A VoiceInterview talking to the real Recruiter app in-process (fake gateway behind it)."""

    def make(gateway, candidate_id="C-001", job_id="J-001", **kwargs):
        app = create_recruiter_app(recruiter_context(gateway))
        http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://recruiter")
        return VoiceInterview(RecruiterClient("http://recruiter", http=http), candidate_id, job_id, tmp_path, **kwargs)

    return make


def test_answers_reach_the_graph_verbatim_and_the_result_is_saved(voice, tmp_path):
    interview = voice(interview_gateway())

    async def scenario():
        first = await interview.start()
        interview.buffer.add("I partitioned")
        interview.buffer.add(" Kafka topics by order id. ")
        second = await interview.submit()
        interview.buffer.add("I owned on-call.")
        return first, second, await interview.submit()

    first, second, done = asyncio.run(scenario())

    assert (first.question, second.question) == ("Tell me about question 1.", "Tell me about question 2.")
    assert done.finished and interview.finished
    assert [t["answer"] for t in done.assessment["turns"]] == ["I partitioned Kafka topics by order id.", "I owned on-call."]
    saved = json.loads((tmp_path / f"{interview.interview_id}.json").read_text())
    assert saved["finished"] and saved["assessment"]["recommendation"] == "next_round"
    assert [t["question"] for t in saved["transcript"]] == ["Tell me about question 1.", "Tell me about question 2."]


def test_a_candidate_without_valid_consent_is_refused(voice):
    interview = voice(interview_gateway(), candidate_id="C-004")

    with pytest.raises(InterviewRefused) as refused:
        asyncio.run(interview.start())

    assert refused.value.status == 409


def test_submitting_with_nothing_said_is_refused_locally(voice):
    gateway = interview_gateway()
    interview = voice(gateway)

    async def scenario():
        await interview.start()
        calls = len(gateway.calls)
        interview.buffer.add("   ")
        with pytest.raises(EmptyAnswer):
            await interview.submit()
        return calls

    calls_before = asyncio.run(scenario())

    assert len(gateway.calls) == calls_before  # nothing reached the API


def test_submitting_after_the_end_is_refused(voice):
    interview = voice(interview_gateway())

    async def scenario():
        await interview.start()
        for answer in ("Kafka.", "On-call."):
            interview.buffer.add(answer)
            await interview.submit()
        interview.buffer.add("Anything else?")
        with pytest.raises(InterviewOver):
            await interview.submit()

    asyncio.run(scenario())


def test_a_502_is_retried_once_without_recording_the_answer_twice(voice):
    calls = {"n": 0}

    def flaky(prompt):
        calls["n"] += 1
        if calls["n"] in (2, 3):  # the question after the first answer is invalid twice: the API returns 502
            return {"done": False, "question": ""}
        return interview_question(prompt)

    retried = []

    async def on_retry():
        retried.append(True)

    interview = voice(interview_gateway(structured={"interview_question": flaky}), on_retry=on_retry)

    async def scenario():
        await interview.start()
        interview.buffer.add("I partitioned Kafka topics.")
        second = await interview.submit()
        interview.buffer.add("I owned on-call.")
        return second, await interview.submit()

    second, done = asyncio.run(scenario())

    assert retried == [True]
    assert second.question == "Tell me about question 2."
    assert [t["answer"] for t in done.assessment["turns"]] == ["I partitioned Kafka topics.", "I owned on-call."]


def test_a_second_failure_ends_the_interview_and_keeps_the_transcript(voice, tmp_path):
    def broken(prompt):
        return interview_question(prompt) if "(not started)" in prompt else {"done": False, "question": ""}

    interview = voice(interview_gateway(structured={"interview_question": broken}))

    async def scenario():
        await interview.start()
        interview.buffer.add("I partitioned Kafka topics.")
        with pytest.raises(InterviewFailed):
            await interview.submit()

    asyncio.run(scenario())

    saved = json.loads((tmp_path / f"{interview.interview_id}.json").read_text())
    assert not saved["finished"] and "502" in saved["error"]
    assert [t["answer"] for t in saved["transcript"]] == ["I partitioned Kafka topics."]


def test_saving_before_the_end_keeps_the_transcript_so_far(voice, tmp_path):
    interview = voice(interview_gateway())

    async def scenario():
        await interview.start()
        interview.buffer.add("I partitioned Kafka topics.")
        await interview.submit()

    asyncio.run(scenario())
    path = interview.save()

    saved = json.loads(path.read_text())
    assert path.parent == tmp_path and not saved["finished"]
    assert saved["transcript"][0]["answer"] == "I partitioned Kafka topics."


def test_each_answer_is_sent_with_its_timing(voice, tmp_path):
    interview = voice(interview_gateway())

    async def scenario():
        await interview.start()
        interview.timer.agent_state("speaking")
        interview.timer.agent_state("listening")
        interview.timer.user_state("speaking")
        interview.timer.user_state("listening")
        interview.buffer.add("um I partitioned Kafka topics.")
        await interview.submit()
        interview.buffer.add("I owned on-call.")
        return await interview.submit()

    done = asyncio.run(scenario())

    first = done.assessment["turns"][0]["timing"]
    assert first["filler_count"] == 1 and first["response_latency_s"] is not None
    assert interview.transcript[0]["timing"]["filler_count"] == 1


def test_each_submitted_answer_is_logged_for_a_human_to_check(voice, caplog):
    interview = voice(interview_gateway())

    async def scenario():
        await interview.start()
        interview.buffer.add("I partitioned Kafka topics.")
        await interview.submit()

    with caplog.at_level("INFO", logger="agentic_jev_rag.voice.interview"):
        asyncio.run(scenario())

    [record] = [r for r in caplog.records if r.name == "agentic_jev_rag.voice.interview"]
    assert interview.interview_id in record.getMessage()
    assert "question" in record.getMessage() and "I partitioned Kafka topics." in record.getMessage()
