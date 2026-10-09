"""The voice worker: `uv run --group voice python -m agentic_jev_rag.voice dev`.

Needs LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET and OPEN_ROUTER_API_TOKEN, and the Recruiter API running.
First run once: `uv run --group voice python -m agentic_jev_rag.voice download-files`.
"""

import asyncio
import logging
import os
from pathlib import Path

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli, inference
from livekit.plugins import openai, silero

from agentic_jev_rag.config import Config, load_config
from agentic_jev_rag.openrouter import BASE_URL
from agentic_jev_rag.voice.agent import InterviewerVoice
from agentic_jev_rag.voice.interview import VoiceInterview
from agentic_jev_rag.voice.recruiter_client import InterviewRefused, RecruiterClient, RecruiterError
from agentic_jev_rag.voice.script import REFUSAL, SORRY, summary_text

log = logging.getLogger(__name__)

RESULTS_DIR = Path("results") / "voice-interviews"

server = AgentServer()


def build_session(config: Config) -> AgentSession:
    return AgentSession(
        stt=inference.STT(config.voice.stt, language="en"),
        llm=openai.LLM(model=config.models.voice_host, base_url=BASE_URL, api_key=os.environ["OPEN_ROUTER_API_TOKEN"]),
        tts=inference.TTS(config.voice.tts),
        vad=silero.VAD.load(),
        # Candidates pause to think: when the turn detector is unsure, wait up to 5 s of silence.
        turn_handling={"turn_detection": inference.TurnDetector(), "endpointing": {"max_delay": 5.0}},
    )


@server.rtc_session()
async def interview(ctx: JobContext) -> None:
    config = load_config()
    await ctx.connect()
    participant = await ctx.wait_for_participant()
    candidate_id = participant.attributes.get("candidate_id") or config.voice.default_candidate_id
    job_id = participant.attributes.get("job_id") or config.voice.default_job_id
    session = build_session(config)

    async def on_retry() -> None:
        session.say(SORRY)

    voice = VoiceInterview(RecruiterClient(config.voice.recruiter_url), candidate_id, job_id, RESULTS_DIR, on_retry=on_retry)
    try:
        first = await voice.start()
    except (InterviewRefused, RecruiterError) as error:
        if isinstance(error, RecruiterError):
            voice.error = str(error)
            voice.save()
        await session.start(Agent(instructions="You end calls politely."), room=ctx.room)
        await session.say(REFUSAL, allow_interruptions=False)
        session.shutdown()
        return

    done = asyncio.Event()
    closing_started = False

    @session.on("user_input_transcribed")
    def on_transcript(event) -> None:
        if event.is_final:
            voice.buffer.add(event.transcript)

    @session.on("agent_state_changed")
    def on_agent_state(event) -> None:
        # The closing words have been spoken once the agent is back to listening after speaking them.
        nonlocal closing_started
        if not (voice.finished or voice.error):
            return
        if event.new_state == "speaking":
            closing_started = True
        elif event.new_state == "listening" and closing_started:
            done.set()

    session.on("close", lambda event: done.set())

    await session.start(InterviewerVoice(voice, first), room=ctx.room)
    await done.wait()
    voice.save()
    if voice.finished:
        try:
            await ctx.room.local_participant.send_text(summary_text(voice.assessment), topic="lk.chat")
        except Exception:  # the candidate may already have left the room
            log.exception("Could not post the interview summary")
    session.shutdown()


if __name__ == "__main__":
    cli.run_app(server)
