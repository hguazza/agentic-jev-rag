"""Starts the voice worker in `agentic_recruiter.voice.worker`: `uv run --group voice python -m agentic_recruiter.voice dev`."""

from dotenv import find_dotenv, load_dotenv
from livekit.agents import cli

from agentic_recruiter.voice.worker import server

if __name__ == "__main__":
    load_dotenv(find_dotenv(usecwd=True))  # LiveKit reads LIVEKIT_* from the environment at startup
    cli.run_app(server)
