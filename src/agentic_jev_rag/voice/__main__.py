"""Starts the voice worker in `agentic_jev_rag.voice.worker`: `uv run --group voice python -m agentic_jev_rag.voice dev`."""

from dotenv import find_dotenv, load_dotenv
from livekit.agents import cli

from agentic_jev_rag.voice.worker import server

if __name__ == "__main__":
    load_dotenv(find_dotenv(usecwd=True))  # LiveKit reads LIVEKIT_* from the environment at startup
    cli.run_app(server)
