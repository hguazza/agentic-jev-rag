# Voice interview demo

A live voice interview with SwiftAI's AI Interviewer, from the LiveKit Agents Playground.

## Run it

1. `docker compose up -d qdrant rag-api recruiter-api` (index the corpus first if you have not: `docker compose --profile setup run --rm ingest`).
2. `uv sync --group voice`, then once: `uv run --group voice python -m agentic_jev_rag.voice download-files`.
3. `uv run --group voice python -m agentic_jev_rag.voice dev`. Wait for the worker to register with LiveKit Cloud.
4. Open https://agents-playground.livekit.io, connect to your LiveKit Cloud project, and join a room. The worker joins and starts the interview for the default Candidate and Job in `configs/recruiter.toml` (`[voice]`). To choose others, set participant attributes `candidate_id` and `job_id` before connecting.
5. At the end, the assessment summary appears in the Playground chat; the full result is in `results/voice-interviews/<interview_id>.json`.

## Script

1. **Strong answer.** Describe a real incident with specifics (what broke, what you did, the result). The interviewer moves to the next topic.
