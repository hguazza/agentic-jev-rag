# Agentic Jev RAG

A public, re-runnable benchmark of three RAG Levels over EU legal text. See `CONTEXT.md` for the vocabulary and `docs/adr/` for the decisions behind it.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Docker.

```sh
uv sync
cp .env.example .env
docker compose up -d   # Qdrant on localhost:6333
uv run pytest
```
