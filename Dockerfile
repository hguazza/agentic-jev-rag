# Two images from one build: `api` (the RAG and Recruiter APIs; docker-compose.yml and ECS pick the command)
# and `voice` (the LiveKit voice worker). `api` is last, so it is the default target.
FROM python:3.12-slim AS base
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /uvx /bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY configs ./configs
COPY corpus ./corpus
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH" CONFIG=/app/configs/recruiter.toml

# The voice dependency group, with the Silero VAD and turn-detector model files downloaded at build time.
FROM base AS voice
RUN uv sync --frozen --no-dev --group voice && python -m agentic_recruiter.voice download-files
CMD ["python", "-m", "agentic_recruiter.voice", "start"]

FROM base AS api
EXPOSE 8000
CMD ["uvicorn", "agentic_recruiter.rag.api:create_default_rag_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
