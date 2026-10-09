# One image for both services; docker-compose.yml picks the command.
FROM python:3.12-slim
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
EXPOSE 8000
CMD ["uvicorn", "agentic_jev_rag.rag.api:create_default_rag_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
