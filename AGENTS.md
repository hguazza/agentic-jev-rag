# AGENTS.md

SwiftAI Recruiter: a FastAPI RAG service and a LangGraph five-agent Recruiter that calls it, with System 1 decisions made by an LLM or by Jev. Read `CONTEXT.md` for the project's language and `docs/adr/` for decisions before working. The current design is `docs/superpowers/specs/2026-10-08-recruiter-design.md`; the EU legal benchmark it replaced is parked in `docs/parked/`.

Run `uv run pytest` with Qdrant and DynamoDB Local up (`docker compose up -d qdrant dynamodb`). The Model gateway is the only fake; live tests are opt-in (`-m live`).

## Agent skills

### Issue tracker

Issues and specs live in GitHub Issues, managed with the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Uses the five default triage labels (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `CONTEXT.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
