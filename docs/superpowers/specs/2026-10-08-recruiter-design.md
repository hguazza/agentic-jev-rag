# SwiftAI Recruiter: design

Date: 2026-10-08. Vocabulary: `CONTEXT.md`. Company: `docs/company-overview.md`. Target role: `docs/job-description.md`.

## Goal

Build an autonomous AI recruiter for SwiftAI Recruit:

1. A **production RAG service** (FastAPI) over recruiting documents. It exposes Search, Match and Ask as endpoints that agents call as tools.
2. A **LangGraph multi-agent Recruiter** with five agents: Supervisor, Sourcer, Screener, Interviewer and Compliance. It turns a hiring goal into a Recruitment report.
3. A **System 1 comparison.** The four System 1 decisions (Route, Chunk relevance, Input guardrail, Groundedness check) sit behind one System 1 decider interface with two implementations, a small LLM and Jev. An evaluation harness measures accuracy, latency and cost for each.

Success is reached when, from a clean clone, `docker compose up` plus one OpenRouter key gives:

- a RAG API and a Recruiter API that run a full recruitment for a seeded Job;
- a test suite that passes without network access, apart from Qdrant under Docker;
- a live suite and a System 1 comparison that run on real models.

## Decisions made without the owner (overridable)

- **English only.** The Language dimension is dropped.
- **Job applicants only.** Student recruitment for institutions is out of v1.
- **Compliance is the fifth agent.** It is backed by the System 1 decisions, which is where LLM versus Jev runs.
- **Corpus is public plus synthetic.** Public résumés and job postings form the backbone. Interview notes, Outcomes, Clients and Consent dates are synthetic. Policies are hand-written.
- **The Model gateway stays the single seam to OpenRouter (ADR 0005).** Agents call models through it, LangChain chat models are not used, and per-call cost is kept. LangGraph orchestrates; it does not call models itself.
- **AWS deployment is out of scope for v1.** The services ship as Docker images under Compose (ADR 0008), ready to move to ECS later.

## Architecture

```
               ┌──────────────── Recruiter API (FastAPI) ────────────────┐
  goal ──────► │ LangGraph: Compliance(intake) → Supervisor ⇄ {Sourcer,   │
               │ Screener, Interviewer, Compliance(review)} → report      │
               └───────────────┬──────────────────────────────────────────┘
                               │ HTTP tools (search, match, ask, document)
               ┌───────────────▼────────────── RAG API (FastAPI) ─────────┐
               │ Input guardrail → Route → hybrid Search (dense+sparse,   │
               │ RRF) → rerank → Chunk relevance → System 2 answer →      │
               │ Groundedness check                                       │
               └───────┬──────────────────────────────┬───────────────────┘
                       │                              │
                    Qdrant                    Model gateway → OpenRouter
                                       (chat, tools, embeddings, rerank, Jev)
```

Both services are one Python package (`agentic_jev_rag`), served by one Docker image with two commands.

## Components

### Model gateway (`gateway.py`, `openrouter.py`)

This is the only code that talks to OpenRouter. It covers:

- `complete` (text, with optional structured output);
- `chat` (a message list with optional tools; returns the assistant message, including tool calls);
- `embed`;
- `rerank` (`/api/v1/rerank`);
- `decide` (Jev, through `/api/v1/systemone`).

Every result carries latency, reported cost and retries. `FakeModelGateway` is the only fake in the test suite:

- embeddings hash words into vectors;
- rerank scores by word overlap;
- chat and complete answers come from a scripted responder;
- `decide` answers come from a scripted responder.

### Corpus (`corpus/`)

- `Document(id, collection, title, text, metadata)`. The metadata depends on the Collection:
  - **Candidate:** name (synthetic), occupation, years_experience, location, skills, consent_until.
  - **Job:** client, occupation, seniority, location, required_skills.
  - **Interview note:** candidate_id, job_id, stage, outcome, date.
  - **Policy:** topic.
- `build_corpus` is a script. It downloads the public datasets, samples by occupation, extracts structured metadata with the System 1 baseline, generates the synthetic documents with the System 2 model (fixed seed), and writes `corpus/*.jsonl`.
- The built corpus is **committed**, so tests and readers never need to rebuild it.
- Policies are hand-written Markdown in `corpus/policies/`.

### RAG (`rag/`)

- **Chunking is structure-aware.** CVs split on section headings (summary, experience, education, skills). Jobs split into description, requirements and benefits. Policies split on Markdown headings. Interview notes stay whole. Long sections fall back to fixed-size windows. Every chunk carries its document's metadata.
- **Index:** one Qdrant collection per Collection, each with a named dense vector and a named sparse vector.
  - The sparse vector is our own hashed term frequency. Qdrant's `modifier: idf` turns it into BM25-style scoring, so no extra model is needed.
  - Hybrid queries use Qdrant `prefetch` on both vectors, fused with RRF.
- **Retrieval:** hybrid Search, then metadata filters (occupation, location, min years, valid Consent), then optional rerank with the reranker Model role.
- **Match:** the Job's text is the query. It hybrid-searches the Candidates Collection, groups chunks per Candidate, reranks one evidence summary per Candidate, and returns ranked Candidates with Evidence. Candidates without valid Consent are excluded.
- **Ask:**
  1. Input guardrail (refusal ends the request).
  2. Route, unless Collections were given.
  3. Search and rerank.
  4. Chunk relevance (dropped chunks are not shown to the model).
  5. The System 2 model answers with `[n]` citations.
  6. Groundedness check.

  The System 1 trace is returned with the answer.

### System 1 (`system1/`)

- `System1Decider` protocol:
  - `route(question) -> RouteDecision`
  - `chunk_relevance(question, chunks) -> list[RelevanceDecision]`
  - `input_guardrail(text) -> GuardrailDecision`
  - `groundedness(claim, evidence) -> GroundednessDecision`
- Each call also returns `System1TraceEntry`s (decision, outcome, latency, cost, implementation).
- **`LlmDecider`** prompts the System 1 baseline with strict structured output.
- **`JevDecider`** asks Jev typed questions:
  - Route is a Choice.
  - Chunk relevance is one Noul per chunk, all in one request.
  - Input guardrail is a Choice (allowed, discriminatory, out of scope).
  - Groundedness check is a Noul.
- The requests and prompts carry the same instructions, so the comparison is fair (ADR 0003).
- Which decider runs is chosen by config, and can be overridden per request.

### RAG API (`rag/api.py`)

| Endpoint | Purpose |
|---|---|
| `POST /search` | Hybrid search with filters and optional rerank |
| `POST /match` | Ranked Candidates for a Job, with Evidence |
| `POST /ask` | Grounded answer, citations and System 1 trace |
| `GET /documents/{id}` | One document |
| `GET /health` | Health check |

Requests and responses are Pydantic models. The System 1 decider is chosen with `system_1: "llm" | "jev"`.

### Recruiter (`recruiter/`)

- **Tools** are typed functions over a `RagClient`, an httpx client for the RAG API. In tests the client talks to the in-process FastAPI app through an ASGI transport.
- **Agent loop:** each agent is a tool-calling loop driven through `gateway.chat`. The agent ends by calling its own `submit_*` tool, which carries the structured result. There is a step limit, and running out of steps is reported in the Recruitment report.
- **Graph:** a LangGraph `StateGraph`.
  1. `compliance_intake` runs the Input guardrail on the goal. If it refuses, the report says so and the run ends.
  2. `supervisor` is the System 2 model with handoff tools (`to_sourcer`, `to_screener`, `to_interviewer`, `to_compliance`, `finish`).
  3. The Sourcer submits a shortlist.
  4. The Screener runs once per shortlisted Candidate and submits a Screening verdict.
  5. The Interviewer interviews the advanced Candidates and submits an Interview assessment. It uses a Simulated candidate in autonomous runs, or waits for human replies via `interrupt` in the interview API.
  6. `compliance_review` runs the Groundedness check on each verdict against its Evidence, re-checks Consent on the shortlist, and asks Policy questions through Ask.
  7. `report` builds the Recruitment report.
- **Recruiter API:**

  | Endpoint | Purpose |
  |---|---|
  | `POST /runs` | Goal plus `job_id`, returns a Recruitment report |
  | `POST /interviews` | Start a live interview, returns the first question |
  | `POST /interviews/{id}/reply` | Answer and get the next question or the assessment; uses the LangGraph checkpointer |
  | `GET /health` | Health check |

### Evaluation (`evals/`)

- **Labelled sets** in `evals/data/`:
  - Input guardrail: about 60 requests (allowed, discriminatory, out of scope).
  - Route: about 40 questions, each labelled with its Collection.
  - Chunk relevance: about 40 question–chunk pairs from the corpus.
  - Groundedness check: about 30 claim–evidence pairs.
- **System 1 comparison runner:** runs both deciders over every Labelled set and writes per-example records plus a summary to `results/`. The summary has accuracy, latency p50/p95 and cost per decision.
- **Match evaluation:** recall@k and MRR of Match against the synthetic Outcomes. A Candidate advanced or hired for a Job counts as relevant.

## Testing

Tests are written only at these seams:

1. **RAG API:** FastAPI TestClient against real Qdrant (Docker) with the fake gateway. Covers Search, filters, Match, Ask, refusal, routing, citations, the Groundedness check and the System 1 trace.
2. **Model gateway HTTP:** recorded OpenRouter responses for chat, tool calls, embeddings, rerank and Jev.
3. **System 1 deciders:** both implementations, through the fake gateway's scripted decisions and completions. Checks the typed outcomes and the trace entries.
4. **Recruiter:** the Recruiter API (and graph) with the fake gateway, the in-process RAG API and real Qdrant. Covers refusing a discriminatory goal, the full run producing a Recruitment report, the live interview conversation, and the Compliance findings.
5. **Corpus build:** normalisation, sampling and chunking from small fixture files.
6. **Evaluation:** metric aggregation from known records.
7. **Live suite** (`-m live`): one real call per Model role, a real Ask, and a real Recruiter run on a seeded Job.

## Out of scope for v1

- AWS deployment.
- Student recruitment.
- Portuguese.
- A UI.
- Authentication.
- A real ATS integration.
- Email or scheduling.
- An MCP adapter, which would be cheap to add over the RAG API later.
