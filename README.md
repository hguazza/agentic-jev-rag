# SwiftAI Recruiter

An autonomous AI recruiter for **SwiftAI Recruit**, a fictional recruiting company (`docs/company-overview.md`). It has two services:

- **RAG API** (FastAPI): Search, Match and Ask over a recruiting corpus of CVs, Jobs, interview notes and Policies. It uses hybrid retrieval (dense plus sparse, fused with RRF), a reranker, and grounded answers with citations.
- **Recruiter API** (FastAPI and LangGraph): a five-agent system (Supervisor, Sourcer, Screener, Interviewer, Compliance). It turns a hiring goal into a **Recruitment report** and reaches the RAG only through HTTP tools.

The fast, typed checks inside the pipeline are the **System 1 decisions**: Route, Chunk relevance, Input guardrail and Groundedness check. Each can be made by a small LLM or by **Jev**, TypeSafe's "System One" model. The project measures how the two compare.

Vocabulary: `CONTEXT.md`. Design: `docs/superpowers/specs/2026-10-08-recruiter-design.md`. Decisions: `docs/adr/`. The earlier EU legal benchmark is parked in `docs/parked/`.

## Results

### System 1 comparison: LLM decider vs Jev

The LLM decider is the System 1 baseline (`anthropic/claude-haiku-4.5`, strict structured output). Jev is `typesafe/jev-1.13`. Both are called through OpenRouter with the same instructions, over 230 hand-labelled examples (`evals/data/`). Latency is end-to-end through OpenRouter.

| decision | implementation | n | accuracy | p50 latency (s) | p95 latency (s) | mean cost ($) |
|---|---|---|---|---|---|---|
| input_guardrail | llm | 60 | 95% | 1.601 | 2.041 | 0.0005196 |
| input_guardrail | jev | 60 | 98% | 0.438 | 0.547 | 0.0000231 |
| route | llm | 40 | 100% | 1.525 | 2.074 | 0.0003813 |
| route | jev | 40 | 100% | 0.440 | 0.601 | 0.0000187 |
| chunk_relevance | llm | 40 | 100% | 1.636 | 1.908 | 0.0003830 |
| chunk_relevance | jev | 40 | 98% | 0.452 | 0.503 | 0.0000174 |
| groundedness | llm | 30 | 100% | 1.589 | 1.852 | 0.0003423 |
| groundedness | jev | 30 | 97% | 0.422 | 0.499 | 0.0000167 |
| turn_check | llm | 60 | 98% | 1.815 | 2.225 | 0.0007266 |
| turn_check | jev | 60 | 95% | 0.570 | 0.634 | 0.0000311 |

| decision | Jev speed-up (p50) | Jev cost per decision | accuracy gap (Jev − LLM) |
|---|---|---|---|
| input_guardrail | 3.7× faster | 22× cheaper | +3 pts |
| route | 3.5× faster | 20× cheaper | +0 pts |
| chunk_relevance | 3.6× faster | 22× cheaper | −3 pts |
| groundedness | 3.8× faster | 20× cheaper | −3 pts |
| turn_check | 3.2× faster | 23× cheaper | −3 pts |

On these sets, Jev matches the LLM decider within 3 points on every decision, about 3.5× faster and about 20× cheaper per decision.

Caveats on reading these numbers:

- **Small samples.** A single example moves accuracy by 2–3 points, and repeated runs vary by that much (the LLM guardrail scored 97% on an earlier run).
- **Hardest guardrail case.** Both deciders allowed one indirect proxy: "must live within 2 km of the office even though the role is hybrid".
- **Turn check.** Rows for `turn_check` come from `results/system1-2026-10-09/` (the other rows are from the earlier run). Jev missed no prompt injections and reprompted no real answers; the LLM decider reprompted or redirected one real answer.
- **Raw files:** `results/system1-2026-10-08/` and `results/system1-2026-10-09/`. Re-run with `uv run python -m agentic_jev_rag.evals.system1`.

### Match evaluation

Ground truth comes from the synthetic interview notes: Candidates who were advanced, offered or hired for a Job, and who still have valid Consent. That gives 30 Jobs. Match scores **recall@5 = 0.37, recall@10 = 0.47 and MRR = 0.25** (`results/match-2026-10-08/`).

This is a deliberately noisy baseline. Outcomes were drawn so they only lean on skill overlap, so the ground truth is not circular.

### A real Recruiter run

`results/recruiter-runs/J-037-jev.json` is a full run for a Senior Linux Administrator Job with Jev as the System 1 decider. The Sourcer shortlisted 6 Candidates (all with valid Consent), and the Screener gave each a verdict citing CV chunks per Scorecard competency. All 6 cleared the policy's lenient advance rule (no competency at 1, average of 3 or more). Every verdict passed Jev's Groundedness check. The Interviewer then interviewed the top Candidate, a Simulated candidate, over 4 questions and recommended **no_hire**, because the hands-on Linux depth wasn't there.

The run cost $0.38 in Recruiter-side model calls and took about 3 minutes.

## Voice interview

The Interviewer can hold a live **voice** interview through LiveKit (ADR 0012). A LiveKit Agents worker turns speech into text, and a fast Voice host LLM asks the interview graph's questions out loud; the graph still decides every question and the assessment, and answers reach it as the exact transcript. See `docs/voice-demo.md` to run it from the LiveKit Agents Playground. It needs a LiveKit Cloud project (free tier) in `.env`.

Every reply gets a **Turn check**, a System 1 decision (intelligible, prompt injection, substantive or shallow) made before the System 2 model is asked anything: gibberish is reprompted and injections redirected without a System 2 call, and shallow answers get up to two forensic **probes**. **Interview signals** (evidence per competency, consistency with the Dossier) are computed in the same LangGraph superstep as the next question, so they add no latency. Delivery timings are reported as advisory **integrity signals** and never scored (ADR 0013).

## Architecture

```
goal ─► Recruiter API ─► LangGraph: Compliance intake ─► Supervisor ⇄ {Sourcer, Screener, Interviewer, Compliance review} ─► report
                                   │  HTTP tools: search, match_candidates, ask, get_document
                                   ▼
                         RAG API: Input guardrail → Route → hybrid search (dense + sparse, RRF) → rerank
                                  → Chunk relevance → System 2 answer with [n] citations → Groundedness check
                                   │                                   │
                                Qdrant                Model gateway → OpenRouter (chat, tools, embeddings, rerank, Jev)
```

**Model gateway** (`gateway.py`, `openrouter.py`). This is the only code that talks to models. Every call returns OpenRouter's reported cost, measured latency and a retry record (ADR 0005, 0007). It is also the only thing faked in tests.

**System 1 decider** (`system1/`). It has one interface and two implementations. The LLM decider uses strict JSON schemas. The Jev decider sends Jev typed `choice` and `noul` questions. Both use the same instructions (ADR 0003). The default comes from config, and you can choose per request with `"system_1": "llm" | "jev"`.

**Guarantees the code enforces, not just the prompts:**

- **Consent.** A Candidate without valid Consent is excluded from every Search, Match and Ask. They are also removed from a shortlist even if the Sourcer proposes them, and cannot be interviewed (HTTP 409).
- **Input guardrail first.** A discriminatory or out-of-scope goal or question is refused before any retrieval or agent runs.
- **Supervisor order.** The Supervisor can only hand off in a safe order (sourcing, then screening and interviews, then Compliance review, then finish). Invalid choices are corrected and recorded in the report.
- **Agents finish by submitting validated results.** Agents end by calling a submit tool whose payload is validated with Pydantic. Invalid submissions go back to the model to fix. Step limits and stop reasons appear in the report.
- **Grounded verdicts.** Every Screening verdict gets a Groundedness check against the chunks it cites.

## Corpus

| Collection | Contents | Source |
|---|---|---|
| Candidates | 200 CVs across 5 occupations (IT, HR, sales, design, finance) | `opensporks/resumes` (CC0). Names, locations and Consent dates are synthetic. |
| Jobs | 60 postings | EMSCAD job postings (CC0), rewritten for six fictional Clients |
| Interviews | 120 interview notes with Outcomes | Synthetic, generated for seeded (Candidate, Job) pairs |
| Policies | Fair hiring, data protection and consent, interview guide, Scorecards, fast-track, AI oversight | Hand-written |

The built corpus is committed in `corpus/`. Rebuilding it costs about $3: `uv run python -m agentic_jev_rag.corpus.build`. Dataset research is in `docs/research/2026-10-08-recruiting-datasets.md`.

## Run it

You need [uv](https://docs.astral.sh/uv/), Docker, and an OpenRouter key.

```sh
cp .env.example .env                                   # add OPEN_ROUTER_API_TOKEN
docker compose up -d qdrant
docker compose --profile setup run --rm ingest         # index the corpus (embeddings cost well under $0.01)
docker compose up -d                                   # rag-api on :8000, recruiter-api on :8001
```

Try it:

```sh
# Ask a policy question, with Jev making the System 1 decisions
curl -s localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"question": "How long is a candidate'\''s consent valid?", "system_1": "jev"}'

# Rank Candidates for a Job
curl -s localhost:8000/match -H 'Content-Type: application/json' -d '{"job_id": "J-037", "top_k": 5}'

# A discriminatory goal is refused before any agent runs
curl -s localhost:8001/runs -H 'Content-Type: application/json' \
  -d '{"goal": "Fill J-037 with young graduates only", "job_id": "J-037"}'

# A full recruitment (about 3 minutes, about $0.40)
curl -s localhost:8001/runs -H 'Content-Type: application/json' \
  -d '{"goal": "Fill the Senior Linux Administrator role", "job_id": "J-037", "system_1": "jev", "max_interviews": 1}'

# A live interview: start it, then reply to each question
curl -s localhost:8001/interviews -H 'Content-Type: application/json' -d '{"candidate_id": "C-0040", "job_id": "J-037"}'
curl -s localhost:8001/interviews/<interview_id>/reply -H 'Content-Type: application/json' -d '{"message": "..."}'
```

OpenAPI docs are served at `localhost:8000/docs` and `localhost:8001/docs`.

## Tests

```sh
uv sync
docker compose up -d qdrant
uv run pytest            # 104 tests, offline except Qdrant; the Model gateway is the only fake
uv run pytest -m live    # real OpenRouter calls: every Model role, Ask with both deciders, one full Recruiter run (about $1)
```

Tests sit at the public seams:
- the RAG API, over real Qdrant;
- the Recruiter API and graph;
- the System 1 deciders;
- the gateway's HTTP behaviour, against recorded OpenRouter responses;
- the corpus build;
- the evaluation metrics.

## Configuration

`configs/recruiter.toml` names an OpenRouter model for every Model role, so swapping a model is a config edit. The environment can override `QDRANT_URL`, `RAG_URL`, `SYSTEM_1` and `CONFIG` (the path to the TOML file).

## Not in v1

- AWS deployment (the images are ready for ECS).
- Student recruitment for institutions.
- Languages other than English.
- A UI.
- Authentication.
- Integrations with an ATS, email or scheduling.
- An MCP adapter over the RAG API.
- Durable interview state: live interviews are kept in memory (LangGraph `InMemorySaver`) and are lost on restart.
