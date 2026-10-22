# AI engineer showcase: design

Date: 2026-10-09. Vocabulary: `CONTEXT.md`. Builds on `2026-10-08-recruiter-design.md` and `2026-10-09-voice-interview-design.md`.

## Goal

Show the Recruiter hiring for the role this project was built for: micro1's AI/ML Engineer, Internal Platforms (`docs/job-description.md`). A pool of 12 synthetic AI engineer Candidates, spread across fit tiers, competes with the author (Henrique Guazzelli, real name and CV). The Recruiter ranks all 13 with Match, screens them, interviews the ones the Screener advances, and the author holds the same interview by voice through the voice worker. A committed report compares everyone on the same Scorecard, graded by the same judge.

Success: `results/ai-engineer-showcase/report.md` shows a leaderboard of all 13 Candidates (Match rank, Screening verdict, Interview scores per competency, recommendation, probes sustained, red flags), the author's row highlighted, a short analysis of where the author won or lost points against the strong tier with quotes from the transcripts, and the limits of the comparison. The run is reproducible from committed files.

## Decisions

- **Additive pool, not a corpus rebuild.** The public resume dataset has no AI engineering CVs, and rebuilding would renumber Candidates the voice spec depends on (C-0040, C-0150). The pool is appended as C-0201 to C-0213 and J-061; the existing 380 documents and `corpus/build.py` are unchanged.
- **micro1 is J-061's Client.** The one real Client in the corpus, kept on purpose because the project targets that role. The posting is micro1's own public job description. Every other Client stays fictional.
- **The author's CV is public.** Real name, real CV and real Dossier are committed and served by the API.
- **The Screener filters, as in the real pipeline.** Only fake Candidates the Screener advances are interviewed. The author is interviewed regardless, and their Screening verdict is reported honestly beside it.
- **Same judge, enforced.** Every Interview assessment records the config it was graded under; the runner refuses to import a voice interview graded under a different config.
- **Simulated candidates play a Persona.** Without one, every Simulated candidate answers well and the comparison flatters everyone.
- **Personas stay out of the corpus.** Qdrant stores each document's metadata in every chunk and Search returns it to the Screener, so a Persona in the metadata would tell the Screener who is a bluffer. Personas live only in `corpus/ai_pool/personas.toml`, and all 13 pool Candidates, the author included, have `source: "ai_pool"`.
- **Out of scope**: deployment (its own spec), a web UI for the comparison, runs on both System 1 deciders (the configured one runs and the report names it), repeated runs per Candidate, changes to the voice spec beyond what it plans.

## Scorecard: `ai_engineering`

A new occupation. `corpus/policies/scorecards.md` gains `## AI engineering`, parsed by the existing `interviewer.competencies()`:

- **LLM systems:** agents, tool calling, prompt chains and orchestration (LangGraph, LangChain or LlamaIndex).
- **Retrieval and evaluation:** RAG architecture, vector databases and deterministic evaluation harnesses.
- **Production engineering:** Python and FastAPI services that are low-latency, tested and observable.
- **Cloud delivery:** deploys, monitors and scales on AWS; ships customer-facing features fast.

## Job J-061

"AI/ML Engineer, Internal Platforms — micro1", rewritten from `docs/job-description.md` with the same `JOB_SCHEMA` sections as the other Jobs (about, responsibilities, requirements, benefits). Metadata: `client: "micro1"`, `occupation: "ai_engineering"`, `seniority: "senior"`, `location: "Remote"`, `required_skills` from the core stack (Python, FastAPI, LLMs, LangGraph, LangChain, LlamaIndex, RAG, vector databases, evaluation harnesses, AWS), `source: "docs/job-description.md"`.

## The pool (`corpus/ai_pool.py`, `corpus/ai_pool/`)

**Persona grid**: `corpus/ai_pool/personas.toml`, one entry per fake Candidate:

| field | values |
|---|---|
| `id` | C-0201 … C-0212 |
| `tier` | `strong` (3), `gaps` (4), `adjacent` (3), `bluffer` (2) |
| `background` | one line, e.g. "senior LLM platform engineer, agents in production at a fintech"; "strong FastAPI backend engineer, RAG side projects only"; "data scientist, classical ML, notebooks"; "MLOps engineer, Kubernetes and model serving"; "LangChain tutorial portfolio, no production systems" |
| `years_experience`, `location` | per persona |
| `speaking_style` | e.g. "concise and precise", "rambling, many fillers", "nervous, self-correcting", "confident, buzzword-heavy" |
| `honesty` | `truthful`, or `bluffer` for the bluffer tier |

**Generation**: `uv run python -m agentic_jev_rag.corpus.ai_pool --seed 7`. For each persona, the System 2 model writes:

- a plain-text CV in the layout of the existing Candidates (summary, highlights, dated experience, education);
- a Dossier in the voice spec's shape, `metadata.dossier = {"cover_letter", "linkedin": {"headline", "about", "positions"}}`, added with the voice spec's own `corpus.dossier` once it is merged: consistent with the CV for truthful personas, and with its deliberate discrepancies (a different employer, different years) for the two bluffers, so the contradiction signal has something to catch.

Profile metadata follows the existing Candidates: `name` from `corpus.synthetic` (seeded, unique), `consent_until` 6 to 24 months ahead (always valid), `headline` and `skills` extracted as in `build.extract_profile`, `occupation: "ai_engineering"`, `years_experience` and `location` from the persona, `source: "ai_pool"`. No Persona fields.

**The author, C-0213**: hand-written sources in `corpus/ai_pool/henrique/` (`profile.toml` with name and location, `cv.md`, `cover_letter.md`, `linkedin.json` in the Dossier shape), converted to the same document shape, `source: "ai_pool"`. Consent date: one year from the run.

**Writing**: the script removes any existing C-0201 to C-0213 and J-061 rows from `corpus/candidates.jsonl` and `corpus/jobs.jsonl` and appends the new ones, so re-running replaces the pool and touches nothing else. Re-index with `docker compose --profile setup run --rm ingest`.

## Simulated candidate with a Persona (`recruiter/interviewer.py`, `recruiter/prompts.py`)

`simulated_answer` takes an optional Persona (the runner loads it from `personas.toml`) and the turns so far; with a Persona it uses a Persona prompt, without one today's prompt is unchanged.

- **Spoken answers**: first person, with the fillers, false starts and uneven length of speech in the persona's `speaking_style`, so the text compares with an STT transcript.
- **Depth by tier**: `strong` gives the how, why, numbers and trade-offs; `gaps` is specific in its strong areas and thin elsewhere; `adjacent` steers toward the work it knows.
- **Honesty**: `truthful` stays within the CV and Dossier. `bluffer` answers the first question on a topic confidently, then turns vague, generic or self-contradictory when a probe asks how something worked.
- **Context**: the whole Dossier (`interviewer.dossier_text`, from the voice spec) and the interview so far, so answers to probes follow from earlier answers.

New Model role `simulated_candidate` in `[models]`, default the System 2 model (the System 1 baseline plays bluffers poorly).

## Same judge (`recruiter/schemas.py`)

`InterviewAssessment` gains `graded_under: {system_2_model, system_1, interview_guide_sha}`, filled by `assess()`: the System 2 model id, the System 1 decider name, and the SHA-256 of the Interview Guide and Scorecards text it was briefed with. Voice and simulated interviews fill it the same way, since both go through the interview graph.

## Runner (`src/agentic_jev_rag/showcase.py`)

`uv run python -m agentic_jev_rag.showcase [--live-interview results/voice-interviews/<id>.json] [--out results/ai-engineer-showcase]`

In-process against the RAG service and the Recruiter graph, with the configured System 1 decider:

1. **Match** J-061 over Candidates filtered to `occupation = ai_engineering`, `top_k = 13`: every Candidate ranked, with Evidence.
2. **Screen** all 13 with the Screener: a Screening verdict with a score per competency.
3. **Interview** the fake Candidates the Screener advanced through the Recruiter API in-process (`POST /interviews`, `POST /interviews/{id}/reply`, the same endpoints the voice worker calls), with the Persona Simulated candidate answering.
4. **Import** the author's voice interview from `--live-interview` (the `InterviewState` file the voice worker writes). The runner checks the file is for C-0213 and J-061 and that its `graded_under` equals the run's; on a mismatch it stops and names the differing field. Without `--live-interview`, the author's row shows Match and Screening only and the report says the interview is pending.

**Output**, committed in `results/ai-engineer-showcase/`:

- `run.json`: the full run, every verdict and assessment, and the System 1 trace.
- `transcripts/C-02xx.md`: one readable transcript per interviewed Candidate, marking probes, reprompts, redirects and red flags.
- `report.md`:
  - the leaderboard (Match rank, Screening verdict and scores, Interview scores per competency, recommendation, probes sustained, red flags), sorted by Interview score then Screening score, the author's row highlighted;
  - where the author won or lost points against the strong tier, per competency, quoting both transcripts (written by the System 2 model from `run.json`, every claim citing a turn);
  - the limits: simulated text against live speech; integrity signals are advisory and exist only for the author; Personas are written by the same model family that grades; one run per Candidate; the System 1 decider used.

## Dependencies and order

1. Scorecard, J-061, the pool and the author's files: buildable now from `main`.
2. Runner steps 1–2 (Match, Screening) and a report without interviews: buildable now.
3. Persona Simulated candidate, `graded_under`, runner steps 3–4: after voice slices 2–5 are merged (Turn check, probes, Interview signals, Dossier), so fake transcripts get the same probes and reprompts as the author's.
4. The author's voice interview, then the final committed run.

Needs from the author: `corpus/ai_pool/henrique/` sources (any format; converted to text).

## Testing

All with `FakeModelGateway`:

- **Corpus**: the persona grid parses and has the agreed tier counts; generated documents have the right ids and metadata and no Persona fields; the author's Dossier has the voice spec's shape; missing author files fail before any model call; re-running replaces only C-0201 to C-0213 and J-061; `competencies()` returns the four AI engineering competencies.
- **Simulated candidate**: the Persona prompt carries tier, style and honesty; without a persona the prompt is today's.
- **Same judge**: `assess()` fills `graded_under`; the runner rejects a live interview with a different System 2 model, decider or guide hash, or the wrong Candidate or Job.
- **Runner**: from scripted gateway replies, only advanced fakes are interviewed, the author is always included, and `report.md` has 13 rows with the author's highlighted.
- **Live** (`-m live`): one end-to-end showcase run without `--live-interview`.

## Docs

- ADR 0014: the AI engineer showcase: micro1 as the one real Client, the author's public CV, Persona Simulated candidates and their limits, the same-judge rule.
- `CONTEXT.md`: Persona, Showcase, the `ai_engineering` occupation, and the micro1 exception under **Client**.
- README: a "How I compare" section linking `results/ai-engineer-showcase/report.md`, with the commands to regenerate the pool and rerun the showcase.
