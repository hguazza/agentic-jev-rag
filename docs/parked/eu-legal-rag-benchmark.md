# Parked: EU legal RAG benchmark

Status: parked on 2026-10-08 so the project can focus on a RAG for an AI recruiting platform. Nothing here is deleted; this note says where everything lives and how to pick it back up.

## The idea

A public, re-runnable benchmark of three RAG Levels over EU legal text, first in English and then in Portuguese. It tests one Headline claim: "Level 3 matches Level 2 answer quality at a fraction of the latency and cost."

- **Level 1, Vanilla RAG:** fixed-size chunks, dense retrieval, then the System 2 model answers.
- **Level 2, Agentic RAG:** structure-aware chunks, hybrid search, a reranker and an agent loop. A small LLM (the System 1 baseline) makes four System 1 decisions: Route, Chunk relevance, Input guardrail and Groundedness check.
- **Level 3, Agentic Jev RAG:** Level 2 with only the System 1 decisions moved to TypeSafe's Jev, called through OpenRouter.

The corpus has two Collections, each parallel in English and Portuguese:

- **Case law:** CJEU judgments, fetched from the EU Publications Office Cellar.
- **Legislation:** EU acts from MultiEURLEX.

## Where it lives

- **Full spec:** GitHub issue #1, with tickets #2–#15 under it.
- **Glossary:** `CONTEXT.md` as of commit `15e4af3`.
- **Decisions:** `docs/adr/`.
  - ADR 0004: Language as a benchmark dimension.
  - ADR 0006: the EU parallel corpus.
  - The other ADRs (0001–0003, 0005, 0007, 0008) are not specific to legal text and may carry over to the recruiting RAG.
- **Research:** `docs/research/2026-10-01-*.md`.
  - Cellar/SPARQL access and the EU reuse licence (Decision 2011/833/EU).
  - Jev on OpenRouter: endpoints, pricing and the 32k context limit.
  - English legal data sources that were considered and rejected.

## What was built before parking

- **#2, merged:** the Level 1 tracer bullet. `answer()` runs over three committed CJEU judgments, kept in `tests/fixtures/corpus/case_law/en/`.
- **#3, merged:** the real OpenRouter Model gateway (chat, structured output, embeddings, cost, latency, retries) and the TOML Model config.

## Where to resume

The next ticket was #8 (Level 2 retrieval). Tickets #4 (Case law ingestion) and #6 (Benchmark harness tracer) were also unblocked. The path that set the pace was #8 → #9 → #10 → #13 → #14.

How to verify Cellar access: content negotiation on `http://publications.europa.eu/resource/celex/{CELEX}` with `Accept: text/html, application/xhtml+xml` and `Accept-Language: eng|por`.
