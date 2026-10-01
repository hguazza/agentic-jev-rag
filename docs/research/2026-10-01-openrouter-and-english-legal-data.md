# Research: Jev via OpenRouter, OpenRouter model IDs, and English legal data sources

Date: 2026-10-01. Method: live `curl` against public endpoints (no API keys used), first-party docs (OpenRouter `llms.txt` pages, TypeSafe, CourtListener, case.law, EU Publications Office, UK National Archives), the `langchain-typesafe==0.0.1a3` wheel source, and the Hugging Face Hub API. Every claim cites its source. Anything not confirmed against a primary source is marked **UNVERIFIED**. This note follows on from `2026-10-01-jev-and-legal-apis.md`.

## TL;DR

- **Jev is served on OpenRouter as `typesafe/jev-1.13`**, with the alias `~typesafe/jev-latest`. It is **not a chat-completions model.** OpenRouter lists its modality as `text->decisions` with `has_text_output: false`. It is reached through two dedicated endpoints:
  - `POST https://openrouter.ai/api/v1/systemone`, which is wire-compatible with TypeSafe's own API.
  - `POST https://openrouter.ai/api/alpha/decisions`.

  Both take the same `{model, state, questions}` body, and both return `Choice`, `Score` and `Noul` answers unchanged.
- **Access and price:** it works with a normal OpenRouter key. OpenRouter says there is "no waitlist or separate TypeSafe account". The price is **$0.042 per million input tokens, and output is free**, the same as TypeSafe direct. Each response includes `usage.cost`.
- **Latency:** OpenRouter's own stats over a 30-minute window of about 2.66M requests show **p50 219 ms, p90 328 ms and p99 about 470 ms**.
- **Context mismatch:** OpenRouter caps context at **32,000 tokens**, while TypeSafe direct allows 64k (32k for state plus the longest question).
- **`langchain-typesafe` works against OpenRouter with no code change.** Set `base_url="https://openrouter.ai/api"` (or `TYPESAFE_BASE_URL`), pass the OpenRouter key as `api_key` (or `TYPESAFE_API_KEY`), and set `model="typesafe/jev-1.13"`. The wrapper posts `{state, model, questions}` to `{base_url}/v1/systemone`. Its response models ignore the extra fields OpenRouter adds (`id`, `provider`, `usage.cost`), so **`usage.cost` is dropped by the wrapper.** `ChatOpenAI` or `ChatOpenRouter` are the wrong tools for Jev itself.
- **Do not confuse `typesafe/jev-router` with Jev.** `typesafe/jev-router` is a *chat-completions router* that uses Jev internally to pick a model for each request. It is not a way to call `Choice`, `Score` or `Noul`.
- **Current OpenRouter IDs:**
  - Claude: `anthropic/claude-haiku-4.5` ($1/$5), `anthropic/claude-sonnet-5.5` ($2/$10) and `anthropic/claude-opus-5.5` ($4/$20).
  - GPT mini/nano class: `openai/gpt-5.4-mini` ($0.75/$4.50) and `openai/gpt-5.4-nano` ($0.20/$1.25).
  - Small open-weight: `qwen/qwen3-30b-a3b-instruct-2507` (~$0.05/$0.19), `google/gemma-4-26b-a4b-it` ($0.09/$0.30) and `mistralai/ministral-8b-2512` ($0.15/$0.15).
  - OpenRouter **does serve embeddings** (33 models, e.g. `baai/bge-m3`, `qwen/qwen3-embedding-8b`, `openai/text-embedding-3-small`) **and rerankers** (9 models, e.g. `cohere/rerank-4-fast`, `cohere/rerank-v3.5`, `qwen/qwen3-reranker-8b`).
- **English legal data:**
  - **Best English-only corpus: Caselaw Access Project static files.** They need no auth, have no rate limit and have no license friction (the HF mirror is CC0). Per-volume JSON carries id, court, decision_date, judges, an author per opinion, full text and citations. Coverage ends in 2018 and has no subject category.
  - **CourtListener:** its 2026 REST limits are now very tight (**5/min, 50/h, 125/day** for authenticated users). Use its quarterly public-domain bulk CSVs (opinions ≈55 GB bz2) instead of the API.
  - **UK Find Case Law** has rich Akoma Ntoso XML, but the Open Justice Licence **excludes "computational analysis" without a separate (free) licence**.
  - **LexGLUE** text has no metadata. **Pile of Law** is CC-BY-NC-SA and has thin metadata.
- **Parallel PT/EN text exists: EUR-Lex/CJEU through the EU Publications Office "Cellar".**
  - One CELEX id returns the English or Portuguese full text by content negotiation (verified live on C-311/18 *Schrems II*).
  - Metadata comes from a public SPARQL endpoint: ECLI, date, court formation, judge, subject matter and procedure.
  - About 7.7k CJEU judgment works since 2015 have both an ENG and a POR expression (SPARQL count, may double-count).
  - **Caveat: this is European Portuguese and EU law, not Brazilian law.** The `eur-lex.europa.eu` HTML site sits behind an AWS WAF challenge, so use Cellar.
- **Recommendation:**
  - **English run:** CAP static JSON, e.g. U.S. Supreme Court (`us/`) volumes plus one or two state reporters, with categories derived via an LLM or Jev `Choice`.
  - **EN-vs-PT comparison:** use CJEU judgments from Cellar so both runs index *the same documents*. Keep STJ as the Brazilian corpus for the headline benchmark.

---

## 1. Jev on OpenRouter

### 1.1 How it is listed

| Fact | Source |
|---|---|
| The default `GET /api/v1/models` list (462 entries, all text/image/audio output) **does not include `typesafe/jev-1.13`**. The only TypeSafe entry there is `typesafe/jev-router` | https://openrouter.ai/api/v1/models (live) |
| `GET /api/v1/models?output_modalities=decisions` returns `typesafe/jev-1.13` and `~typesafe/jev-latest`, both priced `prompt: 0.000000042`, `completion: 0`, `context_length: 32000`. The same filter also lists other "decisions" models (`togethercomputer/tev1-4b-experimental`, `upstage/solar-decide`, `inception/mercury-decide:free`, `respan/span-01*`, `jaredpalmer/kev-4b`) | https://openrouter.ai/api/v1/models?output_modalities=decisions (live) |
| The endpoint record shows provider `TypeSafe`, endpoint name `typesafe/jev-1.13-20260917`, modality `text->decisions`, `output_modalities: ["decisions"]`, `context_length: 32000`, `max_completion_tokens: 28800`, and `supported_parameters: []` | https://openrouter.ai/api/v1/models/typesafe/jev-1.13/endpoints (live) |
| The model page data shows `has_text_output: false`, `quick_start_example_type: "decisions"`, `adapter_name: "TypeSafeDecisionsAdapter"` and upstream `baseUrl: https://api.typesafe.ai/v1`. The data policy is `training: false`, `retainsPrompts: false`. The release date is 2026-09-18 | https://openrouter.ai/typesafe/jev-1.13 (embedded page JSON) |
| **Latency** (30-min window, 2,664,618 requests): p50 219 ms, p75 278 ms, p90 328 ms, p95 364 ms, p99 ≈470 ms. No throughput figure is reported (output is not tokens) | same page, `stats` object, fetched 2026-10-01 |
| "`typesafe/jev-1.13` (or the `~typesafe/jev-latest` alias) is available to anyone with an OpenRouter API key, and billed to your OpenRouter account." "There's no waitlist or separate TypeSafe account." | https://openrouter.ai/docs/guides/community/jev.md |
| Jev "does not produce reasoning traces, explanations, or free-form text. It is not a drop-in replacement for a chat model." | same |
| The provider object has the flag `hasChatCompletions: true`, which contradicts the model-level `has_text_output: false` and the docs. **UNVERIFIED**: what `/chat/completions` actually does with `typesafe/jev-1.13`. The docs never offer it, so treat it as unsupported | https://openrouter.ai/typesafe/jev-1.13 (embedded JSON) |

### 1.2 The two request surfaces

| Surface | Endpoint | Notes |
|---|---|---|
| System One API | `POST https://openrouter.ai/api/v1/systemone` | Implements TypeSafe's request and response shapes. It maps bare IDs: `jev-1.13` becomes `typesafe/jev-1.13`, and `jev-latest` becomes `~typesafe/jev-latest`. IDs that already carry a prefix are used as-is. The response adds `id`, `provider` and `usage.cost`. (https://openrouter.ai/docs/guides/community/typesafe-sdk.md, https://openrouter.ai/docs/api/api-reference/systemone/submit-a-system-one-request.md) |
| Decisions API (alpha) | `POST https://openrouter.ai/api/alpha/decisions` | Uses the same `DecisionsRequest` schema: `model`, `state` (string, object or array), `questions` (a discriminated union on `type`: `noul`, `choice` or `score`), plus optional `provider`, `session_id`, `trace` and `user`. OpenRouter's TS, Python and Go SDKs expose it as `alpha.decisions`. (https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-request.md, https://openrouter.ai/docs/guides/community/jev-tutorial.md) |

- **Live check:** `POST /api/v1/systemone` and `POST /api/alpha/decisions` without a key return **401** `"No cookie auth credentials found"`. An unknown path (`/api/v1/doesnotexist123`) returns **404**, so both routes exist (live curl).
- **Primitive mapping is one-to-one.** The OpenAPI example sends `is_bug` as a `noul` with `criteria {true,false}`, `team` as a `choice` with a `criteria` dict, and `urgency` as a `score` with a `criteria` list. The answers are:
  - `noul`: `{noul}`
  - `choice`: `{choice, probabilities, confidence}`
  - `score`: `{score, legend, probabilities, confidence}`

  These are identical to the `NoulAnswer`, `ChoiceAnswer` and `ScoreAnswer` types in `langchain-typesafe` (systemone OpenAPI above; `langchain_typesafe/types.py`).
- The response `model` is the dated snapshot, e.g. `typesafe/jev-1.13-20260917` (tutorial). **Pin `typesafe/jev-1.13`** so thresholds stay stable (tutorial FAQ).
- **Model ID mismatch:** TypeSafe direct calls the model `jev-1.13.0` (https://docs.typesafe.ai/models.md, per the previous note). OpenRouter documents only `jev-1.13` and `jev-latest` as bare IDs. **UNVERIFIED** whether `jev-1.13.0` is accepted on OpenRouter, so use `typesafe/jev-1.13`.
- **Context mismatch:** OpenRouter says "32,000 tokens. That's the `state` you send plus the questions" (jev.md FAQ). TypeSafe direct says 64k per request, with 32k for state plus the longest question (models.md, per the previous note). **Plan for 32k on OpenRouter.**
- **Rate limits on OpenRouter for Jev:** `limit_rpm` and `limit_rpd` are `null` on the model page. OpenRouter's account-level limits were not checked (**UNVERIFIED**).

### 1.3 `langchain-typesafe` against OpenRouter

From the `langchain-typesafe==0.0.1a3` wheel (latest on PyPI, https://pypi.org/pypi/langchain-typesafe/json):

- `base_url: str = Field(default_factory=from_env("TYPESAFE_BASE_URL", default="https://api.typesafe.ai"))`. The docstring says: "Override it for a compatible gateway, test server, or private deployment." The endpoint is `f"{base_url.rstrip('/')}/v1/systemone"` (`classifier.py` lines 188–200 and 427).
- `_payload` sends exactly `{"state", "model", "questions"}` (`classifier.py` line 442), which matches OpenRouter's required fields.
- `ClassifierResponse` and `Usage` are plain pydantic models with the default `extra="ignore"` (only the *classifier config* has `extra="forbid"`). OpenRouter's extra `id`, `provider` and `usage.cost` fields therefore parse without error but are **discarded**. `request_id` is read from the `x-typesafe-request-id` header. **UNVERIFIED** whether OpenRouter forwards that header (expect `None`).
- The `model` validator only strips whitespace, so `typesafe/jev-1.13` is accepted.
- OpenRouter documents the same base-URL swap for the first-party `typesafe_sdk` and `@typesafe-ai/sdk` (`base_url="https://openrouter.ai/api"`). It also warns that the SDK's `models.list()` breaks, because `GET /api/v1/models` returns OpenRouter's shape (typesafe-sdk.md).

```python
from langchain_typesafe import TypeSafeClassifier
clf = TypeSafeClassifier(
    model="typesafe/jev-1.13",
    base_url="https://openrouter.ai/api",      # -> POST https://openrouter.ai/api/v1/systemone
    api_key=os.environ["OPENROUTER_API_KEY"],
).with_retry()                                  # wrapper has no retries (see previous note)
```

**Implication for the benchmark:** you need `usage.cost` to report Level 3 cost. Either compute it as `input_tokens × $0.042/M`, or call `/api/v1/systemone` or `/api/alpha/decisions` with `httpx` directly to keep `usage.cost`. One OpenRouter key then covers Jev, System 2 models, embeddings and reranking, which keeps the Level 2 vs Level 3 cost accounting in one ledger.

### 1.4 `typesafe/jev-router` (not the classifier)

- "picks a model and reasoning effort for each request… Set `model` to `typesafe/jev-router`… works with Chat Completions, Responses, and Messages". The candidate pool can be restricted with `plugins: [{id: "jev-router", models: [...], excluded_models: [...]}]` (https://openrouter.ai/docs/guides/routing/routers/jev-router.md).
- It is listed with `tokenizer: "Router"` and pricing `-1` (i.e. the routed model's price, **UNVERIFIED**). `/endpoints` returns `[]` (https://openrouter.ai/api/v1/models/typesafe/jev-router/endpoints).
- It could serve as an *off-the-shelf* baseline for "Jev picks the System 2 model", but it hides the decision. It is not usable for chunk filtering, guardrails or judging.

### 1.5 Other OpenRouter model IDs (live `/api/v1/models`, prices in $/M tokens, input/output)

| Role | ID | Price | Context | Notes |
|---|---|---|---|---|
| Claude Haiku | `anthropic/claude-haiku-4.5` (alias `~anthropic/claude-haiku-latest`) | 1 / 5 | 200k | |
| Claude Sonnet | `anthropic/claude-sonnet-5.5` (alias `~anthropic/claude-sonnet-latest`) | 2 / 10 | 1M | `sonnet-5`, `sonnet-4.6` and `sonnet-4.5` are also listed |
| Claude Opus | `anthropic/claude-opus-5.5` (alias `~anthropic/claude-opus-latest`) | 4 / 20 | 1M | `anthropic/claude-fable-5.1` is 10 / 50 |
| GPT mini | `openai/gpt-5.4-mini` (alias `~openai/gpt-mini-latest`) | 0.75 / 4.5 | 400k | `openai/gpt-5-mini` is 0.25 / 2 |
| GPT nano | `openai/gpt-5.4-nano` | 0.20 / 1.25 | 400k | `openai/gpt-5-nano` is 0.05 / 0.40. The tier of the newer `openai/gpt-5.6-luna` (0.20 in) is **UNVERIFIED** |
| Open-weight small | `qwen/qwen3-30b-a3b-instruct-2507` (HF `Qwen/Qwen3-30B-A3B-Instruct-2507`) | ~0.048 / 0.193 | 262k | |
| Open-weight small | `google/gemma-4-26b-a4b-it` (HF `google/gemma-4-26B-A4B-it`; a `:free` variant exists) | 0.09 / 0.30 | 262k | |
| Open-weight small | `mistralai/ministral-8b-2512` (HF `mistralai/Ministral-3-8B-Instruct-2512`) | 0.15 / 0.15 | 262k | |
| Open-weight small | `nvidia/nemotron-3-nano-30b-a3b` | 0.05 / 0.20 | 262k | |

All the IDs above list `structured_outputs` in `supported_parameters` (live). Many also have a `:batch` variant at 50% price.

**Embeddings:** `GET /api/v1/embeddings/models` returns 33 models (live). PT-capable options include:

| Model | $/M tokens | Context |
|---|---|---|
| `baai/bge-m3` | 0.01 | 8k |
| `qwen/qwen3-embedding-8b` | 0.01 | 32k |
| `qwen/qwen3-embedding-4b` | 0.02 | 32k |
| `intfloat/multilingual-e5-large` | 0.01 | 512 |
| `openai/text-embedding-3-small` | 0.02 | n/a |
| `openai/text-embedding-3-large` | 0.13 | n/a |
| `google/gemini-embedding-001` | 0.15 | n/a |
| `voyageai/voyage-4` | 0.06 | n/a |

The endpoint is `POST /api/v1/embeddings` (https://openrouter.ai/docs/api_reference/embeddings.md).

**Rerank:** `POST /api/v1/rerank` (live 401 without a key; https://openrouter.ai/docs/api/api-reference/rerank/submit-a-rerank-request.md). `?output_modalities=rerank` lists `cohere/rerank-v3.5`, `cohere/rerank-4-pro`, `cohere/rerank-4-fast`, `qwen/qwen3-reranker-8b`, `voyageai/rerank-2.5(-lite)`, `voyageai/rerank-3(-lite)` and `nvidia/llama-nemotron-rerank-vl-1b-v2:free`. Per-token pricing shows `0`, which suggests per-search billing. **UNVERIFIED**: the actual rerank price. OpenRouter's RAG cookbook uses `cohere/rerank-v3.5` (https://openrouter.ai/docs/cookbook/evaluate-and-optimize/rag.md).

---

## 2. English-language open legal data

Target schema: id, court, decision date, judge/author, full text, category.

### 2.1 CourtListener (Free Law Project)

**Auth and rate limits**
- REST v4 uses token, session or basic auth. **"By default, authenticated users may make up to 5 requests per minute, 50 requests per hour, and 125 requests per day."** Limits are rolling windows, and higher limits come through FLP membership or a commercial agreement (https://www.courtlistener.com/help/api/rest/, v4.7).
- `GET /api/rest/v4/opinions/` without a token returns **401** "Authentication credentials were not provided." (live)
- `GET /api/rest/v4/search/?type=o&q=…` **works anonymously** (live, HTTP 200, `count: 109095` for "fourth amendment"). The anonymous search quota is **UNVERIFIED**.

**Search result shape** (live): `caseName`, `cluster_id`, `court`, `court_id`, `dateFiled`, `docketNumber`, `judge`, `citation[]`, `status`, and `opinions[{id, author_id, type, snippet, download_url, local_path, cites[]}]`. Search returns **snippets only**. Full text needs `/opinions/{id}/` (auth) or the bulk files.

**Full text**
- Docs: "prefer the `html_with_citations` field instead of `plain_text`". Text sources vary by origin (`html_columbia`, `html_lawbox`, `xml_harvard` (CAP OCR), `plain_text` from PDFs) (https://www.courtlistener.com/help/api/rest/v4/case-law/).
- Opinions sit under clusters, which hold judges, date and case name; clusters sit under dockets, which hold the court.

**Bulk**
- PostgreSQL `COPY` CSV snapshots, regenerated quarterly. They are in the public bucket `s3://com-courtlistener-storage/bulk-data/` (https://www.courtlistener.com/help/api/bulk-data/).
- Live listing: `opinions-2026-09-30.csv.bz2` is **55.25 GB** and `opinion-clusters-2026-09-30.csv.bz2` is **2.47 GB** (https://com-courtlistener-storage.s3-us-west-2.amazonaws.com/?list-type=2&prefix=bulk-data/opinions-).
- About 2 TB of case-law embeddings are also published (`s3://com-courtlistener-storage/embeddings/opinions/`).

**License:** "Our bulk data files are free of known copyright restrictions", marked with the Public Domain Mark 1.0 (bulk-data page). The `/terms/` page returned a CloudFront 403 to curl, so the API terms of use are **UNVERIFIED**.

**Category:** none for opinions. `nature_of_suit` exists only on RECAP dockets (case-law doc example).

**Gotchas**
- The API quota is too small to build a corpus. Bulk is all-or-nothing (55 GB compressed for opinions); there is no per-court slice.
- Joining opinions, clusters, dockets and courts is required to fill the schema.

### 2.2 Caselaw Access Project (Harvard LIL), static.case.law

**Auth, rate limits and license**
- Plain static files, no auth, no API.
- "Those agreements have now expired, and there is no longer a need to have any form of metered access to the data" (https://case.law/about/, rendered from `/templates/cap-about-page.js`).
- CAP points to CourtListener for API access (`/templates/cap-docs-page.js`).
- The license is **not stated on static.case.law** (**UNVERIFIED** for the static files). The HF mirror `free-law/Caselaw_Access_Project` is **`cc0-1.0`** but is **gated: manual** (https://huggingface.co/api/datasets/free-law/Caselaw_Access_Project).

**Layout** (live)
- `https://static.case.law/` lists reporter slugs (`us`, `a2d`, …).
- Each volume (e.g. `https://static.case.law/us/500/`) has `cases/NNNN-NN.json`, `html/`, `case-pdfs/`, `VolumeMetadata.json` and `CasesMetadata.json`.
- Root-level `ReportersMetadata.json` exists. `us/VolumesMetadata.json` has 572 volumes.

**Case JSON shape** (live, `https://static.case.law/us/500/cases/0001-01.json`)
- `id` (6215164), `name`, `name_abbreviation`, `decision_date` ("1991-04-24"), `docket_number`, `citations[]` ("500 U.S. 1", …), `court{name, id}` ("Supreme Court of the United States"), `jurisdiction`, `cites_to[]` (with categories such as `laws:leg_statute`), `analysis`, `provenance`.
- `casebody{judges[], parties[], attorneys[], head_matter, opinions[{type, author, text}]}`.
- In this sample: `majority` by "JUSTICE BLACKMUN" (15,962 chars) and `concurring-in-part-and-dissenting-in-part` by "Justice Stevens,".

**Coverage:** published cases through **2018** only. Text is "raw OCR output", while metadata (name, citation, court, date) was corrected (about page).

**Gotchas**
- OCR noise ("Sou-PER, JJ.").
- `author` strings are unnormalised.
- No subject category, so derive `category` yourself (e.g. a Jev `Choice` over a fixed taxonomy).
- Nothing after 2018.

**Fit: best for a PoC.** Pick N volumes and you get every target field except category, with zero auth or quota friction.

### 2.3 EUR-Lex (English) via the EU Publications Office "Cellar"

**Auth and WAF**
- `https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:…` returns **HTTP 202 with `x-amzn-waf-action: challenge`** and an empty body to curl, even with a browser User-Agent (live).
- The legal-notice and webservice help pages are blocked the same way. **Do not scrape eur-lex.europa.eu.**

**Cellar works without auth** (live)
- **Metadata:** SPARQL at `https://publications.europa.eu/webapi/rdf/sparql`. A query on `cdm:resource_legal_id_celex "62018CJ0311"` returns the work URIs.
- **Full text:** content negotiation on `http://publications.europa.eu/resource/celex/{CELEX}` with `Accept: text/html, application/xhtml+xml` and `Accept-Language: eng|por` redirects to the expression manifestation and returns XHTML.
  - English: 346 KB, starts "JUDGMENT OF THE COURT (Grand Chamber) 16 July 2020".
  - Portuguese: 366 KB, starts "ACÓRDÃO DO TRIBUNAL DE JUSTIÇA (Grande Secção) 16 de julho de 2020".

**Metadata available** (CDM predicates on the C-311/18 work, live)

| Field | Predicate and value |
|---|---|
| ECLI | `case-law_ecli` (`ECLI:EU:C:2020:559`) |
| Date | `work_date_document` (2020-07-16) |
| Court formation | `case-law_delivered_by_court-formation` (`CHAMB_GD_C`, Grand Chamber) |
| Judge | `case-law_delivered_by_judge` (a Cellar agent URI that must be resolved to a name) |
| Advocate General | `case-law_delivered_by_advocate-general` |
| Subject | `case-law_is-about_case-law-subject-matter` (`RAPL`) and `resource_legal_is_about_subject-matter` |
| Procedure | `case-law_has_type_procedure_concept_type_procedure` (`PREJ`) |
| Other | `case-law_originates_in_country` (IRL), `resource_legal_type` (`CJ`), `work_has_resource-type` (`JUDG`) |

**Volume:** 7,664 distinct `CJ` works dated ≥2015 have both a POR and an ENG expression (live SPARQL `COUNT`). This may double-count, because C-311/18 maps to two work URIs, so the exact number of unique judgments is **UNVERIFIED**.

**License**
- The Commission's reuse Decision 2011/833/EU, Art. 4: "All documents shall be available for reuse… for commercial or non-commercial purposes… without charge… without the need to make an individual application". Conditions may include acknowledging the source (Art. 6) (fetched via Cellar, `http://publications.europa.eu/resource/celex/32011D0833`).
- The Publications Office's own site content is CC BY 4.0 and defers to EUR-Lex's copyright notice for EUR-Lex content (https://op.europa.eu/en/web/about-us/legal-notices/publications-office-of-the-european-union-copyright).
- **UNVERIFIED:** the EUR-Lex-specific notice (WAF-blocked) and the CJEU's own reuse terms for its judgments, which are Court documents, not Commission documents.

**Gotchas**
- SPARQL plus CDM ontology has a learning curve.
- Judge names need a second lookup.
- The judgment XHTML is long (around 170k chars of text), so chunking matters.
- Rate limits for Cellar and SPARQL are not documented (**UNVERIFIED**).
- The **official EUR-Lex SOAP webservice requires registration** (per general knowledge, **UNVERIFIED**: its help page is WAF-blocked).

### 2.4 UK National Archives: Find Case Law

**Auth and rate limits**
- No auth. "You can make up to 1,000 requests (of any type) to Find Case Law in any rolling five minute period per IP address", with HTTP 429 beyond that (https://nationalarchives.github.io/ds-find-caselaw-docs/public).
- OpenAPI 0.6.0: https://raw.githubusercontent.com/nationalarchives/ds-find-caselaw-docs/refs/heads/main/doc/openapi/public_api.yml

**Endpoints** (live)
- `GET /atom.xml` takes the same params as advanced search: `query`, `court`, `per_page`, `page`, etc. Unfiltered, the feed paginates to page 8,210 at `per_page=2`, so about 16.4k entries.
- Entries carry title, published date, author (= court or tribunal), `tna:identifier`, `tna:contenthash`, and links to `/data.xml` (Akoma Ntoso) and a PDF.

**Document shape** (live, `https://caselaw.nationalarchives.gov.uk/uksc/2026/34/data.xml`, 252 KB)
- `FRBRname` (case name), `FRBRdate name="judgment"` (2026-09-22), `uk:court` (UKSC), `uk:cite` / `neutralCitation` ("[2026] UKSC 34"), `uk:year`.
- Judges appear as `TLCPerson eId="judge-…" showAs="Lord Briggs"`. Citations are marked up as `<ref uk:canonical="[2005] UKHL 51">`.
- Full text is in the AKN body.
- **No subject category field** found in this sample (no `uk:category` / `uk:jurisdiction` tags).

**License: Open Justice Licence v2.0.** It permits commercial reuse, **but** "You need additional permission (a separate licence) for computational analysis of Find Case Law judgments (including indexing by search engines)". Programmatic bulk searching "to identify, extract or enrich contents" needs an application, and "there are no application charges" (https://caselaw.nationalarchives.gov.uk/open-justice-licence, https://caselaw.nationalarchives.gov.uk/re-use-find-case-law-records). **Embedding and indexing judgments for RAG very likely counts as computational analysis. Apply before use.**

**Gotchas**
- Coverage is mostly 2001 onwards (**UNVERIFIED**, not checked).
- XML namespaces need care.
- The judges list mixes parties and judges in `TLCPerson`; filter on the `judge-` prefix.

### 2.5 Hugging Face datasets

| Dataset | License | Shape | Fit |
|---|---|---|---|
| `coastalcph/lex_glue` | CC-BY-4.0 | Configs: `scotus` (7.8k), `ecthr_a`/`b` (11k), `eurlex` (65k), `ledgar` (80k), `unfair_tos` (9.4k), `case_hold` (52.5k). `scotus` rows are just `{text, label}`, with no id, date or judge (datasets-server live) | Poor for RAG metadata; fine as a classification eval for Jev `Choice` |
| `pile-of-law/pile-of-law` | **CC-BY-NC-SA-4.0** | Fields `text, created_timestamp, downloaded_timestamp, url`. `courtListener_opinions` is a CourtListener sync "as of 12/31/2022", sharded into ~1 GB `.jsonl.xz` files. `created_timestamp` is the upload time, not the decision date (README) | Poor: thin metadata, NC license |
| `free-law/Caselaw_Access_Project` | CC0-1.0, **gated (manual)**, parquet, 1M–10M rows | CAP in parquet | Good if access is granted; otherwise use static.case.law |
| `coastalcph/multi_eurlex` | CC-BY-SA-4.0 | 65k EU laws in 23 languages **including `pt`**, with an `all_languages` config and EUROVOC labels as category. Splits are 55k / 5k / 5k by date. Uses a loading script, so the dataset viewer is unsupported (README; datasets-server) | **Parallel PT/EN legislation** (not case law) with a ready-made category label |

Sources: https://huggingface.co/api/datasets/{name}; https://huggingface.co/datasets/coastalcph/multi_eurlex/raw/main/README.md; https://huggingface.co/datasets/pile-of-law/pile-of-law/raw/main/README.md

### 2.6 Parallel PT/EN legal text

- **CJEU judgments via Cellar** (§2.3): the same CELEX id gives both languages, with full metadata. These are **true parallel decisions**, which is the cleanest way to compare English and Portuguese runs on identical content.
- **MultiEURLEX** (§2.5): parallel EU *legislation* with EUROVOC categories, and easy to load.
- **Caveats:**
  - Both use **European Portuguese** ("Secção", "pessoas singulares") and EU-law vocabulary, not Brazilian legal Portuguese.
  - They test language effects on Jev and the retriever, not domain transfer to STJ.
  - No Brazilian (STF/STJ) source with systematic English translations was found. **UNVERIFIED** that none exists; STF publishes only ad-hoc English material, which was not checked.

### 2.7 Fit against the target schema

| Field | CourtListener | CAP static | Cellar (CJEU) | UK FCL | LexGLUE scotus |
|---|---|---|---|---|---|
| id | opinion/cluster id | `id` | CELEX / ECLI | `uk:cite`, `tna:uri` | — |
| court | via docket → court | `court.name` | `court-formation` | `uk:court` | — |
| decision date | cluster `date_filed` | `decision_date` | `work_date_document` | `FRBRdate judgment` | — |
| judge/author | opinion `author_id`, cluster `judges` | `opinions[].author`, `judges[]` | `delivered_by_judge` (URI) | `TLCPerson judge-*` | — |
| full text | `html_with_citations` (auth/bulk) | `opinions[].text` (OCR) | XHTML per language | AKN body | `text` |
| category | — | — (derive) | subject-matter codes | — | `label` (issue area) |
| access | 125 req/day, or 55 GB bulk | static, no auth | SPARQL + HTTP, no auth | 1,000 req/5 min | HF |
| license | PD mark (bulk) | CC0 (HF mirror) | EU reuse, attribution | OJL + **computational licence** | CC-BY-4.0 |

### 2.8 Recommendation

1. **English corpus: CAP static JSON.**
   - Take a bounded slice, e.g. 10–20 volumes of `us/` (U.S. Supreme Court, roughly 700 cases per volume) and optionally one state reporter.
   - Map `id`, `court.name`, `decision_date`, `opinions[].author` and `opinions[].text` directly. Derive `category` with a Jev `Choice` over a small taxonomy, which also gives a quick English-language Jev check.
   - There are no quota or license blockers.
   - Use CourtListener only to look up post-2018 cases one at a time, since its API quota rules it out for corpus building.
2. **EN-vs-PT comparison: CJEU judgments from Cellar.**
   - Select a few hundred CELEX ids (e.g. `CJ` judgments 2018–2024 that have both an ENG and a POR expression) and pull both XHTML versions.
   - Build two collections with identical metadata. The golden set can then be translated question-for-question, and the only variable is language.
   - Report the European-Portuguese caveat.
3. **Keep STJ Íntegras as the PT-BR headline corpus** (previous note). Treat CJEU EN/PT as a controlled language ablation, not a replacement.
4. **Skip UK FCL** unless you apply for the free computational-analysis licence. **Skip LexGLUE and Pile of Law** for RAG, because of missing metadata and the NC license respectively.
5. **Jev transport:** call Jev through OpenRouter (`typesafe/jev-1.13`). Use `langchain-typesafe` with `base_url="https://openrouter.ai/api"`, or raw `/api/v1/systemone` if you want `usage.cost`. That way one key and one bill covers Jev, System 2 (Claude/GPT), embeddings (`baai/bge-m3` or `qwen/qwen3-embedding-8b` for PT) and rerank (`cohere/rerank-4-fast`).
