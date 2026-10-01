# Research: Jev / `langchain-typesafe` and Brazilian open legal data APIs

Date: 2026-10-01. Method: PyPI metadata + downloaded wheel source, first-party docs, live `curl` calls (small requests, no sign-up). Every claim cites its source; anything not confirmed against a primary source is marked **UNVERIFIED**.

## TL;DR

- **Jev is real.** TypeSafe AI ("Made in SF") launched Jev on 2026-09-28 as a "System One model". `langchain-typesafe` is an official LangChain partner package that lives in the `langchain-ai/langchain` monorepo (`libs/partners/typesafe`). It is at **0.0.1a3 (alpha), released 2026-09-20, and marked `@beta`**.
- **The primitives are `Choice`, `Score` and `Noul`. There is no `Nule`, so PROJECT_CONTEXT.md has it wrong.** The entry point is `TypeSafeClassifier`, a LangChain Runnable. Experimental middlewares `ModelRouterMiddleware` and `AutoModeMiddleware` cover the doc's "routing" and "Auto-mode guardrail" ideas almost one-to-one.
- **Access:** you need a `TYPESAFE_API_KEY`, and the service is in early access, letting developers in from a waitlist. Price is **$0.042 per million input tokens; output is free.** Limits are 40 req/s and 100K tok/s, and they "can change without notice". The context window is 64k tokens (32k for state plus the longest question).
- **The "20x–200x faster, 40x–400x cheaper" figure is UNVERIFIED as written.** TypeSafe's own numbers are "40x-200x faster" (blog), "193.6x faster, 444.6x cheaper" (homepage, called "on the higher end of real world gains" by TypeSafe itself) and "238x lower input price than Claude Fable 5.1" (homepage).
- **Big risk for this PoC:** TypeSafe says "English is the primary training language… other languages… not equally well". Our corpus is Brazilian Portuguese legal text, so we must benchmark before relying on Jev.
- **Legal data recommendation:** the primary corpus should be the **STJ Dados Abertos "Íntegras de decisões terminativas e acórdãos"** dataset. It has full text plus metadata (relator, date, process, CNJ subject codes), is CC-BY and needs no auth. Enrich it with **STJ "Espelhos de Acórdãos"** (ementa, órgão julgador, tese) and optionally **DataJud** (metadata only). **Querido Diário** is a good second corpus for municipal gazettes (full text via `.txt`, CC-BY). **STF** has no usable public API: the jurisprudence portal sits behind an AWS WAF challenge.

---

## 1. Jev / `langchain-typesafe`

### 1.1 Existence and publisher

| Fact | Source |
|---|---|
| `langchain-typesafe` on PyPI. Versions 0.0.1a1 and 0.0.1a2 came out 2026-09-17 and 0.0.1a3 on 2026-09-20. MIT license, Python >=3.10,<4 | https://pypi.org/pypi/langchain-typesafe/json |
| Repository is `langchain-ai/langchain`. The package sits at `libs/partners/typesafe`, next to openai and anthropic | https://api.github.com/repos/langchain-ai/langchain/contents/libs/partners |
| Deps: `httpx2>=2,<3` (**httpx2, not httpx**) and `langchain-core>=1.6.2,<2`. The `[experimental]` extra adds `langchain>=1.3.15` | wheel METADATA (`pip download langchain-typesafe==0.0.1a3`) |
| LangChain provider page | https://docs.langchain.com/oss/python/integrations/providers/typesafe |
| TypeSafe AI is the vendor. Jev is its "flagship model and the first System One model", trained with "Reinforcement Learning for Calibrated Decisions (RLCD)" | https://docs.typesafe.ai/llms.txt, https://typesafe.ai/blog/introducing-system-one-models-and-jev |
| A first-party SDK also exists: `typesafe-sdk` (PyPI, 0.7.2), plus a JS SDK `@typesafe-ai/sdk` | `pip index versions typesafe-sdk`; https://docs.typesafe.ai/models.md |
| Community repos already use it, e.g. `saksham-malhotra-27/jev-relevance` ("relevance filter for LangChain RAG retrievers") | GitHub search API `q=langchain-typesafe` |

### 1.2 API surface (from the 0.0.1a3 wheel source)

Public exports (`langchain_typesafe/__init__.py`): `TypeSafeClassifier`, `Choice`, `Score`, `Noul`, `NoulCriteria`, `ClassifierRequest`, `ClassifierResponse`, `ChoiceAnswer`, `ScoreAnswer`, `NoulAnswer`, `Question`, `Answer`, `State`, `Usage`.

```python
# langchain_typesafe/classifier.py
_DEFAULT_BASE_URL = "https://api.typesafe.ai"   # override: TYPESAFE_BASE_URL
_DEFAULT_MODEL = "jev-latest"
_DEFAULT_TIMEOUT = 30.0

@beta()
class TypeSafeClassifier(RunnableSerializable[ClassifierRequest, ClassifierResponse]):
    model: str = "jev-latest"
    api_key: SecretStr          # default: env TYPESAFE_API_KEY
    base_url: str               # POST {base_url}/v1/systemone
    timeout: float = 30.0
    client: httpx2.Client | None
    async_client: httpx2.AsyncClient | None
    def invoke(self, input: ClassifierRequest, config=None) -> ClassifierResponse
    async def ainvoke(self, input: ClassifierRequest, config=None) -> ClassifierResponse
```

```python
# langchain_typesafe/types.py
class Noul(BaseModel):        # binary -> P(yes)
    type: Literal["noul"]; instructions: str | dict | list; criteria: NoulCriteria | None
class NoulCriteria(BaseModel): true: JsonValue = None; false: JsonValue = None
class Choice(BaseModel):      # categorical
    type: Literal["choice"]; criteria: dict[str, JsonValue]  # label -> description (min 1)
    instructions: ...
class Score(BaseModel):       # ordinal rubric, levels 0..n-1
    type: Literal["score"]; criteria: list[JsonValue]  # min 2
    instructions: ...
class ClassifierRequest(TypedDict):
    state: State                       # str | JSON | BaseMessage(s), nested OK
    questions: dict[str, Question]
# Answers
NoulAnswer:   noul: float [0,1]                       # no confidence field
ChoiceAnswer: choice: str; probabilities: dict[str,float]; confidence: float
ScoreAnswer:  score: float (expected value, may be fractional); legend; probabilities; confidence
ClassifierResponse: model, answers, usage(input_tokens, output_tokens), request_id
                    + .nouls / .choices / .scores views
```

- Wire payload is `{"state":…, "model":…, "questions":{id: question}}` sent with `Authorization: Bearer <key>` (`classifier.py` `_payload` and `_request_headers`; cross-checked against https://docs.typesafe.ai/api.md).
- **The LangChain wrapper does not retry.** `classifier.py` contains no retry logic. The first-party `typesafe-sdk` has a `RetryPolicy` (https://docs.typesafe.ai/sdk/python/api/retries.md). With the LangChain wrapper, add `.with_retry()` to handle 429 and 529.
- Errors are `TypeSafeRateLimitError` (exposing `retry_after_ms` and `request_id`) and `TypeSafeAuthenticationError`. They also subclass LangChain's `ModelRateLimitError` and `ModelAuthenticationError` (`client.py`).
- Experimental (`langchain_typesafe.experimental.middleware`, "may change without notice"):
  - `ModelRouterMiddleware(choices={name: ModelChoice(model=..., criteria=...)}, instructions=...)`. It classifies the latest human message once per agent run and stores the `ChoiceAnswer` in state under `model_route`.
  - `AutoModeMiddleware(tools=[...], criteria=NoulCriteria(true=..., false=...), instructions=..., threshold=0.5)`. It runs a `Noul` named `is_risky` on configured tool calls. A call at or above the threshold returns an error `ToolMessage` (`auto_mode.py`, where `_PROBABILITY_THRESHOLD = 0.5`).
- Live check: `POST https://api.typesafe.ai/v1/systemone` with no key returns **HTTP 403** `{"detail":{"error_type":"authentication_error","message":"Must supply an API key!…"}}`. The docs list 401 for this case (https://docs.typesafe.ai/api.md), so the endpoint is real but behaves slightly differently from the docs.

**How this maps onto PROJECT_CONTEXT.md:**

| Context doc idea | Actual primitive or feature |
|---|---|
| Routing (`Choice`) | `Choice`, or `ModelRouterMiddleware` |
| Chunk filtering (`Score` and "`Nule`") | `Score` and **`Noul`**. TypeSafe's cookbook "Classifying RAG passages" sends **one request per query–passage pair** carrying 4 Nouls (relevant / usable evidence / contradicts premise / prompt injection), then routes in code (https://docs.typesafe.ai/cookbooks/classifying_rag_passages.md). Do it in parallel with `classifier.abatch([...])`. |
| Guardrail "Auto-mode" | `AutoModeMiddleware`, and the "Guardrails for LLMs" cookbook (https://docs.typesafe.ai/cookbooks/llm_guardrails.md) |
| Jev-as-a-Judge (groundedness and citations) | "Double-checking citations" cookbook, which uses a `Choice` (https://docs.typesafe.ai/cookbooks/citation_check.md) |
| Reranking | "Re-ranking" cookbook on CLERC legal queries: top-1 accuracy went from 5% to 18% and top-10 from 38% to 62% (https://docs.typesafe.ai/llms.txt) |

### 1.3 Access, pricing, limits (https://docs.typesafe.ai/models.md)

- Model `jev-1.13.0`. The aliases `jev-latest` and `jev-preview` both point to 1.13.0. **Pin `jev-1.13.0` for reproducible benchmarks.**
- Price is **$42 per billion input tokens ($0.042 per million). Output tokens are free.**
- Rate limits are 100K tokens/s and 40 requests/s. TypeSafe warns: "Rate limits are adjusting dynamically… can change without notice."
- Context is 64k tokens per request, with 32k for `state` plus the longest question. Input is text only.
- "English is the primary training language and where accuracy is currently best. Other languages… not equally well; test on your own content." **This is critical for PT-BR legal text.**
- Accuracy drops as `state` grows with unrelated content. Jev cannot count reliably (https://docs.typesafe.ai/model-jaggedness/jev-1.13.md).
- Jev is not fine-tunable: the same weights serve every account. It is not trained on customer requests. ZDR is for enterprise only (models.md).
- Access is "early access… bringing developers off the waitlist" (https://typesafe.ai/blog/introducing-system-one-models-and-jev). Keys come from https://console.typesafe.ai/keys (docs quickstart). Free tier or credits: **UNVERIFIED** (not stated).

### 1.4 Speed and cost claims

| Claim | Source |
|---|---|
| "End-to-end response time is 70ms-500ms… 40x-200x faster for the same levels of frontier intelligence for System One shaped queries", against "3 to 329 seconds for frontier models" | blog (above) |
| "193.6x Faster, 444.6x Cheaper. *based on workflows for System One tasks", with a demo of $0.000081 / 0.114s vs $0.013880 / 8.566s | https://typesafe.ai homepage |
| TypeSafe on those homepage numbers: "we expect that these are on the higher end of real world gains". Also: "We can't prove it isn't subsidized" | blog |
| "238x Lower input price than Claude Fable 5.1" | homepage |
| Batching 13 questions into one call is "12.2x cheaper and 10.0x faster" | https://docs.typesafe.ai/cookbooks/parallel_questions.md (via llms.txt) |
| **"20x–200x faster, 40x–400x cheaper"** (PROJECT_CONTEXT.md) | **UNVERIFIED.** The phrase is not on any TypeSafe page I fetched. It shows up only in secondary press coverage. Cite the blog or homepage numbers instead. |

### 1.5 Fallback if Jev access or PT-BR quality fails

Not researched in depth; these are the obvious drop-ins for the same graph nodes:
- **Routing, guardrail, judge:** a small chat model with `with_structured_output(...)` in LangChain, using Pydantic `Literal` labels. You lose calibrated probabilities but keep the typed contract.
- **Chunk filtering and reranking:** a cross-encoder such as `bge-reranker` via `sentence-transformers`, or Cohere Rerank. Both are already in the stack in PROJECT_CONTEXT.md.
- Design the Level-3 nodes behind one interface (`decide(state, questions) -> typed answers`) so Jev and the fallback can be swapped and benchmarked against each other. That comparison is the PoC's point anyway.

---

## 2. Brazilian open legal data

### 2.1 Querido Diário (Open Knowledge Brasil): municipal official gazettes

- **Base URL:** `https://api.queridodiario.org.br`. The old `api.queridodiario.ok.org.br` **failed the TLS handshake** from macOS LibreSSL, and `queridodiario.ok.org.br/api/docs` returns 302 to `https://api.queridodiario.org.br/docs`. OpenAPI 0.19.0 is at `/openapi.json`. (live curl)
- **Auth:** public read endpoints need none. Only `/scraper/*` writes need an `X-API-Key` header (OpenAPI `securitySchemes`). **Rate limit:** none documented, and the service is behind Cloudflare. **UNVERIFIED.**
- **Key endpoints:**
  - `GET /gazettes`. Params: `territory_ids` (7-digit IBGE), `published_since` and `published_until`, `querystring` (OpenSearch simple query string), `excerpt_size`, `number_of_excerpts`, `size`, `offset`, `sort_by` (relevance, descending_date or ascending_date).
  - `GET /gazettes/by_theme/{theme}`. Only 2 themes exist: "Políticas Ambientais" and "Tecnologias na Educação".
  - `GET /cities`, `GET /cities/{ibge}`.
  - `GET /aggregates/{UF}?territory_id=`, which gives yearly ZIP bulk files (e.g. São Paulo 2025, 22.89 MB).
- **Response shape** (live): `{total_gazettes, gazettes:[{territory_id, territory_name, state_code, date, scraped_at, edition, is_extra_edition, url(.pdf), txt_url(.txt), excerpts[]}]}`.
- **Full text:** the JSON gives **excerpts only**. Full text comes from `txt_url` on `data.queridodiario.ok.org.br`: HTTP 200, `binary/octet-stream`, **3.68 MB for one SP gazette**.
- **No** relator, órgão or legal-category fields. A gazette is a whole-day document mixing many acts, so you have to split it into acts yourself.
- **Gotchas:**
  - The São Paulo `.txt` sampled begins with a list of links rather than act text, so content quality varies by city.
  - The `aggregates.file_path` value is malformed (`data.queridodiario.ok.org.braggregates/…`, with a missing `/`).
  - `/cities` lists all 5,570 municipalities, but 4,615 are at openness level "0". The meaning of the levels isn't documented in the API, so actual coverage is **UNVERIFIED** (probably a few hundred cities).
  - With an empty `querystring` the API returns metadata only and no excerpts.
- **License:** data is **CC-BY 4.0** (https://docs.queridodiario.ok.org.br/pt-br/latest/). Code is MIT (GitHub `okfn-brasil/querido-diario-api`).
- **Fit:** good for an "administrative or municipal law" corpus. It is weak for the jurisprudence metadata the context doc wants (relator, órgão, categoria).

### 2.2 STF: Supremo Tribunal Federal

- **No documented public API.**
  - `https://jurisprudencia.stf.jus.br/pages/search` and its backing `.../api/search/search` both return **HTTP 202 with header `x-amzn-waf-action: challenge`** (AWS WAF bot challenge) to `curl` (live). Programmatic access means getting past an anti-bot challenge, so it is **not suitable**.
  - The Corte Aberta program (https://portal.stf.jus.br/hotsites/corteaberta/) offers Qlik statistical dashboards (https://transparencia.stf.jus.br/extensions/corte_aberta/corte_aberta.html). The "Dados Abertos" card on that page is hidden, with the HTML comment `TODO: remover style="display: none;" quando o recurso de Dados Abertos estiver disponível`, so **bulk open data is not launched yet**.
  - Whether the Qlik panel exports contain decision text: **UNVERIFIED** (almost certainly statistics or metadata only).
- **DataJud does not cover STF.** Its 91 aliases include `stj`, `stm`, `tst` and `tse`, but no `stf` (https://datajud-wiki.cnj.jus.br/api-publica/endpoints).
- **Fit:** poor. Skip STF for the PoC, or hand-download a small curated set of acórdão PDFs.

### 2.3 STJ: Portal de Dados Abertos (CKAN)

- **Base:** `https://dadosabertos.web.stj.jus.br`. It runs the standard CKAN Action API (`/api/3/action/package_list`, `/package_show?id=`). **No auth.** Rate limits are not documented (**UNVERIFIED**). License is **CC-BY** (`license_id: cc-by` in package_show). (live)
- **Distribution is bulk files, not a query API:** daily or monthly files, each its own CKAN resource.
- **Datasets** (live `package_list`): `integras-de-decisoes-terminativas-e-acordaos-do-diario-da-justica`, `espelhos-de-acordaos-{corte-especial|primeira-secao|…|sexta-turma}` (10 órgãos), `precedentes-qualificados`, `atas-de-distribuicao`, `pautas-futuras`, `acervo-em-tramitacao`, and `api-publica-datajud`.
- **Íntegras** (**the best fit**):
  - Daily pairs: `metadadosYYYYMMDD.json` plus `textosYYYYMMDD.zip`. There are 2,621 resources, running from 2021-01-04 to 2026-09-23 (`metadata_modified` 2026-09-30).
  - Metadata record (live): `SeqDocumento` (the same id names the `.txt` file in the ZIP), `dataPublicacao`, `tipoDocumento` (ACORDAO/DECISAO), `numeroRegistro`, `processo` (e.g. "HC 637258"), `dataRecebimento`, `dataDistribuicao`, `NM_MINISTRO` (relator), `recurso`, `teor` (outcome), `descricaoMonocratica`, `assuntos` (CNJ SGT subject codes, which can be mapped to `categoria_juridica` via https://dpj.cnj.jus.br/sgt/api/v1.0/assuntos.csv).
  - The `.txt` holds the **full text of the decision or acórdão**.
  - **Gotchas:**
    - The data dictionary calls the relator field `ministro`, but the JSON key is actually `NM_MINISTRO`.
    - The text uses `<br>` line breaks and has words glued together ("SANTANAem", "Opaciente"), so it needs cleaning.
    - **Party names appear in clear text**, which is a PII concern and a good use case for the guardrail layer.
    - `assuntos` is a dotted-code string.
- **Espelhos de acórdãos:**
  - Monthly JSON per órgão julgador. Terceira Turma for 2026-05 has 737 records, 2.4 MB.
  - Fields (live): `id`, `numeroProcesso`, `numeroRegistro`, `siglaClasse`, `descricaoClasse`, `nomeOrgaoJulgador`, `ministroRelator`, `dataPublicacao`, `ementa`, `tipoDeDecisao`, `dataDecisao`, `decisao`, `jurisprudenciaCitada`, `notas`, `informacoesComplementares`, `termosAuxiliares`, `teseJuridica`, `tema`, `referenciasLegislativas[]`, `acordaosSimilares[]`.
  - **It holds the ementa and the dispositivo, not the inteiro teor.** It only covers acórdãos selected by the Secretaria de Jurisprudência (dataset notes).
  - **Gotchas:** `dataPublicacao` is a string like `"DJEN       DATA:27/04/2026"`, and `dataDecisao` is `"20260422"`. Both need parsing.

### 2.4 CNJ DataJud: API Pública

- **Base:** `https://api-publica.datajud.cnj.jus.br/api_publica_<alias>/_search`. There are 91 aliases covering TST, TSE, STJ, STM, the TRFs, TJs, TRTs, TREs and TJMs (https://datajud-wiki.cnj.jus.br/api-publica/endpoints). The service is **Elasticsearch Query DSL over POST** (response header `X-Elastic-Product: Elasticsearch`).
- **Auth:** there is one **public API key published on the wiki**, sent as `Authorization: APIKey <key>`. "A chave poderá ser alterada pelo CNJ a qualquer momento" (https://datajud-wiki.cnj.jus.br/api-publica/acesso). Read the key from config and don't hardcode it. The wiki returns **403 to non-browser user agents**, so send a browser User-Agent.
- **Paging:** `size` from 10 to 10,000. Deep pagination uses `search_after` with `sort` on `@timestamp` (https://datajud-wiki.cnj.jus.br/api-publica/exemplos/exemplo3). Rate limit is not documented (**UNVERIFIED**). The API is labelled "fase Beta" (…/consideracoes_finais).
- **Shape** (live STJ query, `hits.total ≥ 10000`): `id`, `tribunal`, `grau`, `numeroProcesso`, `dataAjuizamento` (`20260318000000`), `nivelSigilo`, `orgaoJulgador{codigo,nome,codigoMunicipioIBGE}`, `classe{codigo,nome}`, `sistema`, `formato`, `assuntos[{codigo,nome}]`, `movimentos[{codigo,nome,dataHora,complementosTabelados[],orgaoJulgador}]`, `dataHoraUltimaAtualizacao`.
- **It returns metadata only.** The wiki itself says "metadados… capas processuais e movimentações". There is no inteiro teor, ementa or relator name; you only get the gabinete in `orgaoJulgador.nome`. Party data is excluded.
- **Gotcha:** some names come back mis-encoded, e.g. `MARCO AURÃLIO BELLIZZE` (live).
- **Terms of use:** use must be "não comerciais" (3.3). Users may not "modificar, distribuir, vender ou explorar comercialmente" the data (3.8), and must notify CNJ of anything they publish (3.9) (https://datajud-wiki.cnj.jus.br/api-publica/termo-uso). **This is more restrictive than CC-BY** and matters if the PoC becomes a product.

### 2.5 Fit against the target schema

| Target field | QD | STF | STJ Íntegras | STJ Espelhos | DataJud |
|---|---|---|---|---|---|
| id | url hash / territory+date | n/a | `SeqDocumento` | `id` | `id` |
| tribunal/municipio | `territory_name`, `state_code` | n/a | STJ (implicit) | STJ + `nomeOrgaoJulgador` | `tribunal` |
| data_publicacao | `date` | n/a | `dataPublicacao` | `dataPublicacao` (needs parsing) | — (only `dataAjuizamento`) |
| relator/orgao | — | n/a | `NM_MINISTRO` | `ministroRelator`, `nomeOrgaoJulgador` | `orgaoJulgador.nome` (gabinete) |
| inteiro_teor | `.txt` (whole gazette) | n/a | **yes (`.txt`)** | no (ementa + decisao) | **no** |
| categoria_juridica | — | n/a | `assuntos` (CNJ codes) | `siglaClasse`, `tema`, `referenciasLegislativas` | `classe`, `assuntos` |
| Access | REST, no auth | WAF-blocked | bulk CKAN, no auth | bulk CKAN, no auth | ES DSL, public key |
| License | CC-BY 4.0 | — | CC-BY | CC-BY | CNJ ToU, non-commercial |

### 2.6 Recommendation

1. **Primary corpus: STJ Íntegras.** Take N days of `metadados*.json` plus `textos*.zip` from CKAN. A few hundred to a few thousand documents is plenty for a PoC, and a single day file is ~60 KB–1.7 MB of metadata. It is the only source here that gives **full text + relator + date + subject codes with no auth and a CC-BY license**. The ingestion "API" is just CKAN `package_show` followed by resource downloads, which fits `legal_api_service.py`.
2. **Enrich** with the STJ Espelhos (join on `numeroRegistro`) for ementa, tese and legislative references. Ementas are ideal hierarchical-chunk parents for Level 2. Map `assuntos` to readable `categoria_juridica` with the CNJ SGT CSV.
3. **Optional live-API flavour: DataJud** for "process status" tool calls in the agentic levels, e.g. "what is the latest movement of process X?". This shows tool routing (a Jev `Choice` between the vector store and the DataJud tool) without needing full text. Respect the non-commercial terms.
4. **Optional second collection: Querido Diário** (municipal acts), to make the router's choice between collections meaningful. Split gazettes into acts before chunking.
5. **Skip STF** for now: no API, and the portal is behind a WAF challenge.

Before committing to Jev for Level 3, run a small PT-BR eval of `Noul` relevance on STJ passages against an LLM `with_structured_output` baseline, given TypeSafe's English-first caveat.
