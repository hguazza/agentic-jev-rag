# Agentic Recruiter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task by task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A FastAPI RAG service over a recruiting corpus, a LangGraph five-agent Recruiter that calls it as a tool, and a System 1 comparison of an LLM decider against Jev.

**Architecture:** One package, `agentic_recruiter`, with these layers:

- Model gateway (OpenRouter, faked in tests);
- corpus;
- RAG (hybrid Qdrant index, Search, Match, Ask, FastAPI);
- System 1 deciders (LLM and Jev);
- Recruiter (tool-calling agent loop, five agents, LangGraph graph, FastAPI);
- evaluation.

Two FastAPI apps are served from one Docker image.

**Tech stack:** Python 3.12 via `uv`, FastAPI, Pydantic v2, LangGraph 1.2, httpx, qdrant-client 1.19 against a Qdrant 1.19.1 server, and pytest.

**Spec:** `docs/superpowers/specs/2026-10-08-recruiter-design.md`. Vocabulary: `CONTEXT.md`.

**Execution:** the owner asked for autonomous execution in this session (Native). Do one task at a time with TDD at the seams the spec lists. Commit after each task on branch `recruiter-v1`. Run one whole-branch review at the end.

## Global Constraints

- **OpenRouter only.** Every model call goes through `ModelGateway`. No LangChain chat models. The OpenRouter key variable is `OPEN_ROUTER_API_TOKEN`.
- **Fakes.** `FakeModelGateway` is the only fake. Qdrant is real (Docker) in tests.
- **Model IDs come from `configs/recruiter.toml`.** Model roles: `system_2_model`, `system_1_baseline`, `reference_judge`, `embedding_model`, `reranker`, `jev_model`.
- **Collection names are `candidates`, `jobs`, `interviews` and `policies`.** The Qdrant collection is `{collection_prefix}{collection}`.
- **System 1 implementations are named `"llm"` and `"jev"`.**
- **Wording:** use `CONTEXT.md` terms in code and tests (Candidate, Job, Evidence, Screening verdict, Recruitment report).
- **Default test run:** `uv run pytest` must pass with Qdrant up and no network. Live tests are marked `live`.

## Review Focus

1. **A Candidate with expired Consent must never reach a shortlist,** even if they are the best match. Tested in Tasks 7 and 10.
2. **Requests whose discrimination is phrased indirectly** ("digital natives only", "no career gaps for mothers") must be refused, not just obvious ones. Tested in Task 6 through the decider mapping, and measured in the Labelled set (Task 12).
3. **An agent that never calls its submit tool** (looping or chatting) must stop at the step limit, and the report must say so. Tested in Task 8.
4. **A citation number outside the sources, or an answer with no citations,** must never crash Ask. The groundedness result must be false when there are no citations. Tested in Task 7.
5. **A Match or Ask for an unknown `job_id` or `candidate_id`** returns 404, not 500. Tested in Task 7.

---

### Task 1: Clean slate and dependencies

**Files:**
- Delete: `src/agentic_recruiter/pipeline.py`, `chunking.py`, `indexing.py`, `corpus.py`; `tests/test_level_pipeline.py`; `tests/fixtures/corpus/`. The parked doc points at commit `15e4af3` for recovery.
- Rename: `configs/en.toml` → `configs/recruiter.toml`. Collections become `["candidates", "jobs", "interviews", "policies"]`. Add `system_1 = "llm"`, `rerank_candidates = 30` and `rag_url = "http://localhost:8000"`.
- Modify: `src/agentic_recruiter/config.py` and `tests/test_config.py`.
- Modify: `tests/conftest.py`, which keeps only the Qdrant and config fixtures.
- Modify: `pyproject.toml`. Add `fastapi`, `uvicorn[standard]`, `langgraph>=1.2,<1.3` and `pydantic>=2`.

**Interfaces produced:**

```python
@dataclass(frozen=True)
class Config:
    models: ModelConfig
    collections: tuple[str, ...] = ("candidates", "jobs", "interviews", "policies")
    qdrant_url: str = "http://localhost:6333"
    collection_prefix: str = ""
    corpus_dir: str = "corpus"
    chunk_size: int = 1200
    chunk_overlap: int = 150
    top_k: int = 8
    rerank_candidates: int = 30
    system_1: str = "llm"
    rag_url: str = "http://localhost:8000"

def load_config(path) -> Config  # env overrides: QDRANT_URL, RAG_URL, SYSTEM_1
```

**Steps:**
- [ ] Update `test_config.py` to load `configs/recruiter.toml` and assert each Model role, `system_1 == "llm"`, and the environment overrides for `QDRANT_URL` and `RAG_URL`. Run it and see it fail.
- [ ] Implement, delete the legal modules, run `uv sync`, then `uv run pytest`. Gateway and config tests should pass.
- [ ] Commit: "Clear the legal pipeline; recruiter config".

### Task 2: Model gateway: chat with tools, rerank, Jev decide

**Files:**
- Modify: `src/agentic_recruiter/gateway.py` and `openrouter.py`.
- Test: `tests/test_openrouter_gateway.py`.
- Recorded responses: `tests/fixtures/openrouter/chat_tool_call.json`, `rerank.json` and `jev_systemone.json`. These are already captured live.

**Interfaces produced (`gateway.py`):**

```python
@dataclass(frozen=True)
class ToolCall: id: str; name: str; arguments: dict

@dataclass(frozen=True)
class ChatResult:
    content: str | None; tool_calls: tuple[ToolCall, ...]
    latency_s: float; cost_usd: float; retries: tuple[str, ...] = ()
    def as_message(self) -> dict   # OpenAI assistant message incl. tool_calls

@dataclass(frozen=True)
class RerankResult: scores: list[float]; latency_s: float; cost_usd: float; retries: tuple[str, ...] = ()  # scores in input order

@dataclass(frozen=True)
class Decisions: answers: dict[str, dict]; latency_s: float; cost_usd: float; retries: tuple[str, ...] = ()

class ModelGateway(Protocol):
    def complete(self, model, prompt, *, system=None, response_schema=None) -> Completion
    def chat(self, model: str, messages: list[dict], *, tools: list[dict] | None = None) -> ChatResult
    def embed(self, model, texts) -> Embeddings
    def rerank(self, model: str, query: str, documents: list[str]) -> RerankResult
    def decide(self, model: str, state: str | dict, questions: dict[str, dict]) -> Decisions

@dataclass
class FakeModelGateway:
    chat_responder: Callable[[str, list[dict], list[dict] | None], str | list[ToolCall]] | None = None
    complete_responder: Callable[[str, str, str | None, tuple | None], str] | None = None
    decide_responder: Callable[[str | dict, dict[str, dict]], dict[str, dict]] | None = None
    dimensions: int = 512
    calls: list[tuple[str, str, Any]]   # (kind, model, payload)
```

**Fake defaults:**
- **chat:** returns `"OK"`.
- **complete:** returns `"The documents answer this [1]."`
- **decide:**
  - `noul` → `{"type": "noul", "noul": 0.0}`;
  - `choice` → the first criteria key with probability 1;
  - `score` → 0.
- **rerank:** word-overlap Jaccard.
- **embed:** hashed words (as before).

**Tests:**
- `test_chat_returns_tool_calls_with_parsed_arguments` (recorded `chat_tool_call.json`): one ToolCall `match_candidates` with `{"job_id": "J-001"}`, `content` is None, and cost 0.000882.
- `test_chat_sends_tools_and_messages` checks the request body has `tools` and `messages` exactly as passed.
- `test_rerank_returns_scores_in_input_order_with_cost` (recorded): scores[1] ≈ 0.8399 is the highest and cost is 0.002. The URL is `/api/v1/rerank`, and the body has `model`, `query` and `documents`.
- `test_decide_posts_to_systemone_and_returns_answers` (recorded): `answers["discriminatory"]["noul"] == 0.99`, `answers["route"]["choice"] == "candidates"`, cost 0.000018144. The URL is `/api/v1/systemone`, and the body has `model`, `state` and `questions`.
- `test_fake_gateway_scripted_chat_returns_tool_calls` checks the responder's ToolCalls come back in `ChatResult.tool_calls`.

**Steps:** write the tests, run them red, implement, run them green, commit with "Gateway: tool calls, rerank, Jev decide".

### Task 3: Corpus records, Policies and test fixtures

**Files:**
- Create: `src/agentic_recruiter/corpus/__init__.py` and `documents.py`.
- Create: `corpus/policies/*.md`, with six hand-written Policies:
  - `fair-hiring.md` (protected characteristics, indirect discrimination examples);
  - `data-protection-and-consent.md` (GDPR, consent window, retention of 24 months, right to erasure);
  - `interview-guide.md`;
  - `scorecards.md` (competencies per occupation, 1–5 scale);
  - `fast-track.md`;
  - `ai-act-human-oversight.md` (high-risk, human review of rejections).
- Create: `tests/fixtures/corpus/candidates.jsonl` (8 Candidates across backend and data science, one with expired Consent), `jobs.jsonl` (3 Jobs), `interviews.jsonl` (4 notes), plus a symlink or copy of the policies.
- Test: `tests/test_corpus.py`.

**Interfaces produced:**

```python
@dataclass(frozen=True)
class Document:
    id: str; collection: str; title: str; text: str; metadata: dict
    def consent_valid_on(self, day: date) -> bool   # candidates only; True for other collections

def load_corpus(directory: Path) -> list[Document]  # *.jsonl per collection + policies/*.md (id = "policy:<stem>", title = first heading)
def write_jsonl(documents: list[Document], path: Path) -> None
```

**Tests:** `load_corpus` returns every Collection; Policy ids and titles come from the Markdown; `consent_valid_on` is false after `consent_until`; a document with an unknown collection raises `ValueError`.

**Commit:** "Corpus records, Policies, fixtures".

### Task 4: Structure-aware chunking

**Files:** create `src/agentic_recruiter/rag/__init__.py` and `rag/chunking.py`. Test: `tests/test_chunking.py`.

**Interfaces produced:**

```python
@dataclass(frozen=True)
class Chunk:
    id: str; document_id: str; collection: str; section: str; text: str; metadata: dict

def chunk_document(document: Document, size: int, overlap: int) -> list[Chunk]
```

**Rules:**
- A section is a line that is a Markdown heading, or a line matching one of the known CV and Job headings, case-insensitive with an optional trailing colon:
  - Summary, Profile, Experience, Work History, Employment, Education, Skills, Certifications, Projects;
  - Responsibilities, Requirements, Qualifications, Benefits, About.
- Sections longer than `size` are split into windows with `overlap`.
- Interview notes are one chunk unless longer than `size`.
- The chunk id is `{document_id}#{n}`.
- Each chunk's text is prefixed with `"{title} — {section}\n"`, so the embedding carries context.

**Tests:**
- A CV with Experience, Education and Skills gives three sections named that way.
- A long section is windowed with overlap.
- Text with no headings gives one "body" section.
- Metadata is copied to every chunk.

**Commit:** "Structure-aware chunking".

### Task 5: Hybrid index and Search

**Files:** create `src/agentic_recruiter/rag/sparse.py` and `rag/index.py`. Test: `tests/test_index.py` (real Qdrant).

**Interfaces produced:**

```python
def sparse_vector(text: str) -> models.SparseVector   # hashed token TF, indices sorted, 2**20 buckets, light English stopword list
def qdrant_name(config: Config, collection: str) -> str
def index_corpus(documents: list[Document], config: Config, gateway: ModelGateway, qdrant: QdrantClient) -> dict[str, int]  # chunks per collection; recreates collections with vectors {"dense": cosine}, sparse {"sparse": modifier=IDF}

@dataclass(frozen=True)
class SearchFilters:
    occupation: str | None = None; location: str | None = None; min_years_experience: float | None = None
    candidate_id: str | None = None; job_id: str | None = None; consent_valid_on: date | None = None

@dataclass(frozen=True)
class ScoredChunk: chunk: Chunk; score: float

def hybrid_search(query: str, collections: Sequence[str], filters: SearchFilters, limit: int, *, config, gateway, qdrant) -> list[ScoredChunk]
```

**Implementation notes:**
- Use the `query_points` prefetch on `dense` and `sparse` with `FusionQuery(fusion=Fusion.RRF)`.
- `consent_valid_on` filters `metadata.consent_until >= day`. It is stored as an ISO string plus `consent_until_ord` (an int date ordinal) so a Range filter works.
- Filters on a field that a Collection lacks don't apply to that Collection: build the filter per Collection.

**Tests** (fixture corpus, fake gateway):
- A query naming a backend skill returns that Candidate first.
- An exact rare term (a specific tool name present in one CV) is found through the sparse side even when the dense side misses it. Check by giving the fake a different embedding hash seed so dense similarity is weak.
- The occupation filter excludes other occupations.
- `consent_valid_on` excludes the expired Candidate.
- Re-indexing is idempotent (counts are equal).

**Commit:** "Hybrid index and Search".

### Task 6: System 1 deciders (LLM and Jev)

**Files:** create `src/agentic_recruiter/system1/__init__.py`, `types.py`, `llm.py`, `jev.py` and `prompts.py` (shared instructions). Test: `tests/test_system1.py`.

**Interfaces produced:**

```python
@dataclass(frozen=True)
class System1TraceEntry: decision: str; outcome: str; latency_s: float; cost_usd: float; implementation: str

GuardrailVerdict = Literal["allowed", "discriminatory", "out_of_scope"]

@dataclass(frozen=True) class RouteDecision: collection: str; trace: System1TraceEntry
@dataclass(frozen=True) class RelevanceDecision: keep: list[bool]; scores: list[float]; trace: System1TraceEntry
@dataclass(frozen=True) class GuardrailDecision:
    verdict: GuardrailVerdict; trace: System1TraceEntry
    @property
    def allowed(self) -> bool
@dataclass(frozen=True) class GroundednessDecision: grounded: bool; score: float; trace: System1TraceEntry

class System1Decider(Protocol):
    name: str
    def route(self, question: str) -> RouteDecision
    def chunk_relevance(self, question: str, chunks: list[str]) -> RelevanceDecision
    def input_guardrail(self, text: str) -> GuardrailDecision
    def groundedness(self, claim: str, evidence: list[str]) -> GroundednessDecision

def make_deciders(config: Config, gateway: ModelGateway) -> dict[str, System1Decider]  # {"llm": LlmDecider, "jev": JevDecider}
```

**Shared instructions** in `prompts.py`:
- `ROUTE_CRITERIA` maps each Collection to its description;
- `GUARDRAIL_CRITERIA` maps each verdict to its description, with the protected characteristics and indirect-discrimination examples listed;
- `RELEVANCE_INSTRUCTIONS`;
- `GROUNDEDNESS_INSTRUCTIONS`.

**`LlmDecider`** calls `gateway.complete(system_1_baseline, prompt, system=..., response_schema=(name, schema))`:
- **Route:** enum over Collections.
- **Relevance:** `{"relevant": [bool...]}`, with the chunks numbered.
- **Guardrail:** enum over verdicts.
- **Groundedness:** `{"grounded": bool}`.

**`JevDecider`** calls `gateway.decide(jev_model, state, questions)`:
- **Route:** a `choice` with the `ROUTE_CRITERIA`.
- **Relevance:** one `noul` per chunk, keyed `chunk_{i}`. The state carries the question and the numbered chunks. Keep the chunk when the noul is ≥ 0.5.
- **Guardrail:** a `choice` with the `GUARDRAIL_CRITERIA`.
- **Groundedness:** a `noul`. The state is the claim plus the evidence. Grounded when the noul is ≥ 0.5.

Trace outcomes are the collection name, `"kept 3/5"`, the verdict, and `"grounded"` or `"ungrounded"`.

**Tests** (fake gateway):
- For both deciders: a guardrail answer of `discriminatory` from the scripted model makes `allowed` false, and the trace implementation is `"llm"` or `"jev"`.
- The Jev relevance request has one noul per chunk, and a scripted 0.7/0.2 gives keep `[True, False]`.
- The LLM route request uses `response_schema` with an enum equal to the Collections.
- The trace carries the gateway's latency and cost.

**Commit:** "System 1 deciders: LLM and Jev".

### Task 7: RAG service and API (Search, Match, Ask)

**Files:**
- Create: `src/agentic_recruiter/rag/models.py` (Pydantic request and response models), `rag/service.py` and `rag/api.py`.
- Test: `tests/test_rag_api.py` (TestClient, real Qdrant, fake gateway, fixture corpus indexed once per session).

**Interfaces produced (Pydantic):**

```python
class Filters(BaseModel): occupation: str | None = None; location: str | None = None; min_years_experience: float | None = None; candidate_id: str | None = None; job_id: str | None = None
class SearchRequest(BaseModel): query: str; collections: list[str] | None = None; filters: Filters = Filters(); top_k: int = 8; rerank: bool = True
class ChunkOut(BaseModel): chunk_id: str; document_id: str; collection: str; section: str; text: str; score: float; metadata: dict
class SearchResponse(BaseModel): chunks: list[ChunkOut]
class MatchRequest(BaseModel): job_id: str | None = None; job_text: str | None = None; top_k: int = 5; filters: Filters = Filters()   # exactly one of job_id/job_text
class CandidateMatch(BaseModel): candidate_id: str; name: str; score: float; evidence: list[ChunkOut]
class MatchResponse(BaseModel): job_id: str | None; candidates: list[CandidateMatch]
class AskRequest(BaseModel): question: str; collections: list[str] | None = None; filters: Filters = Filters(); system_1: Literal["llm", "jev"] | None = None
class TraceOut(BaseModel): decision: str; outcome: str; latency_s: float; cost_usd: float; implementation: str
class Citation(BaseModel): n: int; chunk_id: str; document_id: str
class AskResponse(BaseModel): answer: str | None; refused: bool; refusal_reason: str | None; citations: list[Citation]; grounded: bool | None; evidence: list[ChunkOut]; system_1_trace: list[TraceOut]; cost_usd: float; latency_s: float

class RagService:
    def __init__(self, config: Config, gateway: ModelGateway, qdrant: QdrantClient, documents: list[Document], *, today: Callable[[], date] = date.today)
    def search(self, request: SearchRequest) -> SearchResponse
    def match(self, request: MatchRequest) -> MatchResponse     # raises UnknownDocument
    def ask(self, request: AskRequest) -> AskResponse
    def document(self, document_id: str) -> Document            # raises UnknownDocument

def create_rag_app(service: RagService) -> FastAPI    # UnknownDocument -> 404; /health
def create_default_rag_app() -> FastAPI               # load_config(CONFIG env or configs/recruiter.toml), OpenRouterGateway.from_env(), load_corpus
```

**Match algorithm:**
1. Query = the Job's title, requirements and skills.
2. Run `hybrid_search` on `candidates`, limited to `rerank_candidates` chunks, filtered by Consent valid on `today()`.
3. Group by Candidate, keeping the top three chunks each.
4. Rerank one string per Candidate (their concatenated Evidence) against the query.
5. Sort by rerank score and take `top_k`.

**Ask flow:** described in the spec. When the guardrail refuses, there are no retrieval or model calls. Citations are parsed from `[n]` and out-of-range numbers are ignored. No citations means `grounded = False` without calling the decider.

**Tests:**
- `/search` returns chunks with metadata; the filter works.
- `/match` for the backend Job ranks the backend Candidates first, never returns the expired-Consent Candidate (even though they're the best match), and each Candidate has Evidence.
- `/match` with an unknown `job_id` → 404.
- `/ask` with a discriminatory question (scripted decider) → `refused` with no evidence, and the trace has only the guardrail entry.
- `/ask` routes to the Collection the scripted router picks; the answer's citations map to evidence chunks; `grounded` comes from the Groundedness check.
- `/ask` where the model cites `[9]` with three sources → no crash, the citation is ignored, `grounded` is False.
- `system_1: "jev"` switches the trace implementation to `jev`.
- `/documents/{id}` returns 200, or 404 when unknown.

**Commit:** "RAG service and API".

### Task 8: Recruiter tools and the agent loop

**Files:** create `src/agentic_recruiter/recruiter/__init__.py`, `rag_client.py`, `tools.py` and `agent_loop.py`. Test: `tests/test_agent_loop.py`.

**Interfaces produced:**

```python
class RagClient:
    def __init__(self, base_url: str, *, transport: httpx.BaseTransport | None = None)
    def search(self, query: str, collections: list[str] | None = None, filters: dict | None = None, top_k: int = 8) -> dict
    def match(self, job_id: str, top_k: int = 5) -> dict
    def ask(self, question: str, collections: list[str] | None = None, filters: dict | None = None, system_1: str | None = None) -> dict
    def document(self, document_id: str) -> dict

@dataclass(frozen=True)
class Tool:
    name: str; description: str; parameters: dict; fn: Callable[..., Any]
    def schema(self) -> dict   # OpenAI tools format

@dataclass
class AgentResult:
    submitted: dict | None; steps: int; cost_usd: float; latency_s: float
    stopped_reason: Literal["submitted", "step_limit", "no_tool_call"]; transcript: list[dict]

def run_agent(gateway: ModelGateway, model: str, system: str, task: str, tools: list[Tool], submit: Tool, *, max_steps: int = 8) -> AgentResult
def rag_tools(rag: RagClient) -> dict[str, Tool]   # search, match_candidates, ask, get_document
```

**Loop behaviour:**
- Send the messages, then execute each tool call and append its result (JSON, truncated to 6,000 characters).
- Calling `submit` validates the arguments against its schema's required keys and stops.
- A tool error is returned to the model as `{"error": ...}`.
- A reply with no tool call gets one nudge ("Call a tool or submit."). A second reply with no tool call stops with `"no_tool_call"`.

**Tests** (fake gateway with a scripted chat):
- Tool call → result → submit returns `submitted`, and costs are summed.
- An agent that never submits stops at `max_steps` with `"step_limit"`.
- An unknown tool name gets an error result and the loop continues.
- `RagClient` against the in-process RAG app (ASGI transport) returns a match.

**Commit:** "Agent loop and RAG tools".

### Task 9: The five agents

**Files:** create `src/agentic_recruiter/recruiter/context.py`, `schemas.py`, `prompts.py`, `sourcer.py`, `screener.py`, `interviewer.py`, `compliance.py` and `supervisor.py`. Test: `tests/test_agents.py`.

**Interfaces produced:**

```python
@dataclass
class RecruiterContext: config: Config; gateway: ModelGateway; rag: RagClient; deciders: dict[str, System1Decider]; system_1: str
    @property
    def decider(self) -> System1Decider

# schemas.py (Pydantic)
class ShortlistEntry(BaseModel): candidate_id: str; name: str; reason: str
class Shortlist(BaseModel): job_id: str; candidates: list[ShortlistEntry]
class CompetencyScore(BaseModel): competency: str; score: int (1..5); evidence_chunk_ids: list[str]; rationale: str
class ScreeningVerdict(BaseModel): candidate_id: str; job_id: str; decision: Literal["advance", "reject"]; scores: list[CompetencyScore]; summary: str
class InterviewTurn(BaseModel): question: str; answer: str
class InterviewAssessment(BaseModel): candidate_id: str; job_id: str; turns: list[InterviewTurn]; scores: list[CompetencyScore]; recommendation: Literal["hire", "no_hire", "next_round"]; summary: str
class ComplianceFinding(BaseModel): check: Literal["input_guardrail", "consent", "groundedness", "policy"]; subject: str; passed: bool; detail: str
class AgentRunNote(BaseModel): agent: str; stopped_reason: str; steps: int; cost_usd: float

def source(ctx, job_id: str, goal: str) -> tuple[Shortlist, AgentResult]
def screen(ctx, candidate_id: str, job_id: str) -> tuple[ScreeningVerdict, AgentResult]
def next_question(ctx, candidate: dict, job: dict, turns: list[InterviewTurn], max_questions: int = 4) -> str | None   # None when done
def assess(ctx, candidate: dict, job: dict, turns: list[InterviewTurn]) -> InterviewAssessment
def simulated_answer(ctx, candidate: dict, question: str) -> str
def intake(ctx, goal: str) -> tuple[ComplianceFinding, list[System1TraceEntry]]
def review(ctx, shortlist: Shortlist, verdicts: list[ScreeningVerdict]) -> tuple[list[ComplianceFinding], list[System1TraceEntry]]
def supervise(ctx, progress: dict) -> str   # one of "sourcer", "screener", "interviewer", "compliance", "finish"
```

**Agent details:**
- **Sourcer:** the tools are `match_candidates`, `search` and `get_document`, and it submits through `submit_shortlist`.
- **Screener:** the tools are `get_document`, `ask` and `search`, and it submits through `submit_verdict`. The Scorecard comes from the Policies via `ask` or `search` on `policies`.
- **Interviewer:** `complete` with structured output for the next question (or `{"done": true}`) and for the assessment.
- **Compliance review:**
  - Consent: `get_document` for each Candidate, checking `consent_until`.
  - Groundedness: the decider's `groundedness(verdict.summary + rationales, evidence texts)`. The evidence texts are fetched by chunk id via `search` with a `candidate_id` filter.
- **Supervisor:** `chat` with handoff tools. An invalid choice (for example screener before a shortlist exists) is corrected to the next valid step, and the correction is recorded.

**Tests** (fake gateway scripted per agent, RAG app in-process over real Qdrant):
- The Sourcer submits a shortlist of backend Candidates.
- The Screener returns a verdict whose scores cite chunk ids.
- The Interviewer stops after `max_questions` and assesses.
- `intake` refuses "only men under 35".
- `review` flags expired Consent and an ungrounded verdict.
- `supervise` corrects an invalid choice.

**Commit:** "Five agents".

### Task 10: LangGraph Recruiter graph and Recruiter API

**Files:** create `src/agentic_recruiter/recruiter/graph.py`, `interview_graph.py`, `report.py` and `api.py`. Test: `tests/test_recruiter_api.py`.

**Interfaces produced:**

```python
class RecruiterState(TypedDict, total=False):
    goal: str; job_id: str; refused: bool; shortlist: Shortlist | None; verdicts: list[ScreeningVerdict]
    assessments: list[InterviewAssessment]; findings: list[ComplianceFinding]; trace: list[System1TraceEntry]
    notes: list[AgentRunNote]; supervisor_steps: int; next: str; compliance_done: bool

def build_recruiter_graph(ctx: RecruiterContext, *, simulate_interviews: bool = True, max_interviews: int = 2) -> CompiledStateGraph

class RecruitmentReport(BaseModel):
    goal: str; job_id: str; refused: bool; shortlist: Shortlist | None; verdicts: list[ScreeningVerdict]
    assessments: list[InterviewAssessment]; findings: list[ComplianceFinding]; system_1_trace: list[TraceOut]
    notes: list[AgentRunNote]; cost_usd: float

def build_interview_graph(ctx) -> CompiledStateGraph   # question -> interrupt(question) -> record answer -> ... -> assess; MemorySaver checkpointer

class RunRequest(BaseModel): goal: str; job_id: str; system_1: Literal["llm", "jev"] | None = None; simulate_interviews: bool = True
class InterviewStart(BaseModel): candidate_id: str; job_id: str
class InterviewReply(BaseModel): message: str
class InterviewState(BaseModel): interview_id: str; question: str | None; assessment: InterviewAssessment | None

def create_recruiter_app(ctx: RecruiterContext) -> FastAPI   # POST /runs, POST /interviews, POST /interviews/{id}/reply, GET /health
def create_default_recruiter_app() -> FastAPI
```

**Graph:**

```
START → compliance_intake →(refused)→ report → END
                          →(ok)→ supervisor
supervisor → sourcer | screener | interviewer | compliance_review | report
sourcer → supervisor ; screener (all shortlisted) → supervisor ; interviewer (advanced, max N) → supervisor ; compliance_review → supervisor
```

The Supervisor is limited to 10 steps. When it's exhausted the run goes to the report, with a note.

**Tests** (fake gateway with a scripted script per agent, in-process RAG):
- `POST /runs` with "Find backend engineers, men under 35 only" → `refused: true`, no shortlist, and a guardrail finding.
- `POST /runs` for the backend Job → the report has a shortlist (no expired-Consent Candidate), verdicts for each shortlisted Candidate, assessments for advanced Candidates, a Consent and groundedness finding, and a System 1 trace containing `input_guardrail` and `groundedness`.
- `POST /interviews` returns the first question. Replying N times returns an assessment. An unknown `interview_id` → 404.
- `system_1: "jev"` → every trace entry's implementation is `jev`.

**Commit:** "Recruiter graph and API".

### Task 11: Build the real corpus

**Files:** create `src/agentic_recruiter/corpus/public.py` (download and sample the datasets chosen in `docs/research/2026-10-08-recruiting-datasets.md`), `corpus/synthetic.py` (Clients, Consent dates, names, interview notes, Outcomes via the gateway, seeded) and `corpus/build.py` (CLI: `uv run python -m agentic_recruiter.corpus.build`). Output: `corpus/*.jsonl`, committed. Also create `src/agentic_recruiter/rag/ingest.py` (CLI: index the corpus into Qdrant). Test: `tests/test_corpus_build.py` (sampling and normalisation from small fixture extracts, with a fake gateway for metadata extraction and generation).

**Notes:**
- Sample about 200 résumés and 60 Jobs over 6 occupations.
- Metadata extraction (skills, years, occupation) uses `complete` with a schema on the System 1 baseline.
- Interview notes: about 120, generated for (Candidate, Job) pairs with planted Outcomes. Pairs with good skill overlap lean towards advanced or hired, so the Match evaluation has ground truth that is not trivially circular. The Outcome is drawn from a seeded distribution conditioned on the overlap.
- Downloads are cached in `data/` (gitignored).

**Tests:**
- Normalising a public résumé row gives a candidate Document with an id and metadata.
- Sampling is deterministic for a seed.
- Generated interview notes reference existing ids.

Run the build live once and commit the corpus.

**Commit:** "Corpus build and committed corpus".

### Task 12: Evaluation: Labelled sets, System 1 comparison, Match evaluation

**Files:**
- Create: `evals/data/input_guardrail.jsonl` (60 rows), `route.jsonl` (40), `chunk_relevance.jsonl` (40, built from the corpus) and `groundedness.jsonl` (30).
- Create: `src/agentic_recruiter/evals/system1.py`, `evals/match.py` and `evals/metrics.py`.
- CLIs: `uv run python -m agentic_recruiter.evals.system1 [--implementations llm jev]` and `uv run python -m agentic_recruiter.evals.match`. Output goes to `results/<timestamp>/` (records.jsonl and summary.md).
- Test: `tests/test_evals.py`.

**Interfaces:**

```python
def accuracy(pairs: list[tuple[str, str]]) -> float
def percentile(values: list[float], p: float) -> float
def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float
def mrr(ranked: list[str], relevant: set[str]) -> float
def run_system1_comparison(deciders: dict[str, System1Decider], data_dir: Path, out_dir: Path) -> dict   # summary per implementation per decision: accuracy, p50, p95, cost
```

**Tests:**
- Metrics on known values: recall@2 of [a, b, c] with {b, c} is 0.5; MRR is 0.5; p50 of [1, 2, 3, 4] is 2.5.
- Comparison over a tiny data dir with scripted deciders writes records and a summary with the expected accuracies.

Run the comparison live once and commit the summary under `results/`.

**Commit:** "Evaluation harness, Labelled sets, first System 1 comparison".

### Task 13: Docker, Compose, README, live suite

**Files:**
- Create: `Dockerfile` (uv-based, one image).
- Modify: `docker-compose.yml`. Services:
  - `qdrant`;
  - `ingest` (one-shot, profile `setup`);
  - `rag-api` on port 8000;
  - `recruiter-api` on port 8001, with `RAG_URL=http://rag-api:8000`.
- Modify: `.env.example`, `README.md` and `tests/test_live.py`. The live suite covers each Model role (chat, embed, rerank, Jev decide), a real Ask, and one real Recruiter run on a seeded Job with `system_1=jev`.
- Modify: `AGENTS.md`, adding a pointer to the spec and plan.

**Verify:**
- `docker compose build`, then `docker compose --profile setup run --rm ingest`, then `docker compose up -d`.
- `curl` against `/health` on both services.
- `curl POST /runs` with a real goal.
- `uv run pytest -m live`.

**Commit:** "Docker services, README, live suite".

### Task 14: Whole-branch review and PR

- [ ] Dispatch one fresh reviewer subagent over `git diff main...recruiter-v1` against the spec and plan. Fix the confirmed findings.
- [ ] Run `uv run pytest` and `uv run pytest -m live`, and confirm the results.
- [ ] Push `recruiter-v1` and open a PR (not merged) whose body sums up what was built, the System 1 comparison table, and the decisions the owner should check.
