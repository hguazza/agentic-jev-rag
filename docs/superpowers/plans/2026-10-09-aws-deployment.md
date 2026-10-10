# AWS Deployment (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deploy the RAG API, Recruiter API and Voice worker to ECS Fargate behind CloudFront, with durable DynamoDB state, API-key auth, Terraform and GitHub Actions — phase 1 of the AWS deployment spec.

**Architecture:** App changes first (Qdrant Cloud key, a `persistence` module for LangGraph checkpoints and run records, async `/runs`, `X-API-Key` auth, two Docker build targets), each test-driven against real Qdrant and DynamoDB Local. Then Terraform in `infra/bootstrap` (state bucket, GitHub OIDC role) and `infra/main` (VPC, ECR, DynamoDB, Secrets Manager, ECS + Cloud Map, internal ALB, CloudFront VPC origin, WAF, alarms, budget), a first manual deployment, and CI/CD.

**Tech Stack:** Python 3.12, FastAPI, LangGraph, `langgraph-checkpoint-aws` (`DynamoDBSaver`), boto3, Terraform ≥ 1.10 with AWS provider ~> 6.68, ECS Fargate (arm64), CloudFront, WAFv2, DynamoDB, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-10-09-aws-deployment-design.md` (decision: `docs/adr/0014-ecs-services-and-agentcore-agents.md`). Phase 2 (AgentCore for `/runs`) is out of this plan.

## Global Constraints

- Region `us-east-1`, one environment, Graviton: every image `linux/arm64`, every task `cpu_architecture = "ARM64"`.
- The Model gateway is the only fake in tests (AGENTS.md). DynamoDB Local and Qdrant run for real (`docker compose up -d qdrant dynamodb`).
- No secret in git or Terraform state: Terraform creates Secrets Manager secrets, `infra/scripts/put-secrets.sh` sets their values from git-ignored `.env.aws`.
- Interviews stay in-process on the Recruiter API; nothing in this plan touches AgentCore.
- The Recruiter API contract is unchanged except `/runs` (202 + `GET /runs/{run_id}`) and `X-API-Key` on every route but `/health`. With no key configured (local default), auth is off.
- Terraform: `required_version = ">= 1.10"`, `hashicorp/aws ~> 6.68`, S3 backend with `use_lockfile = true`. Resource names prefixed `swiftai`.
- Commit messages in this repo are one plain sentence describing the change (see `git log`), ending with the attribution line `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.

## Review Focus

- **Checkpointed interview state must come back as the same types** (pydantic `InterviewTurn`, `InterviewAssessment`) from DynamoDB, or replies after a restart crash. Pinned by Task 3's two-app test, which reads the assessment's turns after resuming on a second app.
- **A run whose model calls fail must end `failed`, not stay `running` forever.** Pinned in Task 4 (`test_a_run_whose_models_fail_is_reported_as_failed`).
- **A run whose task died must stop reporting `running`.** Pinned in Task 4 (`test_a_run_left_running_too_long_is_reported_as_interrupted`).
- **A full Recruitment report must fit a DynamoDB item (400 KB).** Pinned in Task 2 (`test_a_large_report_round_trips`, ~300 KB).
- **A wrong or missing API key must be refused, while `/health` and `/docs` stay open** (CloudFront and the ALB health check call `/health`; reviewers open `/docs`). Pinned in Task 5.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/agentic_jev_rag/config.py` (modify) | New settings: `qdrant_api_key`, `persistence`, `checkpoint_table`, `runs_table`, `dynamodb_endpoint`, `api_key` |
| `src/agentic_jev_rag/recruiter/persistence.py` (create) | Durable state: `make_checkpointer`, `RunRecord`, `MemoryRuns`, `DynamoRuns`, `make_run_store` |
| `src/agentic_jev_rag/recruiter/interview_graph.py` (modify) | `build_interview_graph(ctx, checkpointer=None)` |
| `src/agentic_jev_rag/recruiter/api.py` (modify) | Async `/runs`, `GET /runs/{run_id}`, `X-API-Key` dependency |
| `src/agentic_jev_rag/recruiter/api_models.py` (modify) | `RunStatus` |
| `src/agentic_jev_rag/voice/recruiter_client.py`, `voice/worker.py` (modify) | Send `X-API-Key` |
| `src/agentic_jev_rag/rag/api.py`, `rag/ingest.py`, `evals/match.py` (modify) | Qdrant API key |
| `Dockerfile`, `docker-compose.yml` (modify) | `base`/`voice`/`api` targets; DynamoDB Local service |
| `tests/conftest.py` (modify) | `dynamodb_tables` fixture |
| `tests/test_persistence.py` (create) | Run stores and checkpointer against DynamoDB Local |
| `tests/test_recruiter_api.py`, `tests/test_recruiter_api_docs.py`, `tests/test_voice_interview.py`, `tests/test_config.py` (modify) | New behaviour |
| `tests/test_aws_live.py` (create) | `-m live` checks against the deployed URL |
| `infra/bootstrap/main.tf` | State bucket, GitHub OIDC provider and deploy role |
| `infra/main/*.tf` | The stack, split by concern (versions, variables, network, ecr, data, secrets, iam, ecs, edge, observability, outputs) |
| `infra/scripts/{push-images,put-secrets,ingest}.sh` | Operations |
| `.github/workflows/{ci,deploy}.yml`, `.trivyignore` | CI/CD |
| `README.md`, `AGENTS.md`, `docs/adr/0008-docker-compose-runtime.md`, `.gitignore`, `.env.aws.example` | Docs and hygiene |

---

### Task 1: Config for Qdrant Cloud, persistence and the API key

**Files:**
- Modify: `src/agentic_jev_rag/config.py`
- Modify: `src/agentic_jev_rag/rag/api.py:226-231`, `src/agentic_jev_rag/rag/ingest.py:17`, `src/agentic_jev_rag/evals/match.py:62`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `Config.qdrant_api_key: str = ""`, `Config.persistence: str = "memory"` (`"memory" | "dynamodb"`), `Config.checkpoint_table: str = "swiftai-checkpoints"`, `Config.runs_table: str = "swiftai-runs"`, `Config.dynamodb_endpoint: str = ""`, `Config.api_key: str = ""`. Environment overrides: `QDRANT_API_KEY`, `PERSISTENCE`, `CHECKPOINT_TABLE`, `RUNS_TABLE`, `DYNAMODB_ENDPOINT`, `RECRUITER_API_KEY`.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_config.py`:

```python
def test_aws_settings_can_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("QDRANT_API_KEY", "qdrant-secret")
    monkeypatch.setenv("PERSISTENCE", "dynamodb")
    monkeypatch.setenv("CHECKPOINT_TABLE", "swiftai-checkpoints")
    monkeypatch.setenv("RUNS_TABLE", "swiftai-runs")
    monkeypatch.setenv("DYNAMODB_ENDPOINT", "http://localhost:8002")
    monkeypatch.setenv("RECRUITER_API_KEY", "api-secret")

    config = load_config(RECRUITER)

    assert config.qdrant_api_key == "qdrant-secret"
    assert config.persistence == "dynamodb"
    assert (config.checkpoint_table, config.runs_table) == ("swiftai-checkpoints", "swiftai-runs")
    assert config.dynamodb_endpoint == "http://localhost:8002"
    assert config.api_key == "api-secret"
    assert "qdrant-secret" not in repr(config) and "api-secret" not in repr(config)


def test_persistence_defaults_to_memory_and_rejects_unknown_backends(monkeypatch):
    monkeypatch.delenv("PERSISTENCE", raising=False)
    assert load_config(RECRUITER).persistence == "memory"

    monkeypatch.setenv("PERSISTENCE", "redis")
    with pytest.raises(ValueError, match="persistence"):
        load_config(RECRUITER)
```

If `tests/test_config.py` does not import `pytest` yet, add `import pytest` at the top.

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_config.py -v`
Expected: the two new tests FAIL (`AttributeError: 'Config' object has no attribute 'qdrant_api_key'`).

- [ ] **Step 3: Implement** — in `src/agentic_jev_rag/config.py`:

Replace the `ENVIRONMENT_OVERRIDES` line with:

```python
ENVIRONMENT_OVERRIDES = {
    "QDRANT_URL": "qdrant_url",
    "QDRANT_API_KEY": "qdrant_api_key",
    "RAG_URL": "rag_url",
    "SYSTEM_1": "system_1",
    "PERSISTENCE": "persistence",
    "CHECKPOINT_TABLE": "checkpoint_table",
    "RUNS_TABLE": "runs_table",
    "DYNAMODB_ENDPOINT": "dynamodb_endpoint",
    "RECRUITER_API_KEY": "api_key",
}
```

Add these fields to `Config` after `rag_url`:

```python
    qdrant_api_key: str = field(default="", repr=False)  # Qdrant Cloud; empty for local Qdrant
    persistence: str = "memory"  # where interview checkpoints and run records live: memory or dynamodb
    checkpoint_table: str = "swiftai-checkpoints"
    runs_table: str = "swiftai-runs"
    dynamodb_endpoint: str = ""  # DynamoDB Local in tests; empty means AWS
    api_key: str = field(default="", repr=False)  # the Recruiter API's X-API-Key; empty turns auth off
```

In `load_config`, after the `system_1` check, add:

```python
    if data.get("persistence", "memory") not in ("memory", "dynamodb"):
        raise ValueError(f"persistence must be 'memory' or 'dynamodb', got {data['persistence']!r}")
```

Update the `load_config` docstring's override list to: "`QDRANT_URL`, `QDRANT_API_KEY`, `RAG_URL`, `SYSTEM_1`, `PERSISTENCE`, `CHECKPOINT_TABLE`, `RUNS_TABLE`, `DYNAMODB_ENDPOINT`, `RECRUITER_API_KEY` and `RECRUITER_URL` in the environment override the file."

Then pass the key at the three Qdrant call sites (`or None`: an empty key must not send an empty `api-key` header):

```python
# src/agentic_jev_rag/rag/api.py, create_default_rag_app
    qdrant = QdrantClient(url=config.qdrant_url, api_key=config.qdrant_api_key or None)
    service = RagService(config, OpenRouterGateway.from_env(), qdrant, load_corpus(config.corpus_dir))

# src/agentic_jev_rag/rag/ingest.py, main
    qdrant = QdrantClient(url=config.qdrant_url, api_key=config.qdrant_api_key or None)
    counts = index_corpus(documents, config, OpenRouterGateway.from_env(), qdrant)

# src/agentic_jev_rag/evals/match.py
    qdrant = QdrantClient(url=config.qdrant_url, api_key=config.qdrant_api_key or None)
    service = RagService(config, OpenRouterGateway.from_env(), qdrant, documents)
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/agentic_jev_rag/config.py src/agentic_jev_rag/rag/api.py src/agentic_jev_rag/rag/ingest.py src/agentic_jev_rag/evals/match.py tests/test_config.py
git commit -m "Config for Qdrant Cloud's API key, DynamoDB persistence and the Recruiter API key"
```

---

### Task 2: The persistence module and DynamoDB Local

**Files:**
- Modify: `pyproject.toml`, `uv.lock` (via `uv add`)
- Modify: `docker-compose.yml`, `tests/conftest.py`, `AGENTS.md`
- Create: `src/agentic_jev_rag/recruiter/persistence.py`
- Test: `tests/test_persistence.py`

**Interfaces:**
- Consumes: `Config.persistence`, `checkpoint_table`, `runs_table`, `dynamodb_endpoint` (Task 1).
- Produces:
  - `RunRecord(run_id: str, status: Literal["running", "complete", "failed"], started_at: float, report: dict | None = None, error: str | None = None)` with `.current(now: float) -> RunRecord`
  - `RunStore` protocol: `put(record: RunRecord) -> None`, `get(run_id: str) -> RunRecord | None`
  - `MemoryRuns()`, `DynamoRuns(table)`, `make_run_store(config: Config) -> RunStore`
  - `make_checkpointer(config: Config) -> BaseCheckpointSaver`
  - `STALE_AFTER_S = 1200`, `RETENTION_S = 604800`
  - Fixture `dynamodb_tables` → `dict` of Config overrides: `{"persistence": "dynamodb", "checkpoint_table": ..., "runs_table": ..., "dynamodb_endpoint": ...}`

- [ ] **Step 1: Add dependencies and DynamoDB Local**

Run: `uv add 'langgraph-checkpoint-aws>=1.2,<2' 'boto3>=1.40'`
Expected: resolves and updates `uv.lock`. If it reports a conflict with the locked `langgraph-checkpoint`, stop and report the conflict rather than downgrading LangGraph.

In `docker-compose.yml`, add after the `qdrant` service:

```yaml
  # DynamoDB Local: interview checkpoints and run records for the tests (AWS uses DynamoDB itself).
  dynamodb:
    image: amazon/dynamodb-local:3.3.1
    ports:
      - "8002:8000"
```

Run: `docker compose up -d qdrant dynamodb`
Expected: both containers running.

In `AGENTS.md`, change the test line to: "Run `uv run pytest` with Qdrant and DynamoDB Local up (`docker compose up -d qdrant dynamodb`)."

- [ ] **Step 2: Add the fixture** — append to `tests/conftest.py`:

```python
DYNAMODB_ENDPOINT = os.environ.get("DYNAMODB_ENDPOINT", "http://localhost:8002")


@pytest.fixture(scope="session")
def dynamodb_tables():
    """Fresh checkpoint and run tables in DynamoDB Local, with the key schemas infra/main/data.tf creates.

    Returns Config overrides that point the Recruiter's persistence at them."""
    import boto3

    client = boto3.client(
        "dynamodb", endpoint_url=DYNAMODB_ENDPOINT, region_name="us-east-1",
        aws_access_key_id="local", aws_secret_access_key="local",
    )
    try:
        client.list_tables()
    except Exception as exc:  # pragma: no cover - environment guard
        pytest.fail(f"DynamoDB Local is not reachable at {DYNAMODB_ENDPOINT}; run `docker compose up -d dynamodb` ({exc})")
    suffix = uuid.uuid4().hex[:8]
    checkpoints, runs = f"test-checkpoints-{suffix}", f"test-runs-{suffix}"
    client.create_table(
        TableName=checkpoints, BillingMode="PAY_PER_REQUEST",
        AttributeDefinitions=[{"AttributeName": "PK", "AttributeType": "S"}, {"AttributeName": "SK", "AttributeType": "S"}],
        KeySchema=[{"AttributeName": "PK", "KeyType": "HASH"}, {"AttributeName": "SK", "KeyType": "RANGE"}],
    )
    client.create_table(
        TableName=runs, BillingMode="PAY_PER_REQUEST",
        AttributeDefinitions=[{"AttributeName": "run_id", "AttributeType": "S"}],
        KeySchema=[{"AttributeName": "run_id", "KeyType": "HASH"}],
    )
    yield {"persistence": "dynamodb", "checkpoint_table": checkpoints, "runs_table": runs, "dynamodb_endpoint": DYNAMODB_ENDPOINT}
    for table in (checkpoints, runs):
        client.delete_table(TableName=table)
```

- [ ] **Step 3: Write the failing tests** — create `tests/test_persistence.py`:

```python
"""Durable state: run records and LangGraph checkpoints, in memory and in DynamoDB Local."""

from dataclasses import replace

import pytest

from agentic_jev_rag.recruiter.persistence import (
    STALE_AFTER_S,
    MemoryRuns,
    RunRecord,
    make_checkpointer,
    make_run_store,
)


@pytest.fixture(params=["memory", "dynamodb"])
def runs(request, config, dynamodb_tables):
    if request.param == "memory":
        return MemoryRuns()
    return make_run_store(replace(config, **dynamodb_tables))


def test_a_run_record_round_trips(runs):
    runs.put(RunRecord(run_id="r1", status="running", started_at=1000.0))
    runs.put(RunRecord(run_id="r1", status="complete", started_at=1000.0, report={"goal": "Fill it", "cost_usd": 0.41}))

    record = runs.get("r1")

    assert record.status == "complete"
    assert record.report == {"goal": "Fill it", "cost_usd": 0.41}
    assert record.started_at == 1000.0 and record.error is None


def test_a_failed_run_keeps_its_error(runs):
    runs.put(RunRecord(run_id="r2", status="failed", started_at=1000.0, error="OpenRouterError: down"))

    assert runs.get("r2").error == "OpenRouterError: down"


def test_an_unknown_run_is_none(runs):
    assert runs.get("nope") is None


def test_a_large_report_round_trips(runs):
    report = {"notes": ["x" * 1000 for _ in range(300)]}  # about 300 KB, under DynamoDB's 400 KB item limit

    runs.put(RunRecord(run_id="big", status="complete", started_at=1000.0, report=report))

    assert runs.get("big").report == report


def test_a_run_still_running_past_the_stale_window_reads_as_failed():
    record = RunRecord(run_id="r3", status="running", started_at=1000.0)

    assert record.current(1000.0 + STALE_AFTER_S - 1).status == "running"
    late = record.current(1000.0 + STALE_AFTER_S + 1)
    assert late.status == "failed" and "interrupted" in late.error
    assert RunRecord(run_id="r4", status="complete", started_at=0.0).current(1e9).status == "complete"


def test_memory_persistence_uses_an_in_memory_checkpointer(config):
    from langgraph.checkpoint.memory import InMemorySaver

    assert isinstance(make_checkpointer(config), InMemorySaver)
    assert isinstance(make_run_store(config), MemoryRuns)


def test_dynamodb_persistence_uses_the_dynamodb_saver(config, dynamodb_tables):
    from langgraph_checkpoint_aws import DynamoDBSaver

    assert isinstance(make_checkpointer(replace(config, **dynamodb_tables)), DynamoDBSaver)
```

- [ ] **Step 4: Run them to verify they fail**

Run: `uv run pytest tests/test_persistence.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agentic_jev_rag.recruiter.persistence'`.

- [ ] **Step 5: Implement** — create `src/agentic_jev_rag/recruiter/persistence.py`:

```python
"""Durable state for the Recruiter API: LangGraph checkpoints and run records, in memory or in DynamoDB.

Memory is the local and test default. On AWS both live in DynamoDB, so interviews and runs survive deploys and
any Recruiter API task can serve any request (ADR 0014)."""

import json
import threading
from dataclasses import dataclass, replace
from typing import Literal, Protocol

import boto3
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph_checkpoint_aws import DynamoDBSaver

from agentic_jev_rag.config import Config

RETENTION_S = 7 * 24 * 3600  # DynamoDB TTL on checkpoints and run records
STALE_AFTER_S = 20 * 60  # a full run takes about 3 minutes; one still running after 20 lost its task


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    status: Literal["running", "complete", "failed"]
    started_at: float
    report: dict | None = None
    error: str | None = None

    def current(self, now: float) -> "RunRecord":
        """This record as it should be reported at `now`: a run running past the stale window was interrupted."""
        if self.status == "running" and now - self.started_at > STALE_AFTER_S:
            return replace(self, status="failed", error="The run was interrupted (the service restarted); start it again.")
        return self


class RunStore(Protocol):
    def put(self, record: RunRecord) -> None: ...

    def get(self, run_id: str) -> RunRecord | None: ...


class MemoryRuns:
    def __init__(self) -> None:
        self._records: dict[str, RunRecord] = {}
        self._lock = threading.Lock()

    def put(self, record: RunRecord) -> None:
        with self._lock:
            self._records[record.run_id] = record

    def get(self, run_id: str) -> RunRecord | None:
        with self._lock:
            return self._records.get(run_id)


class DynamoRuns:
    """Run records in a DynamoDB table keyed by `run_id`; the report is stored as a JSON string."""

    def __init__(self, table) -> None:  # a boto3 DynamoDB Table resource
        self._table = table

    def put(self, record: RunRecord) -> None:
        item = {
            "run_id": record.run_id,
            "status": record.status,
            "started_at": int(record.started_at),
            "ttl": int(record.started_at) + RETENTION_S,
        }
        if record.report is not None:
            item["report"] = json.dumps(record.report)
        if record.error is not None:
            item["error"] = record.error
        self._table.put_item(Item=item)

    def get(self, run_id: str) -> RunRecord | None:
        item = self._table.get_item(Key={"run_id": run_id}).get("Item")
        if not item:
            return None
        return RunRecord(
            run_id=item["run_id"],
            status=item["status"],
            started_at=float(item["started_at"]),
            report=json.loads(item["report"]) if "report" in item else None,
            error=item.get("error"),
        )


def make_checkpointer(config: Config) -> BaseCheckpointSaver:
    if config.persistence == "memory":
        return InMemorySaver()
    return DynamoDBSaver(
        table_name=config.checkpoint_table,
        session=_session(config),
        endpoint_url=config.dynamodb_endpoint or None,
        ttl_seconds=RETENTION_S,
    )


def make_run_store(config: Config) -> RunStore:
    if config.persistence == "memory":
        return MemoryRuns()
    dynamodb = _session(config).resource("dynamodb", endpoint_url=config.dynamodb_endpoint or None)
    return DynamoRuns(dynamodb.Table(config.runs_table))


def _session(config: Config) -> boto3.Session:
    """Credentials and region from the environment (the ECS task role on AWS); fixed ones for DynamoDB Local."""
    if config.dynamodb_endpoint:
        return boto3.Session(aws_access_key_id="local", aws_secret_access_key="local", region_name="us-east-1")
    return boto3.Session()
```

Note: `started_at` is stored as whole seconds, so the round-trip tests use whole-second values.

- [ ] **Step 6: Run the tests**

Run: `uv run pytest tests/test_persistence.py -v`
Expected: PASS. If `DynamoDBSaver` rejects `session` together with `endpoint_url` (TypeError), replace `session=_session(config)` with `region_name="us-east-1"` and, for DynamoDB Local only, set `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` to `local` in `conftest.py` via `os.environ.setdefault`; re-run.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock docker-compose.yml AGENTS.md tests/conftest.py tests/test_persistence.py src/agentic_jev_rag/recruiter/persistence.py
git commit -m "Persistence module: LangGraph checkpoints and run records in memory or DynamoDB, tested against DynamoDB Local"
```

---

### Task 3: Interviews checkpoint to DynamoDB

**Files:**
- Modify: `src/agentic_jev_rag/recruiter/interview_graph.py:1-20,142`
- Modify: `src/agentic_jev_rag/recruiter/api.py` (`create_recruiter_app`, `start` description)
- Test: `tests/test_recruiter_api.py`

**Interfaces:**
- Consumes: `make_checkpointer(config)` (Task 2), fixture `dynamodb_tables` (Task 2).
- Produces: `build_interview_graph(ctx: RecruiterContext, checkpointer: BaseCheckpointSaver | None = None)` — `None` means a fresh `InMemorySaver()`.

- [ ] **Step 1: Write the failing test** — append to `tests/test_recruiter_api.py`:

```python
def test_an_interview_continues_on_another_instance_when_checkpoints_are_in_dynamodb(recruiter_context, dynamodb_tables):
    from dataclasses import replace

    gateway = recruiter_gateway()

    def instance():
        ctx = recruiter_context(gateway)
        ctx.config = replace(ctx.config, **dynamodb_tables)
        return TestClient(create_recruiter_app(ctx))

    first, second = instance(), instance()

    started = first.post("/interviews", json={"candidate_id": "C-001", "job_id": "J-001"}).json()
    next_question = second.post(f"/interviews/{started['interview_id']}/reply", json={"message": "I partitioned Kafka topics."}).json()
    assert next_question["question"] == "Tell me about question 2."
    done = first.post(f"/interviews/{started['interview_id']}/reply", json={"message": "I owned on-call."}).json()

    assert done["assessment"]["recommendation"] == "next_round"
    assert [t["answer"] for t in done["assessment"]["turns"]] == ["I partitioned Kafka topics.", "I owned on-call."]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_recruiter_api.py::test_an_interview_continues_on_another_instance_when_checkpoints_are_in_dynamodb -v`
Expected: FAIL — the second instance answers 404 (`unknown interview`), because each app has its own in-memory checkpointer.

- [ ] **Step 3: Implement**

In `src/agentic_jev_rag/recruiter/interview_graph.py`, add `from langgraph.checkpoint.base import BaseCheckpointSaver` next to the `InMemorySaver` import, change the builder's signature to `def build_interview_graph(ctx: RecruiterContext, checkpointer: BaseCheckpointSaver | None = None):` (keep its current return annotation, if any), and change its last line to:

```python
    return graph.compile(checkpointer=checkpointer or InMemorySaver())
```

Update the module docstring's first line to end "(interrupt + a checkpointer: in memory locally, DynamoDB on AWS)."

In `src/agentic_jev_rag/recruiter/api.py`, import `from agentic_jev_rag.recruiter.persistence import make_checkpointer` and replace `interviews = build_interview_graph(ctx)` with:

```python
    interviews = build_interview_graph(ctx, make_checkpointer(ctx.config))
```

In the `start` endpoint's description, replace "Interviews are kept in memory and are lost if the service restarts." with "Interviews are checkpointed (in DynamoDB when deployed), so they survive restarts."

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_recruiter_api.py tests/test_interview_graph.py tests/test_voice_interview.py -v`
Expected: PASS. If the two-instance test fails while deserialising a checkpoint (an error naming a pydantic class from `agentic_jev_rag.recruiter.schemas`), LangGraph's serializer is refusing an unregistered type: construct `DynamoDBSaver(..., serde=JsonPlusSerializer(allowed_msgpack_modules=[("agentic_jev_rag.recruiter.schemas", "<ClassName>")]))` for each class the error names, then re-run.

- [ ] **Step 5: Commit**

```bash
git add src/agentic_jev_rag/recruiter/interview_graph.py src/agentic_jev_rag/recruiter/api.py tests/test_recruiter_api.py
git commit -m "Interviews checkpoint through the configured persistence, so a DynamoDB-backed interview continues on any instance"
```

---

### Task 4: Asynchronous `/runs`

**Files:**
- Modify: `src/agentic_jev_rag/recruiter/api.py` (`/runs`, new `GET /runs/{run_id}`, `DESCRIPTION`, `create_recruiter_app` signature)
- Modify: `src/agentic_jev_rag/recruiter/api_models.py`
- Modify: `README.md:122-128`
- Test: `tests/test_recruiter_api.py`, `tests/test_recruiter_api_docs.py`

**Interfaces:**
- Consumes: `RunRecord`, `RunStore`, `MemoryRuns`, `make_run_store` (Task 2).
- Produces: `create_recruiter_app(ctx: RecruiterContext, *, runs: RunStore | None = None) -> FastAPI`; `POST /runs` → 202 `RunStatus`; `GET /runs/{run_id}` → 200 `RunStatus` (operation id `run_status`) or 404; `RunStatus(run_id: str, status: Literal["running","complete","failed"], report: RecruitmentReport | None, error: str | None)`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_recruiter_api.py`, add this helper below `recruiter_gateway`:

```python
def run(api, body):
    """Starts a run and reads its result; TestClient finishes background tasks before returning the response."""
    accepted = api.post("/runs", json=body)
    assert accepted.status_code == 202, accepted.text
    assert accepted.json()["status"] == "running"
    status = api.get(f"/runs/{accepted.json()['run_id']}").json()
    assert status["status"] == "complete", status
    return status["report"]
```

Rewrite the existing `/runs` calls to use it:

- `test_a_discriminatory_goal_is_refused_before_any_agent_runs`: `report = run(client(gateway), {"goal": "Backend engineers, men under 35 only", "job_id": "J-001"})`
- `test_a_full_run_produces_a_recruitment_report`: replace the first three lines with `report = run(client(recruiter_gateway()), {"goal": "Fill the senior backend role", "job_id": "J-001"})`
- `test_candidates_without_valid_consent_never_reach_the_shortlist`: `report = run(client(recruiter_gateway(shortlist=("C-004", "C-001"))), {"goal": "Fill the senior backend role", "job_id": "J-001"})`
- `test_ungrounded_verdicts_are_flagged_in_the_report`: `report = run(client(recruiter_gateway(grounded=False)), {"goal": "Fill the senior backend role", "job_id": "J-001"})`
- `test_runs_can_switch_to_the_jev_decider`: `report = run(client(recruiter_gateway()), {"goal": "Fill the senior backend role", "job_id": "J-001", "system_1": "jev", "simulate_interviews": False})`
- `test_a_duplicated_candidate_in_the_shortlist_still_gets_a_compliance_review`: `report = run(client(recruiter_gateway(shortlist=("C-001", "C-001", "C-003"))), {"goal": "Fill the senior backend role", "job_id": "J-001"})`

`test_a_run_for_an_unknown_job_is_404` stays as is (the Job is checked before the run is accepted).

Append the new tests:

```python
def test_a_run_whose_models_fail_is_reported_as_failed(client):
    from agentic_jev_rag.openrouter import OpenRouterError

    def down(model, messages, tools):
        raise OpenRouterError("OpenRouter is down")

    gateway = scripted_gateway(verdict="allowed", chat_responder=by_agent(Supervisor=down))
    api = client(gateway)

    run_id = api.post("/runs", json={"goal": "Fill the senior backend role", "job_id": "J-001"}).json()["run_id"]
    status = api.get(f"/runs/{run_id}").json()

    assert status["status"] == "failed"
    assert "OpenRouter is down" in status["error"] and status["report"] is None


def test_a_run_left_running_too_long_is_reported_as_interrupted(recruiter_context):
    from agentic_jev_rag.recruiter.persistence import MemoryRuns, RunRecord

    runs = MemoryRuns()
    runs.put(RunRecord(run_id="old", status="running", started_at=0.0))
    api = TestClient(create_recruiter_app(recruiter_context(recruiter_gateway()), runs=runs))

    status = api.get("/runs/old").json()

    assert status["status"] == "failed" and "interrupted" in status["error"]


def test_an_unknown_run_is_404(client):
    assert client(recruiter_gateway()).get("/runs/nope").status_code == 404
```

In `tests/test_recruiter_api_docs.py`:
- In `test_every_operation_is_summarised_described_tagged_and_named`, the expected set becomes `{"run_recruitment", "run_status", "start_interview", "reply_to_interview", "health"}`.
- In `test_error_responses_are_documented`, replace the `run_recruitment` line with:

```python
    assert {"202", "404"} <= set(ops["run_recruitment"]["responses"])
    assert {"404"} <= set(ops["run_status"]["responses"])
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_recruiter_api.py tests/test_recruiter_api_docs.py -v`
Expected: the rewritten run tests FAIL (`assert 200 == 202`), the new ones FAIL (404/405 on `GET /runs/...`, `TypeError` on `runs=`).

- [ ] **Step 3: Implement the model** — append to `src/agentic_jev_rag/recruiter/api_models.py` (add `from agentic_jev_rag.recruiter.report import RecruitmentReport` to the imports):

```python
class RunStatus(BaseModel):
    """Where a recruitment run stands. Poll `GET /runs/{run_id}` until `status` is `complete` or `failed`."""

    run_id: str = Field(description="Id to poll with `GET /runs/{run_id}`.", examples=["5f0c3b7e9a8d4c2e8b1a6f4d3c2b1a09"])
    status: Literal["running", "complete", "failed"] = Field(
        description="`running` until the Recruiter finishes; then `complete` with the report, or `failed` with the error."
    )
    report: RecruitmentReport | None = Field(default=None, description="The Recruitment report, once `complete`.")
    error: str | None = Field(
        default=None,
        description="Why the run failed: an upstream failure (OpenRouter, the RAG API, invalid model output) or an "
        "interrupted run.",
    )
```

If importing `RecruitmentReport` into `api_models.py` creates a circular import, define `RunStatus` in `api.py` instead, right above `create_recruiter_app`.

- [ ] **Step 4: Implement the endpoints** — in `src/agentic_jev_rag/recruiter/api.py`:

Imports: add `import time`, `BackgroundTasks` to the `fastapi` import, `RunStatus` to the `api_models` import, and `from agentic_jev_rag.recruiter.persistence import RunRecord, RunStore, make_checkpointer, make_run_store`.

In `DESCRIPTION`, replace the paragraph "A full run takes a few minutes and costs about $0.40 in model calls. Upstream failures (OpenRouter, the RAG API, or invalid model output) return **502**." with:

```
A full run takes a few minutes and costs about $0.40 in model calls, so runs are asynchronous: `POST /runs`
returns `202` with a `run_id`, and `GET /runs/{run_id}` reports `running`, `complete` (with the report) or
`failed`. Upstream failures (OpenRouter, the RAG API, or invalid model output) fail the run, or return **502**
from the interview endpoints.
```

Add a 202 response doc constant next to `NOT_FOUND`:

```python
ACCEPTED = {"description": "The run started; poll `GET /runs/{run_id}`."}
```

Change the signature and set up the store:

```python
def create_recruiter_app(ctx: RecruiterContext, *, runs: RunStore | None = None) -> FastAPI:
    ...
    interviews = build_interview_graph(ctx, make_checkpointer(ctx.config))
    runs = runs or make_run_store(ctx.config)
```

Replace the whole `/runs` endpoint with:

```python
    @app.post(
        "/runs",
        status_code=202,
        tags=["Recruitment"],
        operation_id="run_recruitment",
        summary="Start a recruitment run",
        description=(
            "Starts the whole Recruiter graph for a goal and a Job and returns at once with a `run_id`. Poll "
            "`GET /runs/{run_id}` for the Recruitment report; a full run takes a few minutes. A refused goal "
            "completes at once, with `refused: true` and an `input_guardrail` finding."
        ),
        response_description="The run id, with status `running`.",
        responses={202: ACCEPTED, 404: NOT_FOUND},
    )
    def run(request: Annotated[RunRequest, Body(openapi_examples=RUN_EXAMPLES)], background: BackgroundTasks) -> RunStatus:
        _require(ctx, request.job_id, "jobs")
        record = RunRecord(run_id=uuid.uuid4().hex, status="running", started_at=time.time())
        runs.put(record)
        background.add_task(execute, record, request)
        return _run_status(record)

    def execute(record: RunRecord, request: RunRequest) -> None:
        run_ctx = replace(ctx, system_1=request.system_1 or ctx.system_1)
        try:
            report = run_recruitment(
                run_ctx, request.goal, request.job_id,
                simulate_interviews=request.simulate_interviews, max_interviews=request.max_interviews,
            )
        except Exception as error:  # a background run has no response to fail; GET /runs/{run_id} reports it
            runs.put(replace(record, status="failed", error=f"{type(error).__name__}: {error}"))
        else:
            runs.put(replace(record, status="complete", report=report.model_dump(mode="json")))

    @app.get(
        "/runs/{run_id}",
        tags=["Recruitment"],
        operation_id="run_status",
        summary="Get a recruitment run",
        description=(
            "Returns the run's status: `running`, `complete` with the Recruitment report, or `failed` with the "
            "error. A run still `running` 20 minutes after it started was interrupted and is reported as `failed`."
        ),
        response_description="The run's status and, once complete, its report.",
        responses={404: NOT_FOUND},
    )
    def run_status(
        run_id: Annotated[
            str,
            Path(
                description="The id returned when the run was started.",
                openapi_examples={"run": {"summary": "A run id", "value": "5f0c3b7e9a8d4c2e8b1a6f4d3c2b1a09"}},
            ),
        ],
    ) -> RunStatus:
        record = runs.get(run_id)
        if record is None:
            raise HTTPException(404, f"unknown run {run_id}")
        return _run_status(record)
```

Add a module-level helper next to `_state`:

```python
def _run_status(record: RunRecord) -> RunStatus:
    current = record.current(time.time())
    return RunStatus(run_id=current.run_id, status=current.status, report=current.report, error=current.error)
```

Remove the now-unused `RecruitmentReport` import from `api.py` only if nothing else in the file uses it.

- [ ] **Step 5: Update the README's curl examples** (`README.md`, the "Try it" block):

```sh
# A discriminatory goal is refused before any agent runs (poll the run id it returns)
curl -s localhost:8001/runs -H 'Content-Type: application/json' \
  -d '{"goal": "Fill J-037 with young graduates only", "job_id": "J-037"}'

# A full recruitment (about 3 minutes, about $0.40): start it, then poll until status is complete
curl -s localhost:8001/runs -H 'Content-Type: application/json' \
  -d '{"goal": "Fill the Senior Linux Administrator role", "job_id": "J-037", "system_1": "jev", "max_interviews": 1}'
curl -s localhost:8001/runs/<run_id>
```

Also grep the README and `docs/` (excluding `docs/superpowers/` and `docs/parked/`) for other `POST /runs` descriptions that say it returns the report directly, and update them the same way: `grep -rn "/runs" README.md docs --include=*.md | grep -v superpowers | grep -v parked`.

- [ ] **Step 6: Run the tests**

Run: `uv run pytest tests/test_recruiter_api.py tests/test_recruiter_api_docs.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/agentic_jev_rag/recruiter/api.py src/agentic_jev_rag/recruiter/api_models.py tests/test_recruiter_api.py tests/test_recruiter_api_docs.py README.md
git commit -m "Runs are asynchronous: POST /runs returns 202 and GET /runs/{run_id} reports running, complete or failed"
```

---

### Task 5: API key authentication

**Files:**
- Modify: `src/agentic_jev_rag/recruiter/api.py`
- Modify: `src/agentic_jev_rag/voice/recruiter_client.py`, `src/agentic_jev_rag/voice/worker.py:63`
- Test: `tests/test_recruiter_api.py`, `tests/test_recruiter_api_docs.py`, `tests/test_voice_interview.py`

**Interfaces:**
- Consumes: `Config.api_key` (Task 1).
- Produces: header `X-API-Key` required on `/runs`, `/runs/{run_id}`, `/interviews`, `/interviews/{interview_id}/reply` when `ctx.config.api_key` is non-empty; 401 otherwise. `RecruiterClient(base_url, *, api_key: str = "", http=None, timeout_s=60.0)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_recruiter_api.py`:

```python
@pytest.fixture
def keyed(recruiter_context):
    from dataclasses import replace

    ctx = recruiter_context(recruiter_gateway())
    ctx.config = replace(ctx.config, api_key="s3cret")
    return TestClient(create_recruiter_app(ctx))


def test_with_an_api_key_set_every_route_but_health_and_docs_needs_it(keyed):
    start = {"candidate_id": "C-001", "job_id": "J-001"}

    assert keyed.post("/interviews", json=start).status_code == 401
    assert keyed.post("/interviews", json=start, headers={"X-API-Key": "wrong"}).status_code == 401
    assert keyed.post("/runs", json={"goal": "Fill it", "job_id": "J-001"}).status_code == 401
    assert keyed.get("/runs/any").status_code == 401
    assert keyed.post("/interviews/any/reply", json={"message": "hi"}).status_code == 401
    assert keyed.get("/health").status_code == 200
    assert keyed.get("/docs").status_code == 200 and keyed.get("/openapi.json").status_code == 200

    started = keyed.post("/interviews", json=start, headers={"X-API-Key": "s3cret"})
    assert started.status_code == 200 and started.json()["question"]


def test_without_an_api_key_set_auth_is_off(client):
    assert client(recruiter_gateway()).post("/interviews", json={"candidate_id": "C-001", "job_id": "J-001"}).status_code == 200
```

Append to `tests/test_recruiter_api_docs.py`:

```python
def test_the_api_key_is_a_documented_security_scheme(openapi):
    [scheme] = openapi["components"]["securitySchemes"].values()
    assert scheme == {**scheme, "type": "apiKey", "in": "header", "name": "X-API-Key"}
    assert scheme.get("description")
    ops = operations(openapi)
    assert ops["start_interview"].get("security") and "security" not in ops["health"]
```

In `tests/test_voice_interview.py`, look at the helper around line 21 that builds `RecruiterClient("http://recruiter", http=http)` over an in-process app, and add a test in the same style (reuse that file's existing app-building fixture/helper; the names below assume it is called `recruiter_app(gateway)` — use the real one):

```python
async def test_the_voice_client_sends_the_api_key(recruiter_context):
    from dataclasses import replace

    import httpx

    from agentic_jev_rag.recruiter.api import create_recruiter_app
    from agentic_jev_rag.voice.recruiter_client import RecruiterClient, RecruiterError

    ctx = recruiter_context(interview_gateway())
    ctx.config = replace(ctx.config, api_key="s3cret")
    app = create_recruiter_app(ctx)

    def client(key):
        http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://recruiter")
        return RecruiterClient("http://recruiter", api_key=key, http=http)

    started = await client("s3cret").start("C-001", "J-001")
    assert started["question"]
    with pytest.raises(RecruiterError) as refused:
        await client("").start("C-001", "J-001")
    assert "401" in str(refused.value) and refused.value.retryable is False
```

Mark it the same way the file's other async tests are marked (if they use `@pytest.mark.anyio` or `asyncio.run`, follow that), and import `interview_gateway` the way the file already does.

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_recruiter_api.py tests/test_recruiter_api_docs.py tests/test_voice_interview.py -v`
Expected: the new tests FAIL (200 instead of 401; no `securitySchemes`; `TypeError: unexpected keyword argument 'api_key'`).

- [ ] **Step 3: Implement the API side** — in `src/agentic_jev_rag/recruiter/api.py`:

Imports: add `import secrets`, `Depends` and `Security` to the `fastapi` import, and `from fastapi.security import APIKeyHeader`.

Module level, next to the other response constants:

```python
API_KEY = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
    description="The service's API key. Required on every route but /health when the service has one configured.",
)
UNAUTHORISED = {"model": ErrorOut, "description": "Missing or wrong `X-API-Key`."}
```

Inside `create_recruiter_app`, before the first route:

```python
    def authorised(key: Annotated[str | None, Security(API_KEY)] = None) -> None:
        expected = ctx.config.api_key
        if expected and not (key and secrets.compare_digest(key, expected)):
            raise HTTPException(401, "missing or wrong X-API-Key")

    protected = [Depends(authorised)]
```

Add `dependencies=protected,` to the decorators of `run`, `run_status`, `start` and `reply`, and add `401: UNAUTHORISED` to each of their `responses` dicts. Leave `/health` alone.

- [ ] **Step 4: Implement the voice side**

In `src/agentic_jev_rag/voice/recruiter_client.py`:

```python
class RecruiterClient:
    def __init__(self, base_url: str, *, api_key: str = "", http: httpx.AsyncClient | None = None, timeout_s: float = 60.0):
        """`http` lets tests pass an in-process client (httpx.ASGITransport over the Recruiter app)."""
        self._http = http or httpx.AsyncClient(base_url=base_url, timeout=timeout_s)
        self._headers = {"X-API-Key": api_key} if api_key else {}
```

and in `_send`: `return await self._http.post(path, json=body, headers=self._headers)`.

In `src/agentic_jev_rag/voice/worker.py:63`: `client = RecruiterClient(config.voice.recruiter_url, api_key=config.api_key)`.

- [ ] **Step 5: Run the whole suite**

Run: `uv run pytest`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/agentic_jev_rag/recruiter/api.py src/agentic_jev_rag/voice/recruiter_client.py src/agentic_jev_rag/voice/worker.py tests/test_recruiter_api.py tests/test_recruiter_api_docs.py tests/test_voice_interview.py
git commit -m "X-API-Key on every Recruiter API route but /health when a key is configured; the voice worker sends it"
```

---

### Task 6: Docker build targets for the API and the voice worker

**Files:**
- Modify: `Dockerfile`, `docker-compose.yml`

**Interfaces:**
- Produces: `docker build --target api` (default, unchanged behaviour) and `--target voice` (CMD `python -m agentic_jev_rag.voice start`, model files baked in).

- [ ] **Step 1: Rewrite the Dockerfile**

```dockerfile
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
RUN uv sync --frozen --no-dev --group voice && python -m agentic_jev_rag.voice download-files
CMD ["python", "-m", "agentic_jev_rag.voice", "start"]

FROM base AS api
EXPOSE 8000
CMD ["uvicorn", "agentic_jev_rag.rag.api:create_default_rag_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
```

In `docker-compose.yml`, change each `build: .` (ingest, rag-api, recruiter-api) to:

```yaml
    build:
      context: .
      target: api
```

- [ ] **Step 2: Build both targets for arm64**

Run: `docker build --platform linux/arm64 --target api -t swiftai/api:local . && docker build --platform linux/arm64 --target voice -t swiftai/voice:local .`
Expected: both succeed; the voice build logs model downloads.

- [ ] **Step 3: Smoke-test the images**

Run: `docker run --rm swiftai/voice:local python -c "import livekit.agents, agentic_jev_rag.voice.worker; print('ok')"`
Expected: `ok`.

Run: `docker compose up -d --build rag-api recruiter-api && sleep 5 && curl -fsS localhost:8001/health && docker compose down`
Expected: `{"status":"ok"}`. (If `sleep` is blocked in your shell, poll `/health` instead.)

- [ ] **Step 4: Commit**

```bash
git add Dockerfile docker-compose.yml
git commit -m "Dockerfile builds an api image and a voice image with its model files baked in"
```

---

### Task 7: Terraform bootstrap (state bucket and GitHub OIDC role)

**Files:**
- Create: `infra/bootstrap/main.tf`
- Modify: `.gitignore`

**Interfaces:**
- Produces: outputs `state_bucket` (`swiftai-tfstate-<account_id>`) and `deploy_role_arn`. GitHub environment `production` may assume the role.

- [ ] **Step 1: Upgrade Terraform and check AWS access**

Run: `brew install hashicorp/tap/terraform && terraform version`
Expected: v1.10 or later.

Run: `aws sts get-caller-identity --profile <profile>` and `export AWS_PROFILE=<profile> AWS_REGION=us-east-1`
Expected: the target account. Stop here if this fails.

- [ ] **Step 2: Ignore Terraform's local files** — append to `.gitignore`:

```
# Terraform: local state and plugins (the lock file is committed)
.terraform/
*.tfstate
*.tfstate.*
*.tfvars
!*.tfvars.example
# Secret values for infra/scripts/put-secrets.sh
.env.aws
```

- [ ] **Step 3: Write `infra/bootstrap/main.tf`**

```hcl
# Applied once by hand, with local state: the bucket the main stack keeps its state in, and the role
# GitHub Actions assumes through OIDC to deploy (no AWS keys in GitHub).

terraform {
  required_version = ">= 1.10"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.68"
    }
  }
}

variable "region" {
  type    = string
  default = "us-east-1"
}

variable "github_repository" {
  type    = string
  default = "hguazza/agentic-jev-rag"
}

provider "aws" {
  region = var.region
  default_tags {
    tags = { Project = "swiftai", ManagedBy = "terraform", Stack = "bootstrap" }
  }
}

data "aws_caller_identity" "current" {}

resource "aws_s3_bucket" "state" {
  bucket = "swiftai-tfstate-${data.aws_caller_identity.current.account_id}"
}

resource "aws_s3_bucket_versioning" "state" {
  bucket = aws_s3_bucket.state.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "state" {
  bucket = aws_s3_bucket.state.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "state" {
  bucket                  = aws_s3_bucket.state.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_iam_openid_connect_provider" "github" {
  url            = "https://token.actions.githubusercontent.com"
  client_id_list = ["sts.amazonaws.com"]
}

data "aws_iam_policy_document" "github_trust" {
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]
    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.github.arn]
    }
    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }
    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:sub"
      values   = ["repo:${var.github_repository}:environment:production"]
    }
  }
}

# AdministratorAccess because Terraform creates IAM roles; scoping this down is a known follow-up (spec, CI/CD).
resource "aws_iam_role" "deploy" {
  name               = "swiftai-github-deploy"
  assume_role_policy = data.aws_iam_policy_document.github_trust.json
}

resource "aws_iam_role_policy_attachment" "deploy" {
  role       = aws_iam_role.deploy.name
  policy_arn = "arn:aws:iam::aws:policy/AdministratorAccess"
}

output "state_bucket" {
  value = aws_s3_bucket.state.bucket
}

output "deploy_role_arn" {
  value = aws_iam_role.deploy.arn
}
```

- [ ] **Step 4: Validate, then apply**

Run: `terraform -chdir=infra/bootstrap init && terraform -chdir=infra/bootstrap fmt -check && terraform -chdir=infra/bootstrap validate`
Expected: `Success! The configuration is valid.`

Run: `terraform -chdir=infra/bootstrap apply`
Expected: after the human reviews and approves the plan, 8 resources created and both outputs printed. (If the account already has a GitHub OIDC provider, the create fails with `EntityAlreadyExists`: import it with `terraform -chdir=infra/bootstrap import aws_iam_openid_connect_provider.github arn:aws:iam::<account>:oidc-provider/token.actions.githubusercontent.com` and re-apply.)

- [ ] **Step 5: Write the backend config for the main stack**

Run: `mkdir -p infra/main && printf 'bucket = "%s"\n' "$(terraform -chdir=infra/bootstrap output -raw state_bucket)" > infra/main/backend.hcl && cat infra/main/backend.hcl`
Expected: `bucket = "swiftai-tfstate-<account_id>"`.

- [ ] **Step 6: Commit**

```bash
git add .gitignore infra/bootstrap/main.tf infra/bootstrap/.terraform.lock.hcl infra/main/backend.hcl
git commit -m "Terraform bootstrap: an S3 state bucket and the GitHub OIDC role deployments assume"
```

---

### Task 8: Terraform main stack — network, registries, data, secrets, IAM

**Files:**
- Create: `infra/main/versions.tf`, `variables.tf`, `network.tf`, `ecr.tf`, `data.tf`, `secrets.tf`, `iam.tf`, `terraform.tfvars.example`

**Interfaces:**
- Produces (used by Tasks 9–10): `aws_vpc.main`, `aws_subnet.private[*]`, `aws_subnet.public[*]`, `aws_ecr_repository.image["api"|"voice"]`, `aws_dynamodb_table.checkpoints|runs`, `aws_secretsmanager_secret.app[<NAME>]`, `aws_iam_role.execution`, `aws_iam_role.recruiter`, variables `region`, `project`, `image_tag`, `running`, `alert_email`, `qdrant_url`, `system_1`, `monthly_budget_usd`.

- [ ] **Step 1: `infra/main/versions.tf`**

```hcl
terraform {
  required_version = ">= 1.10"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.68"
    }
  }
  # bucket comes from backend.hcl: terraform init -backend-config=backend.hcl
  backend "s3" {
    key          = "swiftai/main.tfstate"
    region       = "us-east-1"
    encrypt      = true
    use_lockfile = true
  }
}

provider "aws" {
  region = var.region
  default_tags {
    tags = { Project = var.project, ManagedBy = "terraform", Stack = "main" }
  }
}
```

- [ ] **Step 2: `infra/main/variables.tf`**

```hcl
variable "region" {
  type    = string
  default = "us-east-1"
}

variable "project" {
  type    = string
  default = "swiftai"
}

variable "image_tag" {
  description = "Git SHA the api and voice images are tagged with in ECR."
  type        = string
}

variable "running" {
  description = "false scales every service to zero and removes the NAT gateway, for between demos."
  type        = bool
  default     = true
}

variable "alert_email" {
  description = "Receives alarm and budget emails (confirm the SNS subscription email once)."
  type        = string
}

variable "qdrant_url" {
  description = "The Qdrant Cloud cluster URL (its API key is in Secrets Manager)."
  type        = string
}

variable "system_1" {
  description = "Default System 1 decider: llm or jev."
  type        = string
  default     = "llm"
}

variable "monthly_budget_usd" {
  type    = number
  default = 60
}
```

And `infra/main/terraform.tfvars.example`:

```hcl
# Copy to terraform.tfvars (git-ignored). image_tag is passed on the command line by deployments.
alert_email = "you@example.com"
qdrant_url  = "https://<cluster-id>.us-east-1-0.aws.cloud.qdrant.io:6333"
```

- [ ] **Step 3: `infra/main/network.tf`**

```hcl
# Two AZs. Public subnets hold only the NAT gateway; tasks and the internal ALB sit in private subnets.
# One NAT gateway, not one per AZ: a deliberate cost trade (spec, Network).

data "aws_availability_zones" "available" {
  state = "available"
}

locals {
  azs  = slice(data.aws_availability_zones.available.names, 0, 2)
  cidr = "10.20.0.0/16"
}

resource "aws_vpc" "main" {
  cidr_block           = local.cidr
  enable_dns_support   = true
  enable_dns_hostnames = true
  tags                 = { Name = var.project }
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id
}

resource "aws_subnet" "public" {
  count             = 2
  vpc_id            = aws_vpc.main.id
  cidr_block        = cidrsubnet(local.cidr, 8, count.index)
  availability_zone = local.azs[count.index]
  tags              = { Name = "${var.project}-public-${local.azs[count.index]}" }
}

resource "aws_subnet" "private" {
  count             = 2
  vpc_id            = aws_vpc.main.id
  cidr_block        = cidrsubnet(local.cidr, 8, count.index + 10)
  availability_zone = local.azs[count.index]
  tags              = { Name = "${var.project}-private-${local.azs[count.index]}" }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }
}

resource "aws_route_table_association" "public" {
  count          = 2
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_eip" "nat" {
  count  = var.running ? 1 : 0
  domain = "vpc"
}

resource "aws_nat_gateway" "main" {
  count         = var.running ? 1 : 0
  allocation_id = aws_eip.nat[0].id
  subnet_id     = aws_subnet.public[0].id
  depends_on    = [aws_internet_gateway.main]
}

resource "aws_route_table" "private" {
  vpc_id = aws_vpc.main.id
}

resource "aws_route" "private_egress" {
  count                  = var.running ? 1 : 0
  route_table_id         = aws_route_table.private.id
  destination_cidr_block = "0.0.0.0/0"
  nat_gateway_id         = aws_nat_gateway.main[0].id
}

resource "aws_route_table_association" "private" {
  count          = 2
  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private.id
}

# Free: ECR image layers come from S3 without crossing the NAT gateway.
resource "aws_vpc_endpoint" "s3" {
  vpc_id            = aws_vpc.main.id
  service_name      = "com.amazonaws.${var.region}.s3"
  vpc_endpoint_type = "Gateway"
  route_table_ids   = [aws_route_table.private.id]
}
```

- [ ] **Step 4: `infra/main/ecr.tf`**

```hcl
resource "aws_ecr_repository" "image" {
  for_each             = toset(["api", "voice"])
  name                 = "${var.project}/${each.key}"
  image_tag_mutability = "IMMUTABLE"
  force_delete         = true
  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "image" {
  for_each   = aws_ecr_repository.image
  repository = each.value.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep the last 20 images"
      selection    = { tagStatus = "any", countType = "imageCountMoreThan", countNumber = 20 }
      action       = { type = "expire" }
    }]
  })
}
```

- [ ] **Step 5: `infra/main/data.tf`**

```hcl
# Interview checkpoints, with the key schema langgraph-checkpoint-aws's DynamoDBSaver expects.
resource "aws_dynamodb_table" "checkpoints" {
  name         = "${var.project}-checkpoints"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "PK"
  range_key    = "SK"
  attribute {
    name = "PK"
    type = "S"
  }
  attribute {
    name = "SK"
    type = "S"
  }
  ttl {
    attribute_name = "ttl"
    enabled        = true
  }
  point_in_time_recovery {
    enabled = true
  }
  server_side_encryption {
    enabled = true
  }
}

# Recruitment run records: status, report and error per run_id.
resource "aws_dynamodb_table" "runs" {
  name         = "${var.project}-runs"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "run_id"
  attribute {
    name = "run_id"
    type = "S"
  }
  ttl {
    attribute_name = "ttl"
    enabled        = true
  }
  point_in_time_recovery {
    enabled = true
  }
  server_side_encryption {
    enabled = true
  }
}
```

- [ ] **Step 6: `infra/main/secrets.tf`**

```hcl
# Terraform creates the secrets, never their values: infra/scripts/put-secrets.sh sets them from .env.aws.
locals {
  secret_names = [
    "OPEN_ROUTER_API_TOKEN", "QDRANT_API_KEY", "RECRUITER_API_KEY",
    "LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET",
  ]
}

resource "aws_secretsmanager_secret" "app" {
  for_each                = toset(local.secret_names)
  name                    = "${var.project}/${each.key}"
  recovery_window_in_days = 0
}
```

- [ ] **Step 7: `infra/main/iam.tf`**

```hcl
data "aws_iam_policy_document" "ecs_tasks_trust" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

# Pulls images, writes logs and injects secrets for every task.
resource "aws_iam_role" "execution" {
  name               = "${var.project}-ecs-execution"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_trust.json
}

resource "aws_iam_role_policy_attachment" "execution" {
  role       = aws_iam_role.execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

data "aws_iam_policy_document" "read_secrets" {
  statement {
    actions   = ["secretsmanager:GetSecretValue"]
    resources = [for secret in aws_secretsmanager_secret.app : secret.arn]
  }
}

resource "aws_iam_role_policy" "execution_secrets" {
  role   = aws_iam_role.execution.id
  policy = data.aws_iam_policy_document.read_secrets.json
}

# Only the Recruiter API calls AWS at runtime: its checkpoints and run records.
resource "aws_iam_role" "recruiter" {
  name               = "${var.project}-recruiter-api"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_trust.json
}

data "aws_iam_policy_document" "recruiter" {
  statement {
    actions = [
      "dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:DeleteItem", "dynamodb:Query",
      "dynamodb:BatchGetItem", "dynamodb:BatchWriteItem",
    ]
    resources = [aws_dynamodb_table.checkpoints.arn, aws_dynamodb_table.runs.arn]
  }
}

resource "aws_iam_role_policy" "recruiter" {
  role   = aws_iam_role.recruiter.id
  policy = data.aws_iam_policy_document.recruiter.json
}
```

- [ ] **Step 8: Validate**

Run: `terraform -chdir=infra/main init -backend=false && terraform -chdir=infra/main fmt -check && terraform -chdir=infra/main validate`
Expected: `Success! The configuration is valid.` Fix any schema error the provider reports (it is the authority on argument names), then re-run.

- [ ] **Step 9: Commit**

```bash
git add infra/main/versions.tf infra/main/variables.tf infra/main/network.tf infra/main/ecr.tf infra/main/data.tf infra/main/secrets.tf infra/main/iam.tf infra/main/terraform.tfvars.example infra/main/.terraform.lock.hcl
git commit -m "Terraform main stack: VPC with one NAT gateway, ECR, DynamoDB tables, Secrets Manager and task roles"
```

---

### Task 9: Terraform — ECS services, service discovery, autoscaling

**Files:**
- Create: `infra/main/ecs.tf`

**Interfaces:**
- Consumes: everything Task 8 produces; `aws_lb_target_group.recruiter`, `aws_lb_listener.http`, `aws_lb.recruiter`, `aws_security_group.alb` (Task 10 — write Task 10's `edge.tf` before running `validate` here, or validate both together at the end of Task 10).
- Produces: `aws_ecs_cluster.main`, `aws_security_group.task[<name>]`, `aws_ecs_task_definition.task[<name>]` for `rag-api`, `recruiter-api`, `voice-worker`, `ingest`; services for the first three; DNS names `rag-api.swiftai.internal`, `recruiter-api.swiftai.internal`; `local.scaling`.

- [ ] **Step 1: Write `infra/main/ecs.tf`**

```hcl
resource "aws_ecs_cluster" "main" {
  name = var.project
  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

# Names any client in the VPC resolves (Cloud Map DNS, not Service Connect), so AgentCore can use them in phase 2.
resource "aws_service_discovery_private_dns_namespace" "internal" {
  name = "${var.project}.internal"
  vpc  = aws_vpc.main.id
}

locals {
  api_image   = "${aws_ecr_repository.image["api"].repository_url}:${var.image_tag}"
  voice_image = "${aws_ecr_repository.image["voice"].repository_url}:${var.image_tag}"
  internal    = "${var.project}.internal"

  # Every task the stack runs; all but ingest are long-running services.
  tasks = {
    "rag-api" = {
      image   = local.api_image
      cpu     = 512
      memory  = 1024
      role    = null
      command = ["uvicorn", "agentic_jev_rag.rag.api:create_default_rag_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
      env     = { QDRANT_URL = var.qdrant_url, SYSTEM_1 = var.system_1 }
      secrets = ["OPEN_ROUTER_API_TOKEN", "QDRANT_API_KEY"]
    }
    "recruiter-api" = {
      image   = local.api_image
      cpu     = 512
      memory  = 1024
      role    = aws_iam_role.recruiter.arn
      command = ["uvicorn", "agentic_jev_rag.recruiter.api:create_default_recruiter_app", "--factory", "--host", "0.0.0.0", "--port", "8001"]
      env = {
        RAG_URL          = "http://rag-api.${local.internal}:8000"
        SYSTEM_1         = var.system_1
        PERSISTENCE      = "dynamodb"
        CHECKPOINT_TABLE = aws_dynamodb_table.checkpoints.name
        RUNS_TABLE       = aws_dynamodb_table.runs.name
      }
      secrets = ["OPEN_ROUTER_API_TOKEN", "RECRUITER_API_KEY"]
    }
    "voice-worker" = {
      image   = local.voice_image
      cpu     = 1024
      memory  = 2048
      role    = null
      command = ["python", "-m", "agentic_jev_rag.voice", "start"]
      env     = { RECRUITER_URL = "http://recruiter-api.${local.internal}:8001" }
      secrets = ["OPEN_ROUTER_API_TOKEN", "RECRUITER_API_KEY", "LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"]
    }
    "ingest" = {
      image   = local.api_image
      cpu     = 512
      memory  = 1024
      role    = null
      command = ["python", "-m", "agentic_jev_rag.rag.ingest"]
      env     = { QDRANT_URL = var.qdrant_url }
      secrets = ["OPEN_ROUTER_API_TOKEN", "QDRANT_API_KEY"]
    }
  }

  # Services with an HTTP port get a health check and a Cloud Map name.
  ports = { "rag-api" = 8000, "recruiter-api" = 8001 }

  scaling = {
    "rag-api"       = { min = 2, max = 4 }
    "recruiter-api" = { min = 2, max = 4 }
    "voice-worker"  = { min = 1, max = 1 }
  }
}

resource "aws_cloudwatch_log_group" "task" {
  for_each          = local.tasks
  name              = "/ecs/${var.project}/${each.key}"
  retention_in_days = 14
}

resource "aws_security_group" "task" {
  for_each = local.tasks
  name     = "${var.project}-${each.key}"
  vpc_id   = aws_vpc.main.id
}

# Every task calls the internet: OpenRouter, Qdrant Cloud, LiveKit (through the NAT gateway).
resource "aws_vpc_security_group_egress_rule" "task_internet" {
  for_each          = aws_security_group.task
  security_group_id = each.value.id
  cidr_ipv4         = "0.0.0.0/0"
  ip_protocol       = "-1"
}

resource "aws_vpc_security_group_ingress_rule" "recruiter_from_alb" {
  security_group_id            = aws_security_group.task["recruiter-api"].id
  referenced_security_group_id = aws_security_group.alb.id
  ip_protocol                  = "tcp"
  from_port                    = 8001
  to_port                      = 8001
}

resource "aws_vpc_security_group_ingress_rule" "recruiter_from_voice" {
  security_group_id            = aws_security_group.task["recruiter-api"].id
  referenced_security_group_id = aws_security_group.task["voice-worker"].id
  ip_protocol                  = "tcp"
  from_port                    = 8001
  to_port                      = 8001
}

resource "aws_vpc_security_group_ingress_rule" "rag_from_recruiter" {
  security_group_id            = aws_security_group.task["rag-api"].id
  referenced_security_group_id = aws_security_group.task["recruiter-api"].id
  ip_protocol                  = "tcp"
  from_port                    = 8000
  to_port                      = 8000
}

resource "aws_ecs_task_definition" "task" {
  for_each                 = local.tasks
  family                   = "${var.project}-${each.key}"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = each.value.cpu
  memory                   = each.value.memory
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = each.value.role
  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "ARM64"
  }
  container_definitions = jsonencode([{
    name        = each.key
    image       = each.value.image
    essential   = true
    command     = each.value.command
    stopTimeout = 120
    environment = [for name, value in each.value.env : { name = name, value = value }]
    secrets     = [for name in each.value.secrets : { name = name, valueFrom = aws_secretsmanager_secret.app[name].arn }]
    portMappings = [for port in try([local.ports[each.key]], []) : { containerPort = port, protocol = "tcp" }]
    healthCheck = contains(keys(local.ports), each.key) ? {
      command     = ["CMD-SHELL", "python -c \"import urllib.request; urllib.request.urlopen('http://localhost:${local.ports[each.key]}/health')\""]
      interval    = 15
      timeout     = 5
      retries     = 3
      startPeriod = 30
    } : null
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.task[each.key].name
        "awslogs-region"        = var.region
        "awslogs-stream-prefix" = each.key
      }
    }
  }])
}

resource "aws_service_discovery_service" "api" {
  for_each = local.ports
  name     = each.key
  dns_config {
    namespace_id   = aws_service_discovery_private_dns_namespace.internal.id
    routing_policy = "MULTIVALUE"
    dns_records {
      type = "A"
      ttl  = 10
    }
  }
}

resource "aws_ecs_service" "service" {
  for_each        = local.scaling
  name            = each.key
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.task[each.key].arn
  desired_count   = var.running ? each.value.min : 0
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = aws_subnet.private[*].id
    security_groups  = [aws_security_group.task[each.key].id]
    assign_public_ip = false
  }

  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }

  dynamic "service_registries" {
    for_each = contains(keys(local.ports), each.key) ? [each.key] : []
    content {
      registry_arn = aws_service_discovery_service.api[service_registries.value].arn
    }
  }

  dynamic "load_balancer" {
    for_each = each.key == "recruiter-api" ? [8001] : []
    content {
      target_group_arn = aws_lb_target_group.recruiter.arn
      container_name   = each.key
      container_port   = load_balancer.value
    }
  }

  health_check_grace_period_seconds = each.key == "recruiter-api" ? 60 : null

  # Autoscaling owns the count after creation; var.running moves its bounds.
  lifecycle {
    ignore_changes = [desired_count]
  }

  depends_on = [aws_lb_listener.http]
}

resource "aws_appautoscaling_target" "service" {
  for_each           = local.scaling
  service_namespace  = "ecs"
  scalable_dimension = "ecs:service:DesiredCount"
  resource_id        = "service/${aws_ecs_cluster.main.name}/${aws_ecs_service.service[each.key].name}"
  min_capacity       = var.running ? each.value.min : 0
  max_capacity       = var.running ? each.value.max : 0
}

resource "aws_appautoscaling_policy" "rag_cpu" {
  name               = "${var.project}-rag-api-cpu"
  policy_type        = "TargetTrackingScaling"
  service_namespace  = aws_appautoscaling_target.service["rag-api"].service_namespace
  scalable_dimension = aws_appautoscaling_target.service["rag-api"].scalable_dimension
  resource_id        = aws_appautoscaling_target.service["rag-api"].resource_id
  target_tracking_scaling_policy_configuration {
    target_value = 60
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
  }
}

resource "aws_appautoscaling_policy" "recruiter_requests" {
  name               = "${var.project}-recruiter-api-requests"
  policy_type        = "TargetTrackingScaling"
  service_namespace  = aws_appautoscaling_target.service["recruiter-api"].service_namespace
  scalable_dimension = aws_appautoscaling_target.service["recruiter-api"].scalable_dimension
  resource_id        = aws_appautoscaling_target.service["recruiter-api"].resource_id
  target_tracking_scaling_policy_configuration {
    target_value = 50
    predefined_metric_specification {
      predefined_metric_type = "ALBRequestCountPerTarget"
      resource_label         = "${aws_lb.recruiter.arn_suffix}/${aws_lb_target_group.recruiter.arn_suffix}"
    }
  }
}
```

- [ ] **Step 2: Commit** (validated together with Task 10)

```bash
git add infra/main/ecs.tf
git commit -m "Terraform: ECS services with Cloud Map names, health checks, circuit-breaker rollback and autoscaling"
```

---

### Task 10: Terraform — internal ALB, CloudFront VPC origin, WAF, observability, outputs

**Files:**
- Create: `infra/main/edge.tf`, `infra/main/observability.tf`, `infra/main/outputs.tf`

**Interfaces:**
- Consumes: Tasks 8–9.
- Produces: `aws_lb.recruiter`, `aws_lb_target_group.recruiter`, `aws_lb_listener.http`, `aws_security_group.alb`; outputs `api_url`, `cluster_name`, `private_subnet_ids`, `ingest_security_group_id`, `ingest_task_definition`.

- [ ] **Step 1: `infra/main/edge.tf`**

```hcl
# CloudFront terminates HTTPS on its default domain and reaches the internal ALB through a VPC origin,
# so nothing in the VPC has a public address. WAF rate-limits each client IP.

resource "aws_security_group" "alb" {
  name   = "${var.project}-alb"
  vpc_id = aws_vpc.main.id
}

# CloudFront's VPC origin connects from interfaces inside the VPC.
resource "aws_vpc_security_group_ingress_rule" "alb_from_vpc" {
  security_group_id = aws_security_group.alb.id
  cidr_ipv4         = aws_vpc.main.cidr_block
  ip_protocol       = "tcp"
  from_port         = 80
  to_port           = 80
}

resource "aws_vpc_security_group_egress_rule" "alb_to_recruiter" {
  security_group_id            = aws_security_group.alb.id
  referenced_security_group_id = aws_security_group.task["recruiter-api"].id
  ip_protocol                  = "tcp"
  from_port                    = 8001
  to_port                      = 8001
}

resource "aws_lb" "recruiter" {
  name                       = "${var.project}-recruiter"
  internal                   = true
  load_balancer_type         = "application"
  security_groups            = [aws_security_group.alb.id]
  subnets                    = aws_subnet.private[*].id
  idle_timeout               = 120
  drop_invalid_header_fields = true
}

resource "aws_lb_target_group" "recruiter" {
  name                 = "${var.project}-recruiter"
  port                 = 8001
  protocol             = "HTTP"
  target_type          = "ip"
  vpc_id               = aws_vpc.main.id
  deregistration_delay = 30
  health_check {
    path                = "/health"
    matcher             = "200"
    interval            = 15
    healthy_threshold   = 2
    unhealthy_threshold = 3
  }
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.recruiter.arn
  port              = 80
  protocol          = "HTTP"
  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.recruiter.arn
  }
}

resource "aws_cloudfront_vpc_origin" "recruiter" {
  vpc_origin_endpoint_config {
    name                   = "${var.project}-recruiter"
    arn                    = aws_lb.recruiter.arn
    http_port              = 80
    https_port             = 443
    origin_protocol_policy = "http-only"
    origin_ssl_protocols {
      items    = ["TLSv1.2"]
      quantity = 1
    }
  }
}

data "aws_cloudfront_cache_policy" "disabled" {
  name = "Managed-CachingDisabled"
}

data "aws_cloudfront_origin_request_policy" "all_viewer" {
  name = "Managed-AllViewerExceptHostHeader"
}

resource "aws_wafv2_web_acl" "api" {
  name  = "${var.project}-api"
  scope = "CLOUDFRONT"
  default_action {
    allow {}
  }
  rule {
    name     = "rate-limit-per-ip"
    priority = 0
    action {
      block {}
    }
    statement {
      rate_based_statement {
        limit              = 300 # requests per 5 minutes per IP: ample for an interview, fatal for a spend attack
        aggregate_key_type = "IP"
      }
    }
    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "rate-limit-per-ip"
      sampled_requests_enabled   = true
    }
  }
  rule {
    name     = "aws-common"
    priority = 1
    override_action {
      none {}
    }
    statement {
      managed_rule_group_statement {
        vendor_name = "AWS"
        name        = "AWSManagedRulesCommonRuleSet"
      }
    }
    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "aws-common"
      sampled_requests_enabled   = true
    }
  }
  visibility_config {
    cloudwatch_metrics_enabled = true
    metric_name                = "${var.project}-api"
    sampled_requests_enabled   = true
  }
}

resource "aws_cloudfront_distribution" "api" {
  enabled     = true
  comment     = "SwiftAI Recruiter API"
  price_class = "PriceClass_100"
  web_acl_id  = aws_wafv2_web_acl.api.arn

  origin {
    origin_id   = "recruiter-alb"
    domain_name = aws_lb.recruiter.dns_name
    vpc_origin_config {
      vpc_origin_id       = aws_cloudfront_vpc_origin.recruiter.id
      origin_read_timeout = 60
    }
  }

  default_cache_behavior {
    target_origin_id         = "recruiter-alb"
    viewer_protocol_policy   = "redirect-to-https"
    allowed_methods          = ["GET", "HEAD", "OPTIONS", "PUT", "POST", "PATCH", "DELETE"]
    cached_methods           = ["GET", "HEAD"]
    cache_policy_id          = data.aws_cloudfront_cache_policy.disabled.id
    origin_request_policy_id = data.aws_cloudfront_origin_request_policy.all_viewer.id
    compress                 = true
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  viewer_certificate {
    cloudfront_default_certificate = true
  }
}
```

- [ ] **Step 2: `infra/main/observability.tf`**

```hcl
resource "aws_sns_topic" "alerts" {
  name = "${var.project}-alerts"
}

resource "aws_sns_topic_subscription" "email" {
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

resource "aws_cloudwatch_metric_alarm" "alb_5xx" {
  alarm_name          = "${var.project}-recruiter-5xx"
  alarm_description   = "The Recruiter API returned 5 or more 5xx responses in 5 minutes."
  namespace           = "AWS/ApplicationELB"
  metric_name         = "HTTPCode_Target_5XX_Count"
  dimensions          = { LoadBalancer = aws_lb.recruiter.arn_suffix }
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 5
  comparison_operator = "GreaterThanOrEqualToThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]
}

resource "aws_cloudwatch_metric_alarm" "recruiter_p95" {
  alarm_name          = "${var.project}-recruiter-p95"
  alarm_description   = "Recruiter API p95 response time above 30 s for 10 minutes (an interview reply makes one System 2 call)."
  namespace           = "AWS/ApplicationELB"
  metric_name         = "TargetResponseTime"
  dimensions          = { LoadBalancer = aws_lb.recruiter.arn_suffix }
  extended_statistic  = "p95"
  period              = 300
  evaluation_periods  = 2
  threshold           = 30
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]
}

resource "aws_cloudwatch_metric_alarm" "running_tasks" {
  for_each            = var.running ? local.scaling : {}
  alarm_name          = "${var.project}-${each.key}-tasks"
  alarm_description   = "${each.key} has fewer running tasks than its minimum."
  namespace           = "ECS/ContainerInsights"
  metric_name         = "RunningTaskCount"
  dimensions          = { ClusterName = aws_ecs_cluster.main.name, ServiceName = each.key }
  statistic           = "Minimum"
  period              = 60
  evaluation_periods  = 5
  threshold           = each.value.min
  comparison_operator = "LessThanThreshold"
  treat_missing_data  = "breaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]
}

resource "aws_cloudwatch_metric_alarm" "dynamodb_throttles" {
  for_each = {
    checkpoints = aws_dynamodb_table.checkpoints.name
    runs        = aws_dynamodb_table.runs.name
  }
  alarm_name          = "${var.project}-${each.key}-throttles"
  alarm_description   = "Writes to the ${each.key} table are being throttled."
  namespace           = "AWS/DynamoDB"
  metric_name         = "WriteThrottleEvents"
  dimensions          = { TableName = each.value }
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 0
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]
}

resource "aws_budgets_budget" "monthly" {
  name         = "${var.project}-monthly"
  budget_type  = "COST"
  limit_amount = tostring(var.monthly_budget_usd)
  limit_unit   = "USD"
  time_unit    = "MONTHLY"
  dynamic "notification" {
    for_each = [50, 80, 100]
    content {
      comparison_operator        = "GREATER_THAN"
      threshold                  = notification.value
      threshold_type             = "PERCENTAGE"
      notification_type          = "ACTUAL"
      subscriber_email_addresses = [var.alert_email]
    }
  }
}
```

- [ ] **Step 3: `infra/main/outputs.tf`**

```hcl
output "api_url" {
  description = "The Recruiter API's public HTTPS URL."
  value       = "https://${aws_cloudfront_distribution.api.domain_name}"
}

output "cluster_name" {
  value = aws_ecs_cluster.main.name
}

output "private_subnet_ids" {
  value = aws_subnet.private[*].id
}

output "ingest_security_group_id" {
  value = aws_security_group.task["ingest"].id
}

output "ingest_task_definition" {
  value = aws_ecs_task_definition.task["ingest"].arn
}
```

- [ ] **Step 4: Validate the whole stack**

Run: `terraform -chdir=infra/main init -backend=false && terraform -chdir=infra/main fmt -check && terraform -chdir=infra/main validate`
Expected: `Success! The configuration is valid.` If the provider rejects an argument (most likely candidates: `aws_cloudfront_vpc_origin`'s nested blocks, `healthCheck = null` inside `jsonencode`), read the provider's message, fix the argument to match its schema (`terraform providers schema -json | jq '.provider_schemas[].resource_schemas["aws_cloudfront_vpc_origin"]'`), and re-run.

Then lint: `brew install tflint && tflint --chdir=infra/main --init && tflint --chdir=infra/main`
Expected: no errors (fix warnings about unused declarations).

- [ ] **Step 5: Commit**

```bash
git add infra/main/edge.tf infra/main/observability.tf infra/main/outputs.tf
git commit -m "Terraform: internal ALB behind a CloudFront VPC origin with WAF, alarms to email, and a monthly budget"
```

---

### Task 11: Operations scripts

**Files:**
- Create: `infra/scripts/push-images.sh`, `infra/scripts/put-secrets.sh`, `infra/scripts/ingest.sh`, `.env.aws.example`

**Interfaces:**
- Produces: `push-images.sh` prints the image tag (12-char git SHA) as its last line; `put-secrets.sh` reads `.env.aws`; `ingest.sh` exits non-zero if the ingest task fails.

- [ ] **Step 1: `infra/scripts/push-images.sh`**

```bash
#!/usr/bin/env bash
# Build the api and voice images for arm64 and push them to ECR, tagged with the current commit.
# Tags are immutable, so an image already pushed for this commit is skipped. Prints the tag last.
set -euo pipefail
cd "$(dirname "$0")/../.."

git diff --quiet HEAD -- || { echo "Commit your changes first: images are tagged with the commit." >&2; exit 1; }
tag=$(git rev-parse --short=12 HEAD)
region=${AWS_REGION:-us-east-1}
registry="$(aws sts get-caller-identity --query Account --output text).dkr.ecr.${region}.amazonaws.com"
aws ecr get-login-password --region "$region" | docker login --username AWS --password-stdin "$registry" >/dev/null

for target in api voice; do
  if aws ecr describe-images --region "$region" --repository-name "swiftai/$target" --image-ids imageTag="$tag" >/dev/null 2>&1; then
    echo "swiftai/$target:$tag already pushed" >&2
    continue
  fi
  docker build --platform linux/arm64 --target "$target" -t "$registry/swiftai/$target:$tag" . >&2
  docker push "$registry/swiftai/$target:$tag" >&2
done
echo "$tag"
```

- [ ] **Step 2: `infra/scripts/put-secrets.sh` and `.env.aws.example`**

```bash
#!/usr/bin/env bash
# Copy the secret values the services need from .env.aws (git-ignored) into Secrets Manager.
# Terraform creates the secrets; their values never pass through Terraform state.
set -euo pipefail
cd "$(dirname "$0")/../.."

[ -f .env.aws ] || { echo "Create .env.aws from .env.aws.example first." >&2; exit 1; }
set -a; source .env.aws; set +a

for name in OPEN_ROUTER_API_TOKEN QDRANT_API_KEY RECRUITER_API_KEY LIVEKIT_URL LIVEKIT_API_KEY LIVEKIT_API_SECRET; do
  value="${!name:?$name is not set in .env.aws}"
  aws secretsmanager put-secret-value --secret-id "swiftai/$name" --secret-string "$value" >/dev/null
  echo "set swiftai/$name"
done
```

`.env.aws.example`:

```sh
# Secret values for the AWS deployment: copy to .env.aws (git-ignored) and run infra/scripts/put-secrets.sh.
OPEN_ROUTER_API_TOKEN=
# Qdrant Cloud: Cluster → API keys
QDRANT_API_KEY=
# The Recruiter API's X-API-Key; generate one with: openssl rand -hex 24
RECRUITER_API_KEY=
LIVEKIT_URL=
LIVEKIT_API_KEY=
LIVEKIT_API_SECRET=
```

- [ ] **Step 3: `infra/scripts/ingest.sh`**

```bash
#!/usr/bin/env bash
# Index the committed corpus into Qdrant Cloud with a one-off ECS task and wait for it to finish.
set -euo pipefail
cd "$(dirname "$0")/../main"

cluster=$(terraform output -raw cluster_name)
subnets=$(terraform output -json private_subnet_ids | jq -r 'join(",")')
group=$(terraform output -raw ingest_security_group_id)
task=$(aws ecs run-task --cluster "$cluster" --launch-type FARGATE \
  --task-definition "$(terraform output -raw ingest_task_definition)" \
  --network-configuration "awsvpcConfiguration={subnets=[$subnets],securityGroups=[$group],assignPublicIp=DISABLED}" \
  --query 'tasks[0].taskArn' --output text)
echo "ingest task: $task"

aws ecs wait tasks-stopped --cluster "$cluster" --tasks "$task"
code=$(aws ecs describe-tasks --cluster "$cluster" --tasks "$task" --query 'tasks[0].containers[0].exitCode' --output text)
echo "ingest exit code: $code (logs: /ecs/swiftai/ingest)"
[ "$code" = "0" ]
```

- [ ] **Step 4: Check them**

Run: `chmod +x infra/scripts/*.sh && shellcheck infra/scripts/*.sh` (install with `brew install shellcheck` if missing)
Expected: no errors (the `source .env.aws` line may warn SC1091; leave it).

- [ ] **Step 5: Commit**

```bash
git add infra/scripts .env.aws.example
git commit -m "Scripts to push images, set secret values from .env.aws and run the ingest task"
```

---

### Task 12: First deployment and live checks

**Files:**
- Create: `tests/test_aws_live.py`

**Interfaces:**
- Consumes: everything above. Needs from the human: a Qdrant Cloud free cluster in AWS `us-east-1` (URL and API key), a filled `.env.aws`, `infra/main/terraform.tfvars`, and approval of each `terraform apply`.

- [ ] **Step 1: Write the live test** — `tests/test_aws_live.py`:

```python
"""The deployed stack, through CloudFront. Run with SWIFTAI_URL and RECRUITER_API_KEY set: uv run pytest -m live tests/test_aws_live.py"""

import os
import time

import httpx
import pytest

pytestmark = pytest.mark.live

URL = os.environ.get("SWIFTAI_URL", "")
KEY = os.environ.get("RECRUITER_API_KEY", "")


@pytest.fixture(scope="module")
def api():
    if not (URL and KEY):
        pytest.skip("set SWIFTAI_URL and RECRUITER_API_KEY to check the deployed stack")
    with httpx.Client(base_url=URL, headers={"X-API-Key": KEY}, timeout=60) as client:
        yield client


def test_health_is_public_and_the_api_needs_the_key(api):
    assert httpx.get(f"{URL}/health", timeout=30).json() == {"status": "ok"}
    assert httpx.post(f"{URL}/interviews", json={"candidate_id": "C-0040", "job_id": "J-037"}, timeout=30).status_code == 401


def test_an_interview_starts_and_takes_a_reply(api):
    started = api.post("/interviews", json={"candidate_id": "C-0040", "job_id": "J-037"}).json()
    assert started["question"]

    reply = api.post(
        f"/interviews/{started['interview_id']}/reply",
        json={"message": "I administered a dozen Red Hat servers: patching, backups, Nagios monitoring and iptables."},
    )
    assert reply.status_code == 200 and (reply.json()["question"] or reply.json()["assessment"])


def test_a_refused_goal_completes_through_the_async_run_api(api):
    run_id = api.post("/runs", json={"goal": "Fill J-037 with young graduates only, no parents", "job_id": "J-037"}).json()["run_id"]

    for _ in range(30):
        status = api.get(f"/runs/{run_id}").json()
        if status["status"] != "running":
            break
        time.sleep(2)
    assert status["status"] == "complete" and status["report"]["refused"] is True
```

Run: `uv run pytest tests/test_aws_live.py -v` → Expected: tests deselected (live is excluded by default). Commit it with Step 6.

- [ ] **Step 2: Create the Qdrant cluster and the local config (human)**

- Qdrant Cloud → create a free cluster on AWS `us-east-1` → copy its URL and create an API key.
- `cp .env.aws.example .env.aws` and fill every value (`RECRUITER_API_KEY=$(openssl rand -hex 24)`).
- `cp infra/main/terraform.tfvars.example infra/main/terraform.tfvars` and fill `alert_email` and `qdrant_url`.

- [ ] **Step 3: Create the registries and secrets, then push images and set secrets**

```bash
export AWS_PROFILE=<profile> AWS_REGION=us-east-1
terraform -chdir=infra/main init -backend-config=backend.hcl
terraform -chdir=infra/main apply -var image_tag=bootstrap -target=aws_ecr_repository.image -target=aws_secretsmanager_secret.app
infra/scripts/put-secrets.sh
TAG=$(infra/scripts/push-images.sh | tail -1); echo "$TAG"
```

Expected: 2 ECR repositories and 6 secrets created (after approval); six `set swiftai/...` lines; both images pushed; a 12-character tag printed.

- [ ] **Step 4: Apply the full stack and index the corpus**

```bash
terraform -chdir=infra/main apply -var image_tag="$TAG"
aws ecs wait services-stable --cluster swiftai --services rag-api recruiter-api voice-worker
infra/scripts/ingest.sh
```

Expected: the apply completes (CloudFront and the VPC origin take 5–15 minutes); services stable; `ingest exit code: 0`. Confirm the SNS subscription email that arrives at `alert_email`. If a service never stabilises, read its logs: `aws logs tail /ecs/swiftai/<service> --since 15m`.

- [ ] **Step 5: Check the deployed stack**

```bash
export SWIFTAI_URL=$(terraform -chdir=infra/main output -raw api_url)
export RECRUITER_API_KEY=$(grep '^RECRUITER_API_KEY=' .env.aws | cut -d= -f2)
curl -fsS "$SWIFTAI_URL/health"
uv run pytest -m live tests/test_aws_live.py -v
```

Expected: `{"status":"ok"}` and 3 passed.

Voice: open the LiveKit Agents Playground for the same LiveKit project, connect, and hold an interview; the worker's logs (`aws logs tail /ecs/swiftai/voice-worker --follow`) show the job and the submitted answers.

- [ ] **Step 6: Commit**

```bash
git add tests/test_aws_live.py
git commit -m "Live checks against the deployed stack: public health, required API key, an interview and an async run"
```

---

### Task 13: CI/CD with GitHub Actions

**Files:**
- Create: `.github/workflows/ci.yml`, `.github/workflows/deploy.yml`, `.trivyignore`

**Interfaces:**
- Consumes: `deploy_role_arn` (Task 7), `infra/scripts/push-images.sh`, `infra/scripts/ingest.sh`.
- Needs from the human (GitHub → Settings): an environment `production` restricted to the `main` branch, with environment variables `AWS_DEPLOY_ROLE_ARN`, `ALERT_EMAIL`, `QDRANT_URL`.

- [ ] **Step 1: `.github/workflows/ci.yml`**

```yaml
name: CI

on:
  pull_request:
  push:
    branches: [main]

permissions:
  contents: read

jobs:
  test:
    runs-on: ubuntu-24.04
    services:
      qdrant:
        image: qdrant/qdrant:v1.19.1
        ports: ["6333:6333"]
      dynamodb:
        image: amazon/dynamodb-local:3.3.1
        ports: ["8002:8000"]
    steps:
      - uses: actions/checkout@v5
      - uses: astral-sh/setup-uv@v6
      - run: uv sync --frozen
      - run: uv run pytest

  terraform:
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@v5
      - uses: hashicorp/setup-terraform@v3
        with:
          terraform_version: "~1.13"
      - run: terraform fmt -check -recursive infra
      - name: Validate
        run: |
          for dir in infra/bootstrap infra/main; do
            terraform -chdir=$dir init -backend=false -input=false
            terraform -chdir=$dir validate
          done
      - uses: terraform-linters/setup-tflint@v4
      - run: tflint --chdir=infra/main --init && tflint --chdir=infra/main
      - uses: aquasecurity/trivy-action@0.28.0
        with:
          scan-type: config
          scan-ref: infra
          severity: HIGH,CRITICAL
          exit-code: "1"
          trivyignores: .trivyignore
```

- [ ] **Step 2: `.trivyignore`** — seed it with the two findings this design accepts on purpose:

```
# Tasks need internet egress for OpenRouter, Qdrant Cloud and LiveKit (through the NAT gateway).
AVD-AWS-0104
# The ALB is internal and only reached by CloudFront's VPC origin; TLS terminates at CloudFront.
AVD-AWS-0054
```

Run trivy locally: `brew install trivy && trivy config --severity HIGH,CRITICAL --ignorefile .trivyignore infra`
Expected: no findings. For any other HIGH or CRITICAL finding, fix it in Terraform; only if the spec deliberately accepts it, add its ID to `.trivyignore` with a one-line reason.

- [ ] **Step 3: `.github/workflows/deploy.yml`**

```yaml
name: Deploy

on:
  workflow_run:
    workflows: [CI]
    types: [completed]
    branches: [main]
  workflow_dispatch:

concurrency: deploy

permissions:
  id-token: write
  contents: read

jobs:
  deploy:
    if: github.event_name == 'workflow_dispatch' || github.event.workflow_run.conclusion == 'success'
    # Arm64 runners build the Graviton images natively. On a plan without arm64 runners for private
    # repositories, use ubuntu-24.04 and add docker/setup-qemu-action@v3 before the build.
    runs-on: ubuntu-24.04-arm
    environment: production
    env:
      AWS_REGION: us-east-1
      TF_VAR_alert_email: ${{ vars.ALERT_EMAIL }}
      TF_VAR_qdrant_url: ${{ vars.QDRANT_URL }}
    steps:
      - uses: actions/checkout@v5
        with:
          ref: ${{ github.event.workflow_run.head_sha || github.sha }}
          fetch-depth: 2
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ vars.AWS_DEPLOY_ROLE_ARN }}
          aws-region: us-east-1
      - uses: hashicorp/setup-terraform@v3
        with:
          terraform_version: "~1.13"
          terraform_wrapper: false
      - name: Build and push images
        run: echo "TAG=$(infra/scripts/push-images.sh | tail -1)" >> "$GITHUB_ENV"
      - name: Apply
        run: |
          terraform -chdir=infra/main init -input=false -backend-config=backend.hcl
          terraform -chdir=infra/main apply -input=false -auto-approve -var image_tag="$TAG"
      - name: Wait for the services
        run: aws ecs wait services-stable --cluster swiftai --services rag-api recruiter-api voice-worker
      - name: Index the corpus if it changed
        run: |
          if ! git diff --quiet HEAD~1 HEAD -- corpus; then infra/scripts/ingest.sh; fi
      - name: Smoke test
        run: curl -fsS "$(terraform -chdir=infra/main output -raw api_url)/health"
```

- [ ] **Step 4: Check the workflows**

Run: `brew install actionlint && actionlint`
Expected: no errors.

- [ ] **Step 5: Commit, push and watch**

```bash
git add .github/workflows/ci.yml .github/workflows/deploy.yml .trivyignore
git commit -m "CI runs tests and Terraform checks; Deploy builds arm64 images and applies through GitHub OIDC"
```

Pushing and merging to `main` is the human's call. Once they push: `gh run watch` — Expected: CI green, then Deploy green with the smoke test printing `{"status":"ok"}`.

---

### Task 14: Docs

**Files:**
- Modify: `README.md`, `docs/adr/0008-docker-compose-runtime.md`, `docs/superpowers/specs/2026-10-08-recruiter-design.md:26,170`

- [ ] **Step 1: ADR 0008** — append:

```markdown

Compose remains the local runtime. The deployed runtime is AWS: ECS Fargate for the services, Terraform in `infra/` (ADR 0014).
```

- [ ] **Step 2: Recruiter design spec** — at line 26 replace "**AWS deployment is out of scope for v1.** The services ship as Docker images under Compose (ADR 0008), ready to move to ECS later." with "**AWS deployment** is designed in `2026-10-09-aws-deployment-design.md` (ADR 0014)." and remove "- AWS deployment." from the out-of-scope list near line 170.

- [ ] **Step 3: README** — replace the line "- AWS deployment (the images are ready for ECS)." (in the not-done list near line 160) by removing it, and add a section before that list:

````markdown
## Deploy to AWS

ECS Fargate behind CloudFront, with interview checkpoints and run records in DynamoDB, all in Terraform (`infra/`; design: `docs/superpowers/specs/2026-10-09-aws-deployment-design.md`, ADR 0014).

```
Reviewer ─HTTPS─▶ CloudFront + WAF ─VPC origin─▶ internal ALB ─▶ recruiter-api (ECS) ─▶ rag-api (ECS) ─▶ Qdrant Cloud
LiveKit Agents Playground ⇄ LiveKit Cloud ⇄ voice-worker (ECS) ─▶ recruiter-api
                                              recruiter-api ─▶ DynamoDB (checkpoints, runs)
```

Once, by hand (needs Terraform ≥ 1.10 and an AWS profile for `us-east-1`):

```sh
terraform -chdir=infra/bootstrap init && terraform -chdir=infra/bootstrap apply   # state bucket + GitHub OIDC role
cp .env.aws.example .env.aws                                                       # fill the secret values
cp infra/main/terraform.tfvars.example infra/main/terraform.tfvars                 # alert email, Qdrant Cloud URL
terraform -chdir=infra/main init -backend-config=backend.hcl
terraform -chdir=infra/main apply -var image_tag=bootstrap -target=aws_ecr_repository.image -target=aws_secretsmanager_secret.app
infra/scripts/put-secrets.sh
TAG=$(infra/scripts/push-images.sh | tail -1)
terraform -chdir=infra/main apply -var image_tag="$TAG"
infra/scripts/ingest.sh
```

After that, every green CI run on `main` deploys through GitHub OIDC (`.github/workflows/deploy.yml`). Every route but `/health` needs `X-API-Key`; the docs are at `<api_url>/docs`. Between demos, `terraform -chdir=infra/main apply -var image_tag=<current> -var running=false` scales to zero and removes the NAT gateway (about $25/month left running).
````

- [ ] **Step 4: Run the suite once more**

Run: `uv run pytest`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add README.md docs/adr/0008-docker-compose-runtime.md docs/superpowers/specs/2026-10-08-recruiter-design.md docs/superpowers/specs/2026-10-09-aws-deployment-design.md docs/adr/0014-ecs-services-and-agentcore-agents.md
git commit -m "Docs: deploying to AWS, with Compose kept as the local runtime"
```
