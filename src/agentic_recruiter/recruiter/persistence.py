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
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph_checkpoint_aws import DynamoDBSaver

from agentic_recruiter.config import Config

RETENTION_S = 7 * 24 * 3600  # DynamoDB TTL on checkpoints and run records
STALE_AFTER_S = 20 * 60  # a full run takes about 3 minutes; one still running after 20 lost its task
# Types the interview graph keeps in its state. LangGraph only deserialises checkpointed classes it is told about.
CHECKPOINT_TYPES = [
    ("agentic_recruiter.recruiter.schemas", name) for name in ("InterviewAssessment", "InterviewTurn", "RedFlag", "TurnTiming")
]
SERDE = JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_TYPES)


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
        return InMemorySaver(serde=SERDE)
    return _DynamoDBSaver(
        table_name=config.checkpoint_table,
        session=_session(config),
        endpoint_url=config.dynamodb_endpoint or None,
        ttl_seconds=RETENTION_S,
    )


class _DynamoDBSaver(DynamoDBSaver):
    # DynamoDBSaver takes no `serde` argument; the base class picks up this class attribute instead.
    serde = SERDE


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
