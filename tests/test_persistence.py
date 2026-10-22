"""Durable state: run records and LangGraph checkpoints, in memory and in DynamoDB Local."""

from dataclasses import replace

import pytest

from agentic_recruiter.recruiter.persistence import (
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
