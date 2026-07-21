from __future__ import annotations

from pathlib import Path
import sqlite3

import pytest

from gpuq.db import QueueDB, SCHEMA
from gpuq.models import JobStatus

from conftest import submit_job


def test_submit_round_trip_and_priority(db: QueueDB, tmp_path: Path) -> None:
    job = submit_job(
        db,
        tmp_path,
        name="训练 A",
        env={"A": "1"},
        gpu_allowlist=[1, 2],
        priority=7,
    )
    loaded = db.get(job.id)
    assert loaded.name == "训练 A"
    assert loaded.env == {"A": "1"}
    assert loaded.gpu_allowlist == (1, 2)
    assert loaded.required_free_memory_mb == 5000
    assert db.list()[0].id == job.id
    assert db.set_priority(job.id, 99).priority == 99


def test_auto_memory_fields_round_trip(db: QueueDB, tmp_path: Path) -> None:
    job = submit_job(
        db,
        tmp_path,
        memory_mb=1,
        memory_mode="auto",
        reserve_mb=512,
    )
    assert job.is_auto_memory
    assert job.is_uncalibrated_auto_probe
    assert db.claim(
        job.id,
        1,
        launch_free_memory_mb=12000,
        launch_total_memory_mb=24576,
    )
    claimed = db.get(job.id)
    assert claimed.launch_free_memory_mb == 12000
    assert claimed.launch_total_memory_mb == 24576
    db.requeue_after_oom(
        job.id,
        new_memory_mb=9000,
        error="oom",
        requested_mb=2048,
    )
    learned = db.get(job.id)
    assert learned.oom_count == 1
    assert learned.memory_mb == 9000
    assert learned.last_oom_requested_mb == 2048


def test_successful_auto_probe_retains_observed_peak_as_calibration(
    db: QueueDB, tmp_path: Path
) -> None:
    job = submit_job(db, tmp_path, memory_mb=1, memory_mode="auto")
    assert db.claim(job.id, 0, launch_free_memory_mb=12000)
    db.update_peak(job.id, 5432)
    db.finish(job.id, status=JobStatus.COMPLETED.value, exit_code=0)
    finished = db.get(job.id)
    assert finished.memory_mb == 5432
    assert not finished.is_uncalibrated_auto_probe


def test_existing_database_schema_is_migrated(tmp_path: Path) -> None:
    path = tmp_path / "legacy.sqlite3"
    legacy_schema = SCHEMA
    for line in (
        "    memory_mode TEXT NOT NULL DEFAULT 'fixed' CHECK(memory_mode IN ('fixed', 'auto')),\n",
        "    oom_count INTEGER NOT NULL DEFAULT 0,\n",
        "    launch_free_memory_mb INTEGER,\n",
        "    launch_total_memory_mb INTEGER,\n",
        "    last_oom_requested_mb INTEGER,\n",
    ):
        legacy_schema = legacy_schema.replace(line, "")
    connection = sqlite3.connect(path)
    try:
        connection.executescript(legacy_schema)
    finally:
        connection.close()

    QueueDB(path)
    connection = sqlite3.connect(path)
    try:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(jobs)")}
    finally:
        connection.close()
    assert {
        "memory_mode",
        "oom_count",
        "launch_free_memory_mb",
        "launch_total_memory_mb",
        "last_oom_requested_mb",
    } <= columns


def test_cancel_queued_and_retry_terminal(db: QueueDB, tmp_path: Path) -> None:
    job = submit_job(db, tmp_path)
    cancelled = db.request_cancel(job.id)
    assert cancelled.status == JobStatus.CANCELLED.value
    retried = db.retry(job.id, memory_mb=6000, reset_attempts=True)
    assert retried.status == JobStatus.QUEUED.value
    assert retried.memory_mb == 6000


def test_terminal_job_can_switch_to_fresh_auto_memory_probe(
    db: QueueDB, tmp_path: Path
) -> None:
    job = submit_job(db, tmp_path, memory_mb=4000)
    db.finish(job.id, status=JobStatus.FAILED.value, exit_code=1)
    retried = db.retry(job.id, memory_mb=1, memory_mode="auto")
    assert retried.status == JobStatus.QUEUED.value
    assert retried.is_auto_memory
    assert retried.memory_mb == 1
    assert retried.retry_on_oom
    assert retried.oom_count == 0


def test_dependency_success_and_failure(db: QueueDB, tmp_path: Path) -> None:
    parent = submit_job(db, tmp_path, name="parent")
    child = submit_job(db, tmp_path, name="child", dependency_id=parent.id)
    assert [job.id for job in db.runnable()] == [parent.id]
    db.finish(parent.id, status=JobStatus.FAILED.value, exit_code=2)
    assert db.runnable() == []
    assert db.get(child.id).status == JobStatus.BLOCKED_DEPENDENCY.value


def test_invalid_dependency_is_rejected(db: QueueDB, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        submit_job(db, tmp_path, dependency_id=999)


def test_fail_if_active_does_not_overwrite_terminal_result(
    db: QueueDB, tmp_path: Path
) -> None:
    job = submit_job(db, tmp_path)
    assert db.claim(job.id, 0)
    db.mark_running(job.id, pid=101, pgid=101)
    assert db.fail_if_active(
        job.id, error="runner disappeared", failure_kind="orphaned"
    )
    failed = db.get(job.id)
    assert failed.status == JobStatus.FAILED.value
    assert failed.failure_kind == "orphaned"
    assert failed.pid is None

    retried = db.retry(job.id)
    assert db.claim(retried.id, 0)
    db.mark_running(retried.id, pid=102, pgid=102)
    db.finish(retried.id, status=JobStatus.COMPLETED.value, exit_code=0)
    assert not db.fail_if_active(
        retried.id, error="late stale check", failure_kind="orphaned"
    )
    assert db.get(retried.id).status == JobStatus.COMPLETED.value
