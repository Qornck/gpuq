from __future__ import annotations

from pathlib import Path

import pytest

from gpuq.db import QueueDB
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


def test_cancel_queued_and_retry_terminal(db: QueueDB, tmp_path: Path) -> None:
    job = submit_job(db, tmp_path)
    cancelled = db.request_cancel(job.id)
    assert cancelled.status == JobStatus.CANCELLED.value
    retried = db.retry(job.id, memory_mb=6000, reset_attempts=True)
    assert retried.status == JobStatus.QUEUED.value
    assert retried.memory_mb == 6000


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
