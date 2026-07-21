from __future__ import annotations

from pathlib import Path
import os
import signal
import sys
import time

from gpuq.db import QueueDB
from gpuq.models import JobStatus
from gpuq.runner import execute_job, is_oom_failure
from gpuq.scheduler import cancel_job

from conftest import submit_job


def prepare_running(db: QueueDB, job_id: int) -> None:
    assert db.claim(job_id, 1)
    db.mark_running(job_id, pid=999001, pgid=999001)


def test_successful_runner_captures_log_and_environment(db: QueueDB, tmp_path: Path) -> None:
    job = submit_job(
        db,
        tmp_path,
        command=[sys.executable, "-c", "import os; print(os.environ['CUDA_VISIBLE_DEVICES'])"],
        env={"EXAMPLE": "yes"},
    )
    prepare_running(db, job.id)
    assert execute_job(db, job.id, state_dir=tmp_path) == 0
    finished = db.get(job.id)
    assert finished.status == JobStatus.COMPLETED.value
    assert "1" in Path(job.log_path).read_text(encoding="utf-8")


def test_oom_requeues_and_increases_estimate(db: QueueDB, tmp_path: Path) -> None:
    job = submit_job(
        db,
        tmp_path,
        command=[sys.executable, "-c", "import sys; print('CUDA out of memory'); sys.exit(1)"],
        memory_mb=4000,
        max_retries=1,
        retry_on_oom=True,
    )
    prepare_running(db, job.id)
    assert execute_job(db, job.id, state_dir=tmp_path) == 1
    retried = db.get(job.id)
    assert retried.status == JobStatus.RETRY_WAIT.value
    assert retried.memory_mb == 5000
    assert retried.failure_kind == "oom"


def test_oom_detection_is_specific() -> None:
    assert is_oom_failure("RuntimeError: CUDA out of memory")
    assert not is_oom_failure("ordinary validation error")


def test_cancel_stops_worker_group_and_persists_terminal_state(
    db: QueueDB, tmp_path: Path
) -> None:
    job = submit_job(
        db,
        tmp_path,
        command=[sys.executable, "-c", "import time; time.sleep(60)"],
    )
    prepare_running(db, job.id)

    # Reproduce the runner's worker-group arrangement without requiring a GPU.
    child_pid = os.fork()
    if child_pid == 0:  # pragma: no cover - child never returns to pytest
        os.setsid()
        time.sleep(60)
        os._exit(0)
    try:
        db.set_worker_pid(job.id, child_pid)
        cancelled = cancel_job(db, job.id, force_after=0.05)
        assert cancelled.status == JobStatus.CANCELLED.value
        assert cancelled.cancel_requested
    finally:
        try:
            os.killpg(child_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            os.waitpid(child_pid, 0)
        except ChildProcessError:
            pass
