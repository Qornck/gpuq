from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import signal
import subprocess
import time

from .config import load_settings
from .db import QueueDB, utcnow
from .models import JobStatus
from .notify import notify


OOM_MARKERS = (
    "cuda out of memory",
    "cuda error: out of memory",
    "cublas_status_alloc_failed",
    "hip out of memory",
    "outofmemoryerror",
)


def tail_text(path: str | Path, maximum_bytes: int = 131072) -> str:
    target = Path(path)
    if not target.exists():
        return ""
    with target.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - maximum_bytes))
        return handle.read().decode("utf-8", errors="replace")


def is_oom_failure(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in OOM_MARKERS)


def _send(db: QueueDB, job_id: int, title: str, message: str, state_dir: Path) -> None:
    job = db.get(job_id)
    if not job.notify:
        return
    settings = load_settings(state_dir=state_dir)
    notify(
        title=title,
        message=message,
        job_id=job_id,
        command=settings.notify_command,
    )


def execute_job(db: QueueDB, job_id: int, *, state_dir: Path) -> int:
    deadline = time.monotonic() + 10.0
    job = db.get(job_id)
    while job.status == JobStatus.STARTING.value and time.monotonic() < deadline:
        time.sleep(0.05)
        job = db.get(job_id)
    if job.status != JobStatus.RUNNING.value:
        raise RuntimeError(f"job {job_id} was not marked running")
    if job.assigned_gpu is None:
        raise RuntimeError("running job has no assigned GPU")

    log_path = Path(job.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(job.env)
    environment["CUDA_VISIBLE_DEVICES"] = str(job.assigned_gpu)
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write(
            f"\n[gpuq {utcnow()}] job={job.id} attempt={job.attempt} "
            f"physical_gpu={job.assigned_gpu} command={list(job.command)!r}\n"
        )
        try:
            child = subprocess.Popen(
                list(job.command),
                cwd=job.cwd,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                # Keep the actual task tree separate from the management
                # runner.  `gpuq cancel` can then stop the task while leaving
                # this runner alive long enough to persist the terminal state.
                start_new_session=True,
            )
        except Exception as error:
            message = f"launch failed: {type(error).__name__}: {error}"
            db.finish(
                job_id,
                status=JobStatus.FAILED.value,
                exit_code=None,
                error=message,
                failure_kind="launch",
            )
            _send(db, job_id, f"gpuq #{job_id} 启动失败", message, state_dir)
            return 127
        db.set_worker_pid(job_id, child.pid)
        # Close the small race in which cancellation is requested between
        # Popen() and writing worker_pid to SQLite.
        if db.get(job_id).cancel_requested:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        return_code = child.wait()
        log.write(f"[gpuq {utcnow()}] command exited with code {return_code}\n")

    current = db.get(job_id)
    if current.cancel_requested:
        db.finish(
            job_id,
            status=JobStatus.CANCELLED.value,
            exit_code=return_code,
            error="cancelled by user",
            failure_kind="cancelled",
        )
        _send(db, job_id, f"gpuq #{job_id} 已取消", current.name, state_dir)
        return return_code
    if return_code == 0:
        db.finish(job_id, status=JobStatus.COMPLETED.value, exit_code=0)
        _send(db, job_id, f"gpuq #{job_id} 已完成", current.name, state_dir)
        return 0

    log_tail = tail_text(log_path)
    oom = is_oom_failure(log_tail)
    current = db.get(job_id)
    error = f"command exited with code {return_code}"
    if oom and current.retry_on_oom and current.attempt <= current.max_retries:
        learned = max(
            int(math.ceil(current.memory_mb * 1.25)),
            current.peak_memory_mb + 256,
        )
        db.requeue_after_oom(job_id, new_memory_mb=learned, error=error)
        _send(
            db,
            job_id,
            f"gpuq #{job_id} OOM，已重新排队",
            f"显存估计从 {current.memory_mb}MB 调整为 {learned}MB",
            state_dir,
        )
        return return_code
    db.finish(
        job_id,
        status=JobStatus.FAILED.value,
        exit_code=return_code,
        error=error,
        failure_kind="oom" if oom else "command",
    )
    _send(db, job_id, f"gpuq #{job_id} 失败", f"{current.name}: {error}", state_dir)
    return return_code


def main() -> None:
    parser = argparse.ArgumentParser(description="gpuq internal job runner")
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--job-id", required=True, type=int)
    args = parser.parse_args()
    state_dir = Path(args.state_dir).expanduser().resolve()
    db = QueueDB(state_dir / "queue.sqlite3")
    try:
        code = execute_job(db, args.job_id, state_dir=state_dir)
    except Exception as error:
        try:
            db.finish(
                args.job_id,
                status=JobStatus.FAILED.value,
                exit_code=None,
                error=f"runner failed: {type(error).__name__}: {error}",
                failure_kind="runner",
            )
        except Exception:
            pass
        raise
    raise SystemExit(code)


if __name__ == "__main__":
    main()
