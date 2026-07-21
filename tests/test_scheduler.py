from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from gpuq.config import Settings
from gpuq.db import QueueDB
from gpuq.models import GPUProcess, GPUState
from gpuq.scheduler import Scheduler

from conftest import submit_job


class FakeBackend:
    def __init__(self, states: list[GPUState]) -> None:
        self.states = states

    def sample(self) -> list[GPUState]:
        return self.states

    def process_memory_mb(self, pid: int) -> int | None:
        return None


def gpu(index: int, *, free: int, used: int = 1000, util: int = 10, processes=()) -> GPUState:
    return GPUState(
        index=index,
        uuid=f"GPU-{index}",
        name="fake",
        total_memory_mb=free + used,
        used_memory_mb=used,
        free_memory_mb=free,
        utilization_percent=util,
        temperature_c=50,
        processes=tuple(processes),
    )


def test_stable_samples_are_required(settings: Settings, db: QueueDB, tmp_path: Path) -> None:
    job = submit_job(db, tmp_path, stable_samples=3)
    scheduler = Scheduler(settings=settings, db=db, backend=FakeBackend([gpu(0, free=8000)]))
    assert scheduler.tick(dry_run=True) is None
    assert scheduler.tick(dry_run=True) is None
    decision = scheduler.tick(dry_run=True)
    assert decision is not None
    assert decision.job_id == job.id
    assert decision.gpu_index == 0


def test_memory_reserve_and_utilization_gate(settings: Settings, db: QueueDB, tmp_path: Path) -> None:
    submit_job(db, tmp_path, stable_samples=1, max_utilization=40)
    backend = FakeBackend([gpu(0, free=4999, util=0), gpu(1, free=9000, util=80)])
    scheduler = Scheduler(settings=settings, db=db, backend=backend)
    assert scheduler.tick(dry_run=True) is None
    backend.states[0] = replace(backend.states[0], free_memory_mb=5000)
    decision = scheduler.tick(dry_run=True)
    assert decision is not None
    assert decision.gpu_index == 0


def test_best_fit_and_allowlist(settings: Settings, db: QueueDB, tmp_path: Path) -> None:
    submit_job(db, tmp_path, stable_samples=1, gpu_allowlist=[1, 2])
    backend = FakeBackend([gpu(0, free=5100), gpu(1, free=9000), gpu(2, free=6000)])
    scheduler = Scheduler(settings=settings, db=db, backend=backend)
    decision = scheduler.tick(dry_run=True)
    assert decision is not None
    assert decision.gpu_index == 2


def test_lowest_utilization_precedes_best_fit(
    settings: Settings, db: QueueDB, tmp_path: Path
) -> None:
    submit_job(db, tmp_path, stable_samples=1)
    backend = FakeBackend(
        [gpu(0, free=6000, util=80), gpu(1, free=9000, util=15)]
    )
    scheduler = Scheduler(settings=settings, db=db, backend=backend)
    decision = scheduler.tick(dry_run=True)
    assert decision is not None
    assert decision.gpu_index == 1


def test_exclusive_requires_no_compute_process(settings: Settings, db: QueueDB, tmp_path: Path) -> None:
    submit_job(db, tmp_path, stable_samples=1, exclusive=True, memory_mb=100)
    busy = gpu(0, free=23000, used=500, util=0, processes=(GPUProcess(7, 100),))
    scheduler = Scheduler(settings=settings, db=db, backend=FakeBackend([busy]))
    assert scheduler.tick(dry_run=True) is None
    backend = scheduler.backend
    backend.states = [replace(busy, processes=())]  # type: ignore[attr-defined]
    assert scheduler.tick(dry_run=True) is not None
