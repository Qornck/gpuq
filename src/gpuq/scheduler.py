from __future__ import annotations

from dataclasses import dataclass
import fcntl
import os
import signal
import subprocess
import sys
import time
from typing import Callable

from .config import Settings, ensure_state_dirs
from .db import QueueDB
from .gpu import GPUBackend, NvidiaSMIBackend
from .models import GPUState, Job, JobStatus, TERMINAL_STATUSES


@dataclass(frozen=True)
class LaunchDecision:
    job_id: int
    gpu_index: int
    stable_count: int
    free_memory_mb: int
    required_memory_mb: int
    outstanding_reservation_mb: int
    utilization_percent: int


@dataclass(frozen=True)
class GPUReservation:
    outstanding_memory_mb: int = 0
    reserve_mb: int = 0
    active_jobs: int = 0


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class Scheduler:
    def __init__(
        self,
        *,
        settings: Settings,
        db: QueueDB | None = None,
        backend: GPUBackend | None = None,
        launcher: Callable[[Job, int], None] | None = None,
        process_checker: Callable[[int], bool] | None = None,
    ) -> None:
        self.settings = settings
        ensure_state_dirs(settings)
        self.db = db or QueueDB(settings.database_path)
        self.backend = backend or NvidiaSMIBackend(settings.nvidia_smi)
        self.launcher = launcher or self._launch
        self.process_checker = process_checker or process_exists
        self.stability: dict[tuple[int, int], int] = {}
        self.runners: dict[int, subprocess.Popen[bytes]] = {}

    def reconcile_active(self) -> list[int]:
        """Fail records whose runner and worker both disappeared.

        A live runner or worker is preserved across daemon restarts.  The
        conditional database update avoids racing a runner that writes its
        terminal state between the process check and this method.
        """
        recovered: list[int] = []
        for job in self.db.active():
            runner_alive = job.pid is not None and self.process_checker(job.pid)
            worker_alive = (
                job.worker_pid is not None and self.process_checker(job.worker_pid)
            )
            if runner_alive or worker_alive:
                continue
            if self.db.fail_if_active(
                job.id,
                error="runner and worker disappeared before recording an exit status",
                failure_kind="orphaned",
            ):
                recovered.append(job.id)
        return recovered

    def _reservations(self, states: list[GPUState]) -> dict[int, GPUReservation]:
        process_memory = {
            (state.index, process.pid): process.used_memory_mb
            for state in states
            for process in state.processes
        }
        reservations: dict[int, GPUReservation] = {}
        for job in self.db.active():
            if job.assigned_gpu is None:
                continue
            current = reservations.get(job.assigned_gpu, GPUReservation())
            observed = (
                process_memory.get((job.assigned_gpu, job.worker_pid), 0)
                if job.worker_pid is not None
                else 0
            )
            reservations[job.assigned_gpu] = GPUReservation(
                outstanding_memory_mb=(
                    current.outstanding_memory_mb
                    + max(0, job.memory_mb - observed)
                ),
                reserve_mb=max(current.reserve_mb, job.reserve_mb),
                active_jobs=current.active_jobs + 1,
            )
        return reservations

    @staticmethod
    def _fit(
        job: Job,
        state: GPUState,
        reservation: GPUReservation,
    ) -> tuple[bool, int, int]:
        schedulable_free = max(
            0, state.free_memory_mb - reservation.outstanding_memory_mb
        )
        required = job.memory_mb + max(job.reserve_mb, reservation.reserve_mb)
        return schedulable_free >= required, schedulable_free - required, required

    def _allowed(
        self,
        job: Job,
        state: GPUState,
        reservation: GPUReservation,
    ) -> bool:
        if job.gpu_allowlist is not None and state.index not in job.gpu_allowlist:
            return False
        fits, _, _ = self._fit(job, state, reservation)
        if not fits:
            return False
        if job.max_utilization is not None and state.utilization_percent > job.max_utilization:
            return False
        if job.exclusive:
            return (
                reservation.active_jobs == 0
                and state.compute_process_count == 0
                and state.used_memory_mb <= self.settings.exclusive_max_used_mb
                and state.utilization_percent <= self.settings.exclusive_max_utilization
            )
        return True

    def decisions(self, jobs: list[Job], states: list[GPUState]) -> list[LaunchDecision]:
        alive_keys: set[tuple[int, int]] = set()
        decisions: list[LaunchDecision] = []
        reservations = self._reservations(states)
        for job in jobs:
            eligible: list[tuple[int, int, GPUState, GPUReservation, int]] = []
            for state in states:
                key = (job.id, state.index)
                alive_keys.add(key)
                reservation = reservations.get(state.index, GPUReservation())
                if self._allowed(job, state, reservation):
                    count = self.stability.get(key, 0) + 1
                    self.stability[key] = count
                    _, free_after, required = self._fit(job, state, reservation)
                    eligible.append(
                        (free_after, state.utilization_percent, state, reservation, required)
                    )
                else:
                    self.stability[key] = 0
            if not eligible:
                continue
            # Memory is a hard feasibility gate.  Among GPUs that can safely
            # fit the job, prefer the least-loaded device; use best-fit only
            # as a tie-breaker so we do not place new work on an already busy
            # card merely because it has slightly less free memory.
            _, _, selected, reservation, required = min(
                eligible, key=lambda row: (row[1], row[0], row[2].index)
            )
            stable = self.stability[(job.id, selected.index)]
            if stable >= job.stable_samples:
                decisions.append(
                    LaunchDecision(
                        job_id=job.id,
                        gpu_index=selected.index,
                        stable_count=stable,
                        free_memory_mb=selected.free_memory_mb,
                        required_memory_mb=required,
                        outstanding_reservation_mb=reservation.outstanding_memory_mb,
                        utilization_percent=selected.utilization_percent,
                    )
                )
        self.stability = {key: value for key, value in self.stability.items() if key in alive_keys}
        return decisions

    def refresh_peaks(self) -> None:
        for job in self.db.active():
            if job.worker_pid is None:
                continue
            observed = self.backend.process_memory_mb(job.worker_pid)
            if observed is not None and observed > job.peak_memory_mb:
                self.db.update_peak(job.id, observed)

    def tick(self, *, dry_run: bool = False) -> LaunchDecision | None:
        self.reap_runners()
        self.reconcile_active()
        states = self.backend.sample()
        self.refresh_peaks()
        jobs = self.db.runnable()
        decisions = self.decisions(jobs, states)
        if not decisions:
            if not dry_run:
                for job in jobs:
                    self.db.set_waiting_memory(job.id)
            return None
        decision = decisions[0]
        if dry_run:
            return decision

        # The unmanaged multi-user environment has an unavoidable race.  Take a
        # second authoritative snapshot immediately before claiming the job.
        second = {state.index: state for state in self.backend.sample()}
        job = self.db.get(decision.job_id)
        state = second.get(decision.gpu_index)
        reservation = self._reservations(list(second.values())).get(
            decision.gpu_index, GPUReservation()
        )
        if state is None or not self._allowed(job, state, reservation):
            self.stability[(job.id, decision.gpu_index)] = 0
            self.db.set_waiting_memory(job.id)
            return None
        if not self.db.claim(job.id, state.index):
            return None
        claimed = self.db.get(job.id)
        try:
            self.launcher(claimed, state.index)
        except Exception as error:
            self.db.finish(
                job.id,
                status=JobStatus.FAILED.value,
                exit_code=None,
                error=f"scheduler launch failed: {type(error).__name__}: {error}",
                failure_kind="scheduler",
            )
            raise
        self.stability = {key: 0 for key in self.stability}
        return decision

    def _launch(self, job: Job, gpu_index: int) -> None:
        environment = os.environ.copy()
        environment["GPUQ_STATE_DIR"] = str(self.settings.state_dir)
        if self.settings.config_path is not None:
            environment["GPUQ_CONFIG"] = str(self.settings.config_path)
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "gpuq.runner",
                "--state-dir",
                str(self.settings.state_dir),
                "--job-id",
                str(job.id),
            ],
            env=environment,
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.runners[job.id] = process
        self.db.mark_running(job.id, pid=process.pid, pgid=process.pid)

    def reap_runners(self) -> None:
        self.runners = {
            job_id: process
            for job_id, process in self.runners.items()
            if process.poll() is None
        }


def cancel_job(db: QueueDB, job_id: int, *, force_after: float = 10.0) -> Job:
    job = db.request_cancel(job_id)
    if job.status == JobStatus.CANCELLED.value or job.pgid is None:
        return db.get(job_id)

    # The actual command is started in its own session, so worker_pid is also
    # its process-group ID.  Preserve the runner whenever possible: it is the
    # component that records the exit code and final cancelled state.
    target_pgid = job.worker_pid or job.pgid
    try:
        os.killpg(target_pgid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + max(0.0, force_after)
    while time.monotonic() < deadline:
        current = db.get(job_id)
        if current.status in TERMINAL_STATUSES:
            return current
        time.sleep(0.1)

    # The runner may have published worker_pid just after the first snapshot.
    current = db.get(job_id)
    final_target = current.worker_pid or target_pgid
    try:
        os.killpg(final_target, signal.SIGKILL)
    except ProcessLookupError:
        pass
    db.finish(
        job_id,
        status=JobStatus.CANCELLED.value,
        exit_code=None,
        error="cancelled by user",
        failure_kind="cancelled",
    )
    return db.get(job_id)


def run_daemon(settings: Settings, *, once: bool = False, dry_run: bool = False) -> None:
    ensure_state_dirs(settings)
    lock_handle = settings.lock_path.open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("another gpuqd instance is already running") from error
    lock_handle.seek(0)
    lock_handle.truncate()
    lock_handle.write(str(os.getpid()))
    lock_handle.flush()
    scheduler = Scheduler(settings=settings)
    stop = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    while not stop:
        try:
            decision = scheduler.tick(dry_run=dry_run)
            if decision is not None:
                action = "would launch" if dry_run else "launched"
                print(
                    f"gpuqd {action} job={decision.job_id} gpu={decision.gpu_index} "
                    f"free={decision.free_memory_mb}MB "
                    f"reserved={decision.outstanding_reservation_mb}MB "
                    f"required={decision.required_memory_mb}MB "
                    f"stable={decision.stable_count}",
                    flush=True,
                )
        except (OSError, subprocess.SubprocessError) as error:
            print(f"gpuqd monitor error: {error}", file=sys.stderr, flush=True)
        if once:
            break
        deadline = time.monotonic() + settings.poll_seconds
        while not stop and time.monotonic() < deadline:
            time.sleep(min(0.25, deadline - time.monotonic()))
