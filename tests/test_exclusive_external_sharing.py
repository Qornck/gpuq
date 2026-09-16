from dataclasses import replace

from gpuq.config import load_settings
from gpuq.models import GPUProcess
from gpuq.scheduler import Scheduler
from conftest import submit_job
from test_scheduler import FakeBackend, gpu, mark_running


def test_only_opted_in_device_can_share_external_compute(settings, db, tmp_path):
    job = submit_job(db, tmp_path, exclusive=True, stable_samples=3)
    devices = [gpu(i, free=12000, used=12000, util=100,
                   processes=(GPUProcess(900 + i, 12000),)) for i in (0, 1)]
    scheduler = Scheduler(settings=replace(settings, exclusive_shared_gpus=(1,)),
                          db=db, backend=FakeBackend(devices))
    assert scheduler.tick(dry_run=True) is None
    assert scheduler.tick(dry_run=True) is None
    decision = scheduler.tick(dry_run=True)
    assert decision.job_id == job.id and decision.gpu_index == 1


def test_sharing_never_allows_second_queue_task(settings, db, tmp_path):
    first = submit_job(db, tmp_path, exclusive=True, stable_samples=1)
    mark_running(db, first.id, gpu_index=1, runner_pid=101, worker_pid=102)
    submit_job(db, tmp_path, exclusive=True, stable_samples=1)
    scheduler = Scheduler(settings=replace(settings, exclusive_shared_gpus=(1,)), db=db,
                          backend=FakeBackend([gpu(1, free=20000)]), process_checker=lambda pid: True)
    assert scheduler.tick(dry_run=True) is None


def test_sharing_preserves_memory_reserve(settings, db, tmp_path):
    submit_job(db, tmp_path, exclusive=True, stable_samples=1, memory_mb=4000, reserve_mb=1000)
    backend = FakeBackend([gpu(1, free=4999, processes=(GPUProcess(9, 10000),))])
    scheduler = Scheduler(settings=replace(settings, exclusive_shared_gpus=(1,)), db=db, backend=backend)
    assert scheduler.tick(dry_run=True) is None
    backend.states = [replace(backend.states[0], free_memory_mb=5000)]
    assert scheduler.tick(dry_run=True) is not None


def test_config_defaults_strict_and_loads_explicit_devices(tmp_path):
    path = tmp_path / 'policy.toml'
    assert load_settings(state_dir=tmp_path, config_path=path).exclusive_shared_gpus == ()
    path.write_text('[gpuq]\nexclusive_shared_gpus = [1]\n')
    assert load_settings(state_dir=tmp_path, config_path=path).exclusive_shared_gpus == (1,)
