from dataclasses import replace

from gpuq.config import load_settings
from gpuq.recovery import requeue_orphan
from gpuq.scheduler import Scheduler
from conftest import submit_job
from test_scheduler import FakeBackend, mark_running


def test_bounded_recovery_preserves_resources_and_dependencies(settings, db, tmp_path):
    parent = submit_job(db, tmp_path)
    job = submit_job(db, tmp_path, memory_mode='auto', exclusive=True,
                     gpu_allowlist=[0, 1], dependency_id=parent.id)
    scheduler = Scheduler(settings=replace(settings, max_orphan_retries=2), db=db,
                          backend=FakeBackend([]), process_checker=lambda _: False)
    for attempt in range(2):
        mark_running(db, job.id, gpu_index=0, runner_pid=101, worker_pid=102)
        assert scheduler.reconcile_active() == [job.id]
        current = db.get(job.id)
        assert current.status == 'queued'
        assert current.attempt == attempt + 1
        assert current.command == job.command
        assert current.memory_mode == 'auto' and current.exclusive
        assert current.gpu_allowlist == (0, 1)
        assert current.dependency_id == parent.id
        assert current.pid is None and current.worker_pid is None
        assert scheduler.reconcile_active() == []
    mark_running(db, job.id, gpu_index=0, runner_pid=101, worker_pid=102)
    scheduler.reconcile_active()
    assert db.get(job.id).failure_kind == 'orphaned'
    with db.connect() as conn:
        assert conn.execute('SELECT COUNT(*) FROM orphan_recoveries').fetchone()[0] == 2


def test_live_worker_and_terminal_or_cancelled_not_requeued(settings, db, tmp_path):
    job = submit_job(db, tmp_path)
    mark_running(db, job.id, gpu_index=0, runner_pid=101, worker_pid=102)
    scheduler = Scheduler(settings=replace(settings, max_orphan_retries=3), db=db,
                          backend=FakeBackend([]), process_checker=lambda p: p == 102)
    assert scheduler.reconcile_active() == []
    with db.connect() as conn:
        conn.execute('UPDATE jobs SET cancel_requested=1 WHERE id=?', (job.id,))
    assert not requeue_orphan(db, job.id, 3)
    # A completion racing the process check must never be overwritten.
    with db.connect() as conn:
        conn.execute("UPDATE jobs SET status='completed', cancel_requested=0 WHERE id=?", (job.id,))
    assert not requeue_orphan(db, job.id, 3)
    assert db.get(job.id).status == 'completed'


def test_previous_boot_pid_reuse(settings, db, tmp_path):
    job = submit_job(db, tmp_path)
    mark_running(db, job.id, gpu_index=0, runner_pid=101, worker_pid=102)
    with db.connect() as conn:
        conn.execute("UPDATE jobs SET updated_at='2000-01-01T00:00:00+00:00' WHERE id=?", (job.id,))
    scheduler = Scheduler(settings=replace(settings, max_orphan_retries=3), db=db,
                          backend=FakeBackend([]), process_checker=lambda _: True)
    scheduler.boot_time = 1700000000
    assert scheduler.reconcile_active() == [job.id]
    assert db.get(job.id).status == 'queued'


def test_setting(tmp_path):
    path = tmp_path / 'config.toml'
    path.write_text('[gpuq]\nmax_orphan_retries=3\n')
    assert load_settings(state_dir=tmp_path, config_path=path).max_orphan_retries == 3
