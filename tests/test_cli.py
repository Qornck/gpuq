from pathlib import Path
import os

import pytest

from gpuq.cli import (
    _job_page,
    _parse_env,
    _parse_gpus,
    _read_pager_key,
    build_parser,
    main,
)
from gpuq.config import load_settings
from gpuq.db import QueueDB

from conftest import submit_job


def test_resource_argument_aliases_and_command_remainder() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "submit",
            "--memory",
            "4096",
            "--reserve",
            "512",
            "--gpu",
            "1,2",
            "--env",
            "A=B",
            "--",
            "python",
            "train.py",
            "--epochs",
            "3",
        ]
    )
    assert args.memory_mb == 4096
    assert args.reserve_mb == 512
    assert args.command[-2:] == ["--epochs", "3"]
    assert _parse_gpus(args.gpu) == (1, 2)
    assert _parse_env(args.env) == {"A": "B"}


def test_submit_defaults_to_auto_memory() -> None:
    parser = build_parser()
    args = parser.parse_args(["submit", "--", "python", "train.py"])
    assert args.memory_mb == "auto"


def test_version_flag(capsys) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit) as error:
        parser.parse_args(["--version"])
    assert error.value.code == 0
    assert capsys.readouterr().out.strip() == "gpuq 0.2.0"


def test_default_auto_submit_is_persisted(tmp_path: Path) -> None:
    state = tmp_path / "state"
    try:
        main(
            [
                "--state-dir",
                str(state),
                "submit",
                "--no-notify",
                "--",
                "python",
                "train.py",
            ]
        )
    except SystemExit as error:
        assert error.code == 0
    job = QueueDB(state / "queue.sqlite3").list()[0]
    assert job.is_auto_memory
    assert job.memory_mb == 1
    assert job.retry_on_oom


def test_explicit_auto_memory_and_list_pager_arguments() -> None:
    parser = build_parser()
    submit = parser.parse_args(["submit", "--memory", "auto", "--", "python", "x.py"])
    assert submit.memory_mb == "auto"
    listing = parser.parse_args(["list", "--page-size", "20", "--no-pager"])
    assert listing.page_size == 20
    assert listing.no_pager


def test_list_pager_defaults_to_twenty_rows() -> None:
    parser = build_parser()
    listing = parser.parse_args(["list"])
    assert listing.page_size == 20


def test_job_page_shows_twenty_rows_and_navigation_hint(
    db: QueueDB, tmp_path: Path
) -> None:
    jobs = [submit_job(db, tmp_path, name=f"job-{index}") for index in range(22)]
    first = _job_page(jobs, 0, 20)
    second = _job_page(jobs, 1, 20)
    assert "job-0" in first and "job-19" in first
    assert "job-20" not in first
    assert "第 1/2 页" in first
    assert "job-20" in second and "job-21" in second
    assert "←/→ 翻页" in second


def test_pager_reads_arrow_keys_and_quit() -> None:
    reader, writer = os.pipe()
    try:
        os.write(writer, b"\x1b[C")
        assert _read_pager_key(reader) == "right"
        os.write(writer, b"q")
        assert _read_pager_key(reader) == "quit"
    finally:
        os.close(reader)
        os.close(writer)


def test_init_writes_parseable_config_and_allows_zero_reserve(
    tmp_path: Path, monkeypatch
) -> None:
    state = tmp_path / "state"
    config = tmp_path / "config.toml"
    with monkeypatch.context() as scope:
        scope.setattr(
            "sys.argv",
            [
                "gpuq",
                "--state-dir",
                str(state),
                "--config",
                str(config),
                "init",
            ],
        )
        try:
            main()
        except SystemExit as error:
            assert error.code == 0
    text = config.read_text(encoding="utf-8").replace(
        "default_reserve_mb = 1024", "default_reserve_mb = 0"
    )
    config.write_text(text, encoding="utf-8")
    settings = load_settings(config_path=config)
    assert settings.default_reserve_mb == 0
