from pathlib import Path

from gpuq.cli import _parse_env, _parse_gpus, build_parser, main
from gpuq.config import load_settings


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
