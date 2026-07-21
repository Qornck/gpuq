"""PyInstaller entry point for the standalone gpuq binary."""

from __future__ import annotations

import sys

from gpuq.cli import main as cli_main
from gpuq.runner import main as runner_main


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--gpuq-internal-runner":
        del sys.argv[1]
        runner_main()
        return
    cli_main()


if __name__ == "__main__":
    main()
