from __future__ import annotations

import shutil
import subprocess
from typing import Sequence


def notify(
    *,
    title: str,
    message: str,
    job_id: int,
    command: Sequence[str] | None,
) -> None:
    replacements = {
        "{title}": title,
        "{message}": message,
        "{job_id}": str(job_id),
    }
    if command:
        arguments: list[str] = []
        for part in command:
            value = str(part)
            for source, target in replacements.items():
                value = value.replace(source, target)
            arguments.append(value)
        try:
            subprocess.Popen(arguments, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError:
            pass
        return
    executable = shutil.which("notify-send")
    if executable:
        try:
            subprocess.Popen(
                [executable, title, message],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError:
            pass
