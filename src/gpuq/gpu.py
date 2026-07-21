from __future__ import annotations

import csv
from dataclasses import replace
import io
import subprocess
from typing import Protocol

from .models import GPUProcess, GPUState


class GPUBackend(Protocol):
    def sample(self) -> list[GPUState]: ...

    def process_memory_mb(self, pid: int) -> int | None: ...


def _number(value: str, *, default: int = 0) -> int:
    cleaned = value.strip()
    if cleaned in {"", "N/A", "[Not Supported]"}:
        return default
    return int(float(cleaned))


def parse_gpu_csv(text: str) -> list[GPUState]:
    states: list[GPUState] = []
    for row in csv.reader(io.StringIO(text.strip())):
        if not row or len(row) < 8:
            continue
        states.append(
            GPUState(
                index=_number(row[0]),
                uuid=row[1].strip(),
                name=row[2].strip(),
                total_memory_mb=_number(row[3]),
                used_memory_mb=_number(row[4]),
                free_memory_mb=_number(row[5]),
                utilization_percent=_number(row[6]),
                temperature_c=_number(row[7]),
            )
        )
    return states


def parse_process_csv(text: str) -> dict[str, tuple[GPUProcess, ...]]:
    grouped: dict[str, list[GPUProcess]] = {}
    for row in csv.reader(io.StringIO(text.strip())):
        if not row or len(row) < 3 or "No running" in row[0]:
            continue
        uuid = row[0].strip()
        try:
            process = GPUProcess(pid=_number(row[1]), used_memory_mb=_number(row[2]))
        except ValueError:
            continue
        grouped.setdefault(uuid, []).append(process)
    return {uuid: tuple(values) for uuid, values in grouped.items()}


class NvidiaSMIBackend:
    def __init__(self, executable: str = "nvidia-smi", *, timeout_seconds: float = 10.0) -> None:
        self.executable = executable
        self.timeout_seconds = timeout_seconds

    def _run(self, arguments: list[str]) -> str:
        completed = subprocess.run(
            [self.executable, *arguments],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=self.timeout_seconds,
        )
        return completed.stdout

    def sample(self) -> list[GPUState]:
        gpu_text = self._run(
            [
                "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ]
        )
        process_text = self._run(
            [
                "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ]
        )
        processes = parse_process_csv(process_text)
        return [
            replace(state, processes=processes.get(state.uuid, ()))
            for state in parse_gpu_csv(gpu_text)
        ]

    def process_memory_mb(self, pid: int) -> int | None:
        for state in self.sample():
            for process in state.processes:
                if process.pid == pid:
                    return process.used_memory_mb
        return None
