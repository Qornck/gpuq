from gpuq.gpu import parse_gpu_csv, parse_process_csv


def test_parse_gpu_and_process_csv() -> None:
    states = parse_gpu_csv(
        "0, GPU-a, NVIDIA A, 24564, 8367, 16197, 33, 61\n"
        "1, GPU-b, NVIDIA B, 24564, 501, 24063, 0, 45\n"
    )
    assert states[0].index == 0
    assert states[0].free_memory_mb == 16197
    assert states[1].utilization_percent == 0
    processes = parse_process_csv("GPU-a, 123, 8334\nGPU-a, 124, 10\n")
    assert [process.pid for process in processes["GPU-a"]] == [123, 124]
    assert processes["GPU-a"][0].used_memory_mb == 8334


def test_empty_process_output() -> None:
    assert parse_process_csv("") == {}
    assert parse_process_csv("No running processes found") == {}
