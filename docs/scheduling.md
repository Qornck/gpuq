# 调度模型与安全边界

## 1. 普通任务判定

对任务 `j` 和 GPU `g`：

```text
required(j) = estimated_peak_memory(j) + reserve(j)
```

基本可行条件：

```text
free_memory(g) >= required(j)
```

若任务设置最大利用率，还要求：

```text
utilization(g) <= max_utilization(j)
```

条件连续满足 `stable_samples(j)` 次后才成为候选，启动前重新读取一次 GPU 状态。默认三个 10 秒采样，相当于约 30 秒稳定窗口。

## 2. 独占任务

`--exclusive` 额外要求：

- compute process 数为 0；
- 已用显存不超过 `exclusive_max_used_mb`；
- 利用率不超过 `exclusive_max_utilization`。

少量驱动、桌面或监控显存不等于存在 CUDA 训练，因此不要求绝对 0MB。

## 3. 优先级与 best-fit

任务首先按：

```text
priority DESC, job_id ASC
```

处理。一个高优先级任务当前放不下时，不会阻塞能够运行的低优先级任务，这属于 backfill。

当前版本不为暂时放不下的高优先级任务预留未来显存；如果低优先级短任务
持续到达，高显存任务可能等待较久。需要严格防饥饿时，应暂停提交小任务，
限制它们的 GPU allowlist，或临时使用 `--exclusive` 配合空卡窗口。

同一任务可在多张卡运行时，选择：

```text
argmin(free_memory - required_memory, utilization, gpu_index)
```

即优先使用“刚好放得下”的 GPU。

## 4. 为什么不能真正等价于 Slurm

检查时还有 12GB，不代表启动 CUDA context 时仍有 12GB。其他用户可以在检查和分配之间启动任务，且他们不会读取 `gpuq` 的锁。因此只能降低竞态，不能消除竞态。

本工具采用：

1. 用户提供峰值估计；
2. 独立安全余量；
3. 连续稳定采样；
4. 启动前二次检查；
5. OOM 识别和重新排队；
6. 不使用占位 tensor，不侵占尚未真正使用的显存。

多个用户各自运行 `gpuq` 时，各自的 SQLite 和 `flock` 互不相识，仍可能同时启动任务。只有所有用户接入同一个服务，才能获得真正的全局原子分配。

## 5. 失败隔离

每个任务由独立 runner 和进程组承载：

- stdout/stderr 合并到任务日志；
- runner 将退出状态写回 SQLite；
- daemon 不等待单个任务，可以继续调度；
- `cancel` 只向该任务进程组发送信号；
- daemon 或终端重启不会删除队列和日志；
- runner 独立写回结果，因此 daemon 短暂重启时运行任务仍可结束并落库。

## 6. OOM 重试

检测到常见 CUDA/HIP OOM 文本且开启 `--retry-on-oom` 时：

```text
new_estimate = max(ceil(old_estimate * 1.25), observed_peak + 256MB)
```

然后进入 `retry_wait`。只有 OOM 自动重排；数据错误、断言失败和模型代码异常保留为 `failed`，避免无限重试坏任务。

## 7. 峰值显存的局限

daemon 通过 `nvidia-smi` 按 worker PID 采样峰值。以下情况可能低估：

- 任务启动多个独立 CUDA 子进程；
- 峰值短于采样间隔；
- 驱动不暴露进程级显存；
- 分布式任务在多 GPU 上运行。

所以历史峰值用于辅助下一次估计，不能替代 `--reserve`。
