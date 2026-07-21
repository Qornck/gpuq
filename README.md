# gpuq：个人机会式 GPU 任务队列

`gpuq` 面向这样的服务器环境：多人直接运行 CUDA 程序，没有强制 Slurm，空闲显存会随其他用户任务变化。它不会接管整台服务器，而是在当前用户范围内维护任务队列，发现某张 GPU 有足够显存时自动启动任务。

核心原则：

- 以**空闲显存**作为硬可行条件，可选限制 GPU 利用率；多个 GPU 均可行时优先选择低负载设备；
- 不要求整张卡空闲，允许在显存足够时与既有任务共存；
- 连续多个采样周期满足条件后才启动，减少瞬时空闲误判；
- 启动前二次采样，降低检查和 CUDA 分配之间的竞态；
- 永不抢占、暂停或终止其他用户的进程；
- 一个任务失败不阻塞队列，记录日志并继续运行下一项；
- 支持取消、重试、依赖、优先级、OOM 自动提高估计和通知。

它是个人使用的 **opportunistic/backfill scheduler**，不是完整 Slurm。其他用户不使用 `gpuq` 时，工具无法真正预留显存，因此不能从理论上消除资源竞态。

调度 daemon 每个采样周期最多启动一个任务。这样二次显存检查始终针对最新状态；
若同一张卡还能放下更多任务，后续采样周期会继续启动。

## 安装

要求：Python 3.10+、NVIDIA 驱动和可执行的 `nvidia-smi`。

```bash
cd /home/liuzeyang/projects/gpuq
python -m pip install --user -e .
gpuq init
gpuq doctor
```

开发安装和测试：

```bash
python -m pip install -e '.[dev]'
pytest
```

## 最短使用流程

终端一：启动调度器。

```bash
gpuq daemon
```

终端二：提交任务。

```bash
gpuq submit \
  --name p0_3_flat_reader \
  --gpu 1 \
  --memory 5000 \
  --reserve 1500 \
  --max-utilization 70 \
  --stable-samples 3 \
  --cwd /home/liuzeyang/projects/grd \
  --env PYTHONPATH=. \
  -- python train/train_flat_coordinate_reader.py --device cuda:0 --bf16
```

查看状态和日志：

```bash
gpuq list
gpuq show 1
gpuq logs -f 1
gpuq gpus
```

取消或重试：

```bash
gpuq cancel 1
gpuq retry 1 --memory 6500
```

完整参数见 [CLI 参数手册](docs/cli_reference.md)，调度语义与安全边界见 [调度模型](docs/scheduling.md)。

## 显存判断

普通任务的基本条件为：

```text
GPU.free_memory_mb >= job.memory_mb + job.reserve_mb
```

如果设置 `--max-utilization 60`，还要求采样时 GPU 利用率不超过 60%。条件必须连续满足 `--stable-samples` 次；启动前会再采样一次。

调度器按优先级处理任务。显存满足任务估计与安全余量的 GPU 才会成为候选；多个
候选同时满足时，首先选择当前 `GPU-Util` 最低的设备。利用率相同时再使用
best-fit，即选择运行后剩余显存最少的 GPU，尽量为后续大任务保留大块空闲显存。

默认每 10 秒采样一次并要求连续 3 个合格样本。三个样本横跨约 20 秒；考虑任务
条件可能刚好在采样后满足，实际从满足条件到启动通常约 20–30 秒，启动前还会
立即复查一次。

## 物理 GPU 与任务内编号

如果调度器选择物理 GPU 1，会为任务设置：

```bash
CUDA_VISIBLE_DEVICES=1
```

因此任务内部只看见一张卡，并应使用 `cuda:0`，而不是 `cuda:1`：

```bash
gpuq submit --gpu 1 --memory 5000 -- python train.py --device cuda:0
```

## 任务失败和 OOM

- 命令退出码为 0：`completed`；
- 非零退出：`failed`，保留 stdout/stderr；
- 任务失败后 daemon 自动继续检查后续队列；
- 使用 `--retry-on-oom --max-retries 2` 时，检测到常见 CUDA OOM 文本后，显存估计至少扩大 25%，任务进入 `retry_wait`；
- 普通代码错误不会自动伪装成显存问题。

例如：

```bash
gpuq submit \
  --memory 8000 \
  --reserve 1500 \
  --retry-on-oom \
  --max-retries 2 \
  -- python train.py
```

## 常驻运行

仓库提供 [systemd 用户服务](systemd/gpuqd.service)：

```bash
mkdir -p ~/.config/systemd/user
cp systemd/gpuqd.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now gpuqd.service
systemctl --user status gpuqd.service
```

服务默认假定 `gpuq` 安装在 `~/.local/bin/gpuq`。如果使用虚拟环境，需要修改 `ExecStart` 为实际绝对路径。

## 数据位置

默认状态目录：

```text
~/.local/state/gpuq/
├── queue.sqlite3
├── queue.sqlite3-wal
├── gpuqd.lock
└── logs/
```

配置文件默认位于：

```text
~/.config/gpuq/config.toml
```

可通过 `GPUQ_STATE_DIR`、`GPUQ_CONFIG` 或全局 CLI 参数覆盖。

## 当前不做的事情

- 不杀死或抢占其他用户任务；
- 不创建占位 CUDA tensor 来“预留”显存；
- 不保证其他用户不会在二次检查后抢先分配；
- 不提供跨用户公平性、配额、账户和计费；
- 不自动判断训练是否科学有效，只管理进程与资源；
- 第一版一次只为一个任务分配一张物理 GPU，不支持单任务多 GPU。

## 许可证

MIT，见 [LICENSE](LICENSE)。
