# Changelog / 变更日志

All notable changes are recorded here. / 所有重要变更记录于此。

## [0.2.0] - 2026-07-21

### English

- Made automatic VRAM discovery the default: submit without `--memory`.
- Added unbounded retry for recognized CUDA/HIP OOM only, with learned launch-free,
  observed-peak, and failed-allocation floors.
- Added terminal `oom_exceeds_device` detection when no eligible single GPU can fit.
- Kept numeric memory estimates and bounded `--retry-on-oom --max-retries` compatible.
- Added an interactive 20-row list pager with arrow navigation and an alternate screen
  that leaves terminal scrollback untouched.
- Added backward-compatible SQLite migration and new auto-memory diagnostics.
- Fixed Python 3.10 configuration loading with a conditional `tomli` dependency.
- Added bilingual documentation, SVG branding, CI, issue template, package metadata,
  security and contribution policies.

### 中文

- 默认启用自动显存发现：提交时可以不写 `--memory`。
- 只有识别到 CUDA/HIP OOM 才不限次数重试，并综合启动空闲显存、观测峰值和
  失败分配量学习下界。
- 当所有允许单卡都无法容纳时，以 `oom_exceeds_device` 明确终止。
- 保留数字显存估计及有界 `--retry-on-oom --max-retries` 的兼容行为。
- 新增默认每页 20 条的交互列表、方向键翻页和不污染终端历史的备用屏幕。
- 新增向后兼容的 SQLite 迁移及 auto 显存诊断字段。
- 通过条件依赖 `tomli` 修复 Python 3.10 配置读取。
- 补齐双语文档、SVG 品牌标志、CI、issue 模板、包元数据、安全与贡献政策。

## [0.1.0] - 2026-07-20

- Initial personal opportunistic GPU queue with persistent SQLite jobs, stable memory
  sampling, allowlists, priorities, dependencies, cancellation, logs, notifications,
  bounded OOM retry, and a systemd user service.
- 首个个人机会式 GPU 队列版本：包含 SQLite 持久任务、稳定显存采样、GPU 白名单、
  优先级、依赖、取消、日志、通知、有界 OOM 重试和 systemd 用户服务。
