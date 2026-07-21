# Security policy / 安全政策

## English

The supported release line is `0.2.x`. Please report vulnerabilities privately through
GitHub's **Report a vulnerability** feature when available. If private reporting is not
available, contact the maintainer through the email listed in the repository commit
metadata before opening a public issue.

Do not publish credentials, private paths, command contents, environment variables, or
logs that may contain research data. A useful report includes the affected version,
impact, minimal reproduction, and a proposed mitigation if known.

gpuq executes commands supplied by the local user and is not a sandbox. Its SQLite
database, state directory, logs, configuration, and notification command must be
protected with normal user filesystem permissions. Do not point a queue at untrusted
commands and do not expose its state directory for writes by other users.

## 中文

当前支持的版本线为 `0.2.x`。若 GitHub 提供 **Report a vulnerability**，请优先
私密报告漏洞；若不可用，请先通过仓库提交元数据中的维护者邮箱联系，不要直接公开
高风险细节。

不得发布凭据、隐私路径、命令内容、环境变量或可能含研究数据的日志。有效报告应包含
受影响版本、影响范围、最小复现，以及已知时的缓解建议。

gpuq 会执行本地用户提交的命令，它不是安全沙盒。SQLite 数据库、状态目录、日志、
配置和通知命令应使用正常的用户文件权限保护。不要提交不可信命令，也不要允许其他
用户写入状态目录。
