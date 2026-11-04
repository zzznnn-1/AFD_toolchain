# develop 分支的 P3 完善工作

`main` 的初始提交 `2d30fdb` 保存前一轮工程及实际运行报告，`develop` 在此基础上继续 P0–P3。本轮没有扩展到 P4，也没有终止服务器上的其他服务。Git 未配置作者身份，因此本地提交使用 `Codex <codex@localhost>`，没有修改全局 Git 配置。仓库尚无远端。

## 完成的实现

- `preflight.py` 读取 safetensors 头部和 CUDA 当前空闲容量，按常驻参数、最大待搬运模块和运行预留计算 GPU0 预算，生成明确的 GPU0/CPU device_map。GPU1 容量和无关计算进程也被记录。`resident_layers=auto` 选择当前预算内可容纳的层数，显式层数超预算时不自动改动。
- `--strict-deployment` 在权重加载前检查完全常驻与 GPU1 计算独占；不满足时写入 BLOCKED，退出 2。普通功能模式允许明确记录的 CPU 权重存储。实际状态由采集数据推导，移至资源充足设备后不会继续固定返回 BLOCKED。
- Runtime 验证 Backend completion 的 command ID、handle、请求/层归属、dtype/layout、容量和追加长度。非法响应不会覆盖已知缓存状态。Backend 清理确保在 transport 关闭异常时仍释放 KV 管理器。
- 模型 runner 记录 64 层输出设备，要求模型计算输出在 GPU0；检查有限 logits、全局和所有 Full Attention 层缓存长度、Adapter 退出恢复，并保存实际 Prompt Token IDs。每次运行单独重置显存峰值计数。
- Greedy 序列不同会使完整 suite 失败；旧实现只写布尔字段但仍可能通过，已修复。空的数值对比不能通过。
- `evidence.py` 保存测试开始时的代码哈希、Git commit/branch、版本、时间和 run_id，结束时重新校验。JSONL 记录同一 run_id。`acceptance.py` 逐命令重建 Handle 生命周期，检查每个 Token 的四段传输、GPU1 Attention、长度递增和释放，而非只看累计数字。
- 验收检查使用显式异常，不依赖可被 Python 优化模式删除的 `assert`。代码改动、日志混用、丢传输或 kernel、重复位置、缺释放等情况会失败。审计失败会覆盖旧的 acceptance PASS，防止误读上次结果。
- `validate_all.py` 顺序运行所有阶段，保留日志、命令退出码、GPU 采样和依赖清单。每次使用新目录。Markdown 报告按本次产物生成，不复制旧的成功结论。

## 验证与使用

```bash
.venv/bin/python scripts/validate_all.py --output-dir reports/new-run
.venv/bin/python scripts/verify_results.py --artifact-dir reports/new-run/raw
.venv/bin/python scripts/verify_results.py --artifact-dir reports/new-run/raw --strict-deployment
```

统一运行的退出码表示功能管线是否成功；验收 JSON 中分别列出功能和部署状态。严格审计无法通过部署条件时退出 2，数据或数值错误退出 1。

回归测试覆盖资源不足与资源充分两种部署条件、坏 Backend completion、截断权重、运行前后源码变化、错误 run_id、传输/kernel 缺失、Token 重放和未释放缓存。本次完整模型仍执行短 Prompt/32 步、512-token Prompt/128 步、4096-token Prompt/32 步，三模式及 Greedy 比较保持原有精度门限。

本次运行记录位于 `reports/develop/`；旧 `reports/raw/` 是第一轮历史基线，不能被带执行来源校验的新版审计器当作当前结果认证。参数大小与预留预算都保存在 placement/P3 JSON；CPU 权重搬运流量不包含在 PNM Attention 传输统计内。

## 仍然存在的外部限制

24 GiB GPU0 无法容纳约 28.3 GiB 完整模型参数加运行内存。CPU offload 已让完整模型功能跑通，但没有解除全常驻限制。GPU1 的既有其他计算进程使物理独占条件不成立。解除这些条件需要容量足够的 GPU0 与空闲 GPU1，然后重新执行相同验证入口；不能仅修改报告状态。

`auto` 只选择常驻 decoder 层数，保留明确的 CPU 存储策略。在容量足够的机器上验证完全常驻，需要显式运行 `scripts/check_placement.py --fully-resident --strict-deployment` 和 `scripts/run_qwen.py --suite --fully-resident --strict-deployment --output <本次目录>/raw/P3_suite.json`，之后用同一目录的完整 P0/P1/P2/pytest 证据审计。仅把 resident_layers 设为 64 不代表 embedding/lm_head/vision 也常驻。

P4 继续保留为下一阶段：给原生 Cache 增加长度/位置桥接，移除目标 Layer 3 本地 KV。当前 Runtime/Backend/Transport 的 handle 契约无需改为访问 GPU1 内部 tensor。
