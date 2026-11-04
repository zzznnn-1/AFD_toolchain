# 双 GPU PNM / Qwen3.8-27B-FP8

Python/PyTorch/Transformers 原型：GPU0 运行模型计算，GPU1 维护 Layer 3 BF16 KV 并计算 Attention，CPU Runtime 通过显式 pinned-memory staging 传输。现已扩展到 P4 Strict Remote KV 和 P5 性能/压力/故障验证，P6 提供 Backend 接入契约并检测真实硬件依赖。

工程代码位于 `src/pnm/`，配置位于 `configs/pnm.json`，运行入口位于 `scripts/`。项目仅保留 Qwen3.8-27B-FP8 原型及其测试、文档和报告。

`main` 保存首轮 P0–P3 基线；`develop` 补齐加载预检、运行来源记录与失败校验。首轮报告在 [reports/ACCEPTANCE.md](reports/ACCEPTANCE.md)，develop 的完整复跑报告在 [reports/develop/ACCEPTANCE.md](reports/develop/ACCEPTANCE.md)。各次报告对应自己的命令、源代码哈希和实际日志；本机仍需 CPU 权重 offload，GPU1 上有既有其他服务。

最新全方案结果见 [reports/complete/ACCEPTANCE.md](reports/complete/ACCEPTANCE.md)。当前 develop 独立工作区为 `/home/jinyuzhe/attention-develop`，主目录 `/home/jinyuzhe/attention` 保留用户的 main 工作区。新增 `strict_remote` 在 Prefill 后释放 Layer 3 本地 KV，仅 GPU1 保留完整历史；物理 GPU 独占、全部模型权重常驻和 Strict Remote KV 是不同验收条件。

本次实测：37 项自动化测试通过；六个上下文各 128 步 Strict 对比的四级误差均为 0；Strict 共 912 步真正使用 GPU1 O；2048 步设备压力和 7 个完整模型故障场景通过。GPU0 目标层历史 KV 始终为 0。软件功能 P0–P5 PASS；独占部署及 P6 真实硬件接入仍 BLOCKED，完整缺项在报告中。

```bash
cd /home/jinyuzhe/attention-develop
.venv/bin/python scripts/complete_plan.py --output-dir reports/new-complete-run
.venv/bin/python scripts/run_qwen.py --mode strict_remote --prompt-tokens 128 --steps 128
```

全方案入口实际运行 P0/P1/P2、所有回归、16×128 步设备压力、完整模型 4/16/128/1024/4096/8192 上下文各 128 步 local/strict 对比、严格模式 Greedy、Shadow 回归和完整模型故障注入，生成 JSON/CSV/Markdown。16/128-token 性能窗口取同一次 128 步运行的前缀，前 8 步不计入百分位，不伪装成独立重复实验。原先 `validate_all.py` 仍用于 P0–P3 专项复跑。

真实 ASIC/PCIe 驱动及 DMA/队列 ABI 尚未提供，P6 不能声明实际验收通过；插件工厂、描述符校验和缺依赖时的明确失败路径已实现。vLLM 在规范中后置，未把服务器现有的其他 vLLM 服务当成本项目集成。详见 [Strict Remote 说明](docs/STRICT_REMOTE.md) 和 [驱动契约](docs/DRIVER_CONTRACT.md)。

## 环境

本机已有 `.venv`，系统没有 `python` 命令。使用 `.venv/bin/python`，或先 `source .venv/bin/activate`。锁定版本见 `reports/raw/environment.lock.txt` 和 `requirements-pnm.txt`。不得在此实验中使用 `device_map="auto"`。

```bash
source .venv/bin/activate
python -m pip install -r requirements-pnm.txt
python -m pip install -e . --no-deps
python scripts/validate_all.py --output-dir reports/new-run
```

统一入口顺序执行环境检测、容量预检、FP8 kernel、P1、P2、pytest、完整模型 suite、日志审计和报告。每次必须指定新的输出目录，避免覆盖上次结果。模型快照会自动从固定 Revision 的本地缓存读取，缺失时下载。所有命令、退出码和耗时保存到该目录的 `raw/commands.json`。

独立运行：

```bash
python scripts/check_placement.py
python scripts/check_placement.py --strict-deployment
python scripts/run_qwen.py --mode local --steps 32
python scripts/run_qwen.py --mode shadow_compare --steps 32
python scripts/run_qwen.py --mode shadow_remote --steps 128
python scripts/run_qwen.py --suite --output reports/manual/P3_suite.json
python scripts/benchmark.py
python scripts/verify_results.py --artifact-dir reports/new-run/raw
```

官方模型和 Revision 固定在 `configs/pnm.json`。模型下载至 Hugging Face cache，快照路径保存到 `reports/raw/model_path.txt`。依赖及模型可能需要网络，运行结果按输出 JSON/JSONL 保存。

当前 Transformers 5.19.0 的量化排除列表使用正则前缀匹配，官方配置的 `mlp.gate` 会误匹配 `mlp.gate_proj`。加载器依据 checkpoint 的 FP8 scale key 给歧义名称添加结束锚点，避免丢弃 scale 并把 FP8 权重直接转成未缩放 BF16；不修改环境源码或 checkpoint。加载时严格检查 400 个 decoder FP8Linear 与 checkpoint 对应，并要求 missing/unexpected/mismatched keys 为空。直接绕过本加载器调用 from_pretrained 在此版本组合上会出错。

两张 4090 无法让完整 FP8 模型常驻 GPU0。默认 `resident_layers="auto"`，从实际 safetensors 文件头计算各层字节，按 GPU0 当时空闲容量选择常驻层数，并留出最大 CPU 暂存模块及 `workspace_reserve_gib=4.0` 的运行预算。其余层、embedding、lm_head、视觉模块以 CPU 为权重存储，Accelerate 在需要时把相应模块送到 GPU0 执行；GPU1 不放置本实验任何模型权重。可用 `--resident-layers N` 固定，但超过预算会在加载前阻止运行。`--fully-resident` 要求全部权重常驻 GPU0；`--strict-deployment` 额外要求无 CPU 权重和无 GPU1 其他计算进程。不满足条件时退出 2，写入 BLOCKED 及具体字节和 PID。预检预算不是任意上下文的显存保证，仍需实际运行验证。

`--suite` 运行短 Prompt/32 步、512-token Prompt/128 步、4096-token Prompt/32 步；先做 local greedy，再使用相同 token 序列运行 shadow_compare、shadow_remote，比较原始 O、gate/o_proj 后输出、decoder 输出和 logits；短/中上下文还独立运行 remote greedy。固定种子 20261104。JSON 记录每步误差、位置、KV 长度、模式、GPU 显存和远端 O 使用计数；JSONL 记录每条命令、四段传输及 GPU1 kernel 时间。退出码非零或 JSON 中 FAIL 不算验收通过。

精度：独立 BF16 Attention 使用逐元素 `atol=0.02, rtol=0.02`；完整模型跨运行输出预设 `atol=0.25, rtol=0.02`，同时保留四项原始误差，不能仅看 argmax。相对误差分母最小 1e-3，接近零的元素相对误差可能很大。具体测量见报告。

每次测试记录执行前后的源代码 SHA256、Git commit/branch、版本、时间和 run_id；JSONL 也带同一 run_id。审计验证所有阶段的源码与当前代码匹配，逐条验证 ALLOC→INIT→APPEND→FREE、Token 位置、四段传输字节和 GPU1 kernel，不能只凭文件存在通过。Greedy 不一致、缺少日志、不同运行日志混用、代码变更后复用旧报告均失败。64 层实际输出设备必须为 GPU0，目标层 Adapter 必须恢复。

`verify_results.py` 退出码 0 表示功能证据审计通过，部署状态由本次预检和实际 device_map 推导；加 `--strict-deployment` 后，部署 BLOCKED 返回 2，任何证据错误返回 1。旧版没有 execution provenance 的产物应保留为历史报告，不能用当前验证器重新认证。传输日志仅统计 PNM Q/K/V/O，不包括 CPU 权重 offload 流量。

各阶段实际状态、限制与复现日志见 `reports/`。Shadow Mode 保留两份 KV；`strict_remote` 使用 metadata-only CacheBridge。硬件上 GPU1 已有其他服务，本实验未更改其进程；性能数字是共享服务器上的软件路径测量，不代表 ASIC/KXP 峰值性能。
