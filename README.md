# 双 GPU PNM / Qwen3.8-27B-FP8

Python/PyTorch/Transformers 原型：GPU0 运行模型计算，GPU1 维护 Layer 3 BF16 KV 并计算 Attention，CPU Runtime 通过显式 pinned-memory staging 传输。当前范围是 P0–P3 Shadow Mode。

工程代码位于 `src/pnm/`，配置位于 `configs/pnm.json`，运行入口位于 `scripts/`。项目仅保留 Qwen3.8-27B-FP8 原型及其测试、文档和报告。

实际验收见 [reports/ACCEPTANCE.md](reports/ACCEPTANCE.md)：P0/P1/P2 PASS；P3 功能和数值 PASS，严格独占部署 BLOCKED。完整模型三模式共 11 次运行，544 次 GPU1 Attention，其中 352 步实际使用远端 O；短/512/4096 上下文的固定序列四级误差均为 0，32/128 Token Greedy 序列一致。部署仍依赖 CPU 权重 offload，GPU1 上有既有其他服务。

## 环境

本机已有 `.venv`，系统没有 `python` 命令。使用 `.venv/bin/python`，或先 `source .venv/bin/activate`。锁定版本见 `reports/raw/environment.lock.txt` 和 `requirements-pnm.txt`。不得在此实验中使用 `device_map="auto"`。

```bash
source .venv/bin/activate
python -m pip install -r requirements-pnm.txt
python -m pip install -e . --no-deps
python scripts/check_environment.py
python scripts/probe_fp8.py
python scripts/test_attention.py
python scripts/test_emulator.py --steps 128
python -m pytest -q --junitxml=reports/raw/unit_tests.xml
python scripts/download_model.py
python scripts/run_qwen.py --mode local --steps 32
python scripts/run_qwen.py --mode shadow_compare --steps 32
python scripts/run_qwen.py --mode shadow_remote --steps 128
python scripts/run_qwen.py --suite --output reports/raw/P3_suite.json
python scripts/benchmark.py
python scripts/verify_results.py
```

官方模型和 Revision 固定在 `configs/pnm.json`。模型下载至 Hugging Face cache，快照路径保存到 `reports/raw/model_path.txt`。依赖及模型可能需要网络，运行结果按输出 JSON/JSONL 保存。

当前 Transformers 5.19.0 的量化排除列表使用正则前缀匹配，官方配置的 `mlp.gate` 会误匹配 `mlp.gate_proj`。加载器依据 checkpoint 的 FP8 scale key 给歧义名称添加结束锚点，避免丢弃 scale 并把 FP8 权重直接转成未缩放 BF16；不修改环境源码或 checkpoint。加载时严格检查 400 个 decoder FP8Linear 与 checkpoint 对应，并要求 missing/unexpected/mismatched keys 为空。直接绕过本加载器调用 from_pretrained 在此版本组合上会出错。

两张 4090 无法让完整 FP8 模型常驻 GPU0。默认将 48 个 decoder layers 常驻 GPU0，其余层、embedding、lm_head、视觉模块以 CPU 为权重存储，Accelerate 在需要时把相应模块送到 GPU0 执行；GPU1 不放置本实验任何模型权重。所有权重仍来自官方 FP8 checkpoint，不重做 4-bit 量化。可用 `--resident-layers N` 调整。全常驻探测参数为 `--resident-layers 64 --fully-resident`，本机容量不足，不是推荐运行方式。

`--suite` 运行短 Prompt/32 步、512-token Prompt/128 步、4096-token Prompt/32 步；先做 local greedy，再使用相同 token 序列运行 shadow_compare、shadow_remote，比较原始 O、gate/o_proj 后输出、decoder 输出和 logits；短/中上下文还独立运行 remote greedy。固定种子 20261104。JSON 记录每步误差、位置、KV 长度、模式、GPU 显存和远端 O 使用计数；JSONL 记录每条命令、四段传输及 GPU1 kernel 时间。退出码非零或 JSON 中 FAIL 不算验收通过。

精度：独立 BF16 Attention 使用逐元素 `atol=0.02, rtol=0.02`；完整模型跨运行输出预设 `atol=0.25, rtol=0.02`，同时保留四项原始误差，不能仅看 argmax。相对误差分母最小 1e-3，接近零的元素相对误差可能很大。具体测量见报告。

`verify_results.py` 验证现存测试产物；退出码 0 表示功能证据审计通过，整体部署状态以输出中的 BLOCKED 为准。要重新执行测试，请先运行前面的测试及完整模型 suite 命令。传输日志仅统计 PNM Q/K/V/O，不包括 CPU 权重 offload 流量。

P0–P3 的实际状态、限制与复现日志见 `reports/`。Shadow Mode 有本地和远端两份 KV，尚未实现 P4。硬件上 GPU1 已有其他服务，本实验未更改其进程；性能数字是共享服务器上的软件路径测量。
