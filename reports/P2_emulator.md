# P2 PNM Emulator：PASS

实际 GPU0 → pinned CPU → GPU1 → pinned CPU → GPU0；源/目标均为本机 RTX4090，未使用 peer copy。固定种子 20261104。Prefill 128 tokens 仅初始化一次，连续 128 步后长度 256。

| 验收项 | 状态 | 证据 |
|---|---|---|
| Prefill 一次，Decode 仅新 QKV | PASS | JSONL 每 token K/V 各 2048 bytes，Q 12288 bytes；测试精确断言总字节 |
| GPU1 真实 Attention | PASS | 128 次 CUDA event 计时，独立 FP32 参考逐步校验 |
| 128 步输出正确 | PASS | raw/P2_emulator.json 的 steps_detail |
| GPU1 独立 KV 与增量长度 | PASS | 初始 128，最终 256，allocated_bytes=1114112 |
| Host pool 稳定 | PASS | 预分配 1139776 bytes，所有 slot is_pinned=true，循环中容量不增长 |
| 设备显存稳定/释放 | PASS | warm_memory vs final_memory；12 次 alloc/init/append/free 单元测试 |
| 错误与重复命令 | PASS | 无效 handle、ownership、长度、位置、shape、dtype、mask、容量、重复命令、失败回滚单元测试 |
| 4096 历史+128 decode | PASS | raw/benchmark/context_4096.json |

四段字节：`{"command": 0, "cuda:0_d2h": 2621440, "cuda:1_h2d": 2621440, "attention": 0, "cuda:1_d2h": 1572864, "cuda:0_h2d": 1572864}`。四段与 kernel CUDA 时间总和 ms：`{"command": 165.0425000116229, "cuda:0_d2h": 6.588064006529748, "cuda:1_h2d": 5.213216026313603, "attention": 20.65507210791111, "cuda:1_d2h": 2.3525759978219867, "cuda:0_h2d": 1.8907200023531914}`。每步逐项数据见 JSONL，命令 wall time 包含调度开销。

这是同步正确性基线：non_blocking CUDA copy 在专用 streams 发出，但每次用 events 等待 DMA 完成后才复用 host slots；没有宣称传输计算重叠。每步 28 KiB 逻辑负载对应 56 KiB 四段 copy；不把该数值称为精确 PCIe 总线流量。服务器另有 GPU1 服务，性能数字有共享设备干扰。

复现：`.venv/bin/python scripts/test_emulator.py --steps 128`、`.venv/bin/python scripts/benchmark.py --lengths 16 1024 4096`、`.venv/bin/python -m pytest -q`。日志 `raw/P2_emulator.log`、`raw/P2_emulator.jsonl`、`raw/benchmark.log`、`raw/unit_tests.log`、`raw/unit_tests.xml`。

