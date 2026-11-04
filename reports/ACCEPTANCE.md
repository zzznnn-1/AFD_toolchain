# P0–P3 最终验收

> 本文件是首次运行的历史基线。`develop` 最新实现和复跑结果见 [develop/ACCEPTANCE.md](develop/ACCEPTANCE.md)。

**总状态：BLOCKED（严格独占部署）；功能与数值验证：PASS。**

已经创建分层工程并实际运行完整官方 FP8 模型。P3 的 `shadow_remote` 确实使用 GPU1 返回的 Attention O，完成 32/128 Token Decode、固定序列四级数值对比和 4096 Token 上下文测试。未把 CPU offload / 共享设备结果宣称为全量 GPU0 常驻或物理独占验收。

| 阶段 | 最终状态 | 实际结果与证据 |
|---|---|---|
| P0 | PASS | 2×RTX4090、CC8.9、P2P=false、NUMA/版本/磁盘/官方配置/Cache/FP8 kernel/实际 QKV 调查；[报告](P0_environment.md) |
| P1 | PASS | 19 个独立算法场景；1/16/128/1024/4096 长度，固定种子、独立 FP32 参考；[报告](P1_attention.md) |
| P2 | PASS | 128 步真实 Host Staging、独立 GPU1 KV、GQA Attention，额外 4096+128 通过；[报告](P2_emulator.md) |
| P3 | BLOCKED | 功能与正确性全部 PASS；严格独占部署条件未满足；[报告](P3_qwen_integration.md) |

| P3 验收项 | 状态 | 证据 |
|---|---|---|
| 官方完整 FP8 checkpoint，未改用 NF4/BF16 模型 | PASS | Revision `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`，400 FP8Linear，missing/unexpected/mismatched 全空 |
| 本实验 GPU1 无模型权重 | PASS | 显式 device_map、参数设备断言，所有模型计算在 GPU0；CPU 保存部分权重 |
| 原生 Prefill，之后 GPU1 仅初始化一次 | PASS | 8 个远端请求，8 次 INIT，8 次 FREE；按 handle 审计 |
| Decode 只传新 Q/K/V | PASS | 所有 544 次 append 精确断言 Q=12288、K=2048、V=2048 bytes |
| GPU1 真正执行 Attention | PASS | 544 次 GPU1 CUDA kernel 事件，KV 与长度逐步校验 |
| GPU1 O 参与后续模型 | PASS | 352 步实际替换；其余 192 步为 shadow_compare |
| Local / shadow_compare / shadow_remote | PASS | 三模式共 11 次完整运行 |
| 原始 O / Attention / Decoder / Logits | PASS | 6 组固定序列对比；四级 Max abs、Mean abs、Max relative、RMS 均为 0 |
| 连续 32 / 128 Token 生成 | PASS | Remote Greedy token 序列与本地全部一致 |
| 4096 上下文 | PASS | 三模式各 32 步；本地和远端长度最终 4128 |
| 位置、mask、GQA、其他层 Cache | PASS | 每步校验 position_ids、Layer 3 及所有 16 个 Full Attention cache 长度；P1/P2 mask 回归 |
| 完整权重常驻 GPU0 | BLOCKED | 实际参数 28.305 GiB > CUDA 可用总容量 23.517 GiB；采用 CPU 权重 offload |
| 整张 GPU1 物理独占 | BLOCKED | 开始时已有无关 VLLM::EngineCore PID 1189119，占约 11 GiB；未终止其他服务 |

资源实测：GPU0 常驻参数 17.033 GiB，CPU 权重存储 11.272 GiB；GPU0 PyTorch allocated 峰值 19.971 GiB，GPU1 本进程 allocated 峰值约 0.063 GiB。整卡使用量（含其他进程、CUDA context、reserved allocator）另见 `raw/P3_suite_gpu.jsonl`，不能与 allocated 混用。Shadow KV capacity=8192，单层 GPU1 分配 32 MiB；4096+32 测试有效 KV 为 16.125 MiB。

PNM 四段拷贝累计 106.453125 MiB，含 8 次 Prefill；每个 Decode 逻辑双向 28 KiB，四段 copy 56 KiB。该统计不包括 CPU 权重 offload，也不表示实际 PCIe 总线全部流量。GPU1 Attention CUDA event 累计 47.193120 ms；同步 CPU 调度和验证开销另记 command wall time。

| 实际运行 | Prompt tokens | Decode steps | 使用远端 O 步数 | Prefill 秒 | Decode 秒 |
|---|---|---|---|---|---|
| short-local | 30 | 32 | 0 | 20.657 | 42.658 |
| short-shadow_compare | 30 | 32 | 0 | 0.941 | 30.354 |
| short-shadow_remote | 30 | 32 | 32 | 1.006 | 30.077 |
| short-remote-greedy | 30 | 32 | 32 | 1.002 | 30.193 |
| medium-local | 512 | 128 | 0 | 32.160 | 120.839 |
| medium-shadow_compare | 512 | 128 | 0 | 1.044 | 120.926 |
| medium-shadow_remote | 512 | 128 | 128 | 1.040 | 121.166 |
| medium-remote-greedy | 512 | 128 | 128 | 0.970 | 121.104 |
| context4096-local | 4096 | 32 | 0 | 10.060 | 30.260 |
| context4096-shadow_compare | 4096 | 32 | 0 | 2.539 | 30.289 |
| context4096-shadow_remote | 4096 | 32 | 32 | 2.376 | 30.712 |

首次上下文运行包含 kernel 编译/预热，不应把 local 与 remote 的这些时间差作为加速结论。CPU offload 主导本机模型推理时间；模拟器不能代表未来 ASIC 吞吐。

自动化测试最终 **13 passed**，没有 skip/failure。完整模型 suite 退出码 0；`verify_results.py` 已逐命令检查真实数据路径和结果，功能审计 PASS，汇总部署状态仍为 BLOCKED。见 [审计 JSON](raw/acceptance.json)、[单元测试日志](raw/unit_tests.log)、[完整模型日志](raw/P3_suite.log)、[逐步数值](raw/P3_suite.json)、[命令/传输/kernel 日志](raw/P3_suite.jsonl)。

工程入口：[README](../README.md)；完整复现命令 [reproduce_commands.txt](raw/reproduce_commands.txt)；软件锁定 [environment.lock.txt](raw/environment.lock.txt)；模型与缓存调查 [MODEL_INSPECTION.md](../docs/MODEL_INSPECTION.md)。固定随机种子 20261104。

已修复：FP8 exclusion regex 误匹配 gate_proj、kernels 版本不兼容、additive mask dtype、报告 set 序列化。保留失败日志，最终未降低精度容差。环境 Transformers 源文件 SHA256 与初始调查一致。

下一步：在更大显存 GPU0、空闲独占 GPU1 上复跑相同 suite，解除部署阻塞；然后单独开展 P4 Cache metadata bridge，保持 Runtime/Backend/Transport 契约。未实现 ASIC 驱动、真实 DMA、vLLM 集成或严格远端 KV。
