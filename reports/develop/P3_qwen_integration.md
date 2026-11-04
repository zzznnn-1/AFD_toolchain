# P3：BLOCKED

功能及数值状态 **PASS**；严格部署状态 **BLOCKED**。判断来自本次预检、真实 device_map 和 trace，未写死为 PASS/BLOCKED。



| 运行 | Prompt tokens | Decode steps | GPU1 kernel 次数 | 使用远端 O 次数 | Decode 秒 |
|---|---|---|---|---|---|
| short-local | 30 | 32 | 0 | 0 | 40.639 |
| short-shadow_compare | 30 | 32 | 32 | 0 | 29.576 |
| short-shadow_remote | 30 | 32 | 32 | 32 | 28.945 |
| short-remote-greedy | 30 | 32 | 32 | 32 | 28.449 |
| medium-local | 512 | 128 | 0 | 0 | 112.608 |
| medium-shadow_compare | 512 | 128 | 128 | 0 | 112.749 |
| medium-shadow_remote | 512 | 128 | 128 | 128 | 112.349 |
| medium-remote-greedy | 512 | 128 | 128 | 128 | 112.832 |
| context4096-local | 4096 | 32 | 0 | 0 | 28.086 |
| context4096-shadow_compare | 4096 | 32 | 32 | 0 | 28.177 |
| context4096-shadow_remote | 4096 | 32 | 32 | 32 | 28.627 |

| 上下文 | 模式 | 状态 | Max abs 原始 O | Attention 输出 | Decoder 输出 | Logits |
|---|---|---|---|---|---|---|
| short | shadow_compare | PASS | 0 | 0 | 0 | 0 |
| short | shadow_remote | PASS | 0 | 0 | 0 | 0 |
| medium | shadow_compare | PASS | 0 | 0 | 0 | 0 |
| medium | shadow_remote | PASS | 0 | 0 | 0 | 0 |
| context4096 | shadow_compare | PASS | 0 | 0 | 0 | 0 |
| context4096 | shadow_remote | PASS | 0 | 0 | 0 | 0 |

相同 Prompt/token IDs 写入每个运行；固定序列逐级误差保存四项指标，原始 O 同步对比 atol=0.02/rtol=0.02，完整模型跨运行 atol=0.25/rtol=0.02；没有为复跑放宽。Greedy 序列不一致现在会导致失败。每次运行验证 64 个 decoder 的输出设备为 GPU0，卸载的权重仅存于 CPU；逐步检查所有 Full Attention 层长度。Adapter 退出后必须恢复原 config。

P3 使用本机 Qwen forward 的可注册 Attention interface，GPU1 O 在 reshape、sigmoid gate 和 o_proj 前返回。Prefill 保持原生执行。本阶段保留本地完整 KV 用于 Shadow 验证，不是 P4。

GPU1 既有进程：`[{'gpu_uuid': 'GPU-ebe16217-2373-4202-8599-dd10af2b39bd', 'pid': 1189119, 'name': 'VLLM::EngineCore', 'used_mib': 11260}]`。CPU 权重 offload=True；CPU offload 流量不计入 PNM 四段 copy。系统条件不满足时不声称严格部署完成。显存峰值现在每次运行重新计数，记录于各 runs[].gpu_memory。

源码 Git commit `46e1b96e9664dcd4389b873fe057db991dedfb78`，分支 `develop`；run_id `45c2053725734f0c8090351135d11e48`。执行前后 source SHA256 必须一致，审计时还要与当前代码匹配。任何缺失的 transfer/kernel/free、串用其他运行日志、位置重复或来源版本变更都不能通过。

证据：[P3_suite.log](raw/P3_suite.log)、[P3_suite.json](raw/P3_suite.json)、[P3_suite.jsonl](raw/P3_suite.jsonl)、[P3_suite_gpu.jsonl](raw/P3_suite_gpu.jsonl)。复现：`.venv/bin/python scripts/run_qwen.py --suite --output raw/P3_suite.json`。

