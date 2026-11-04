# P3 功能与数值：PASS；严格独占部署：BLOCKED

> 本文件是首次运行的历史基线。`develop` 最新复跑报告见 [develop/P3_qwen_integration.md](develop/P3_qwen_integration.md)。

完整官方 Qwen/Qwen3.8-27B-FP8，Revision `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`。实际结果以 `raw/P3_suite.json` 和 `raw/P3_suite.log` 为准；RUNNING 或 FAIL 不能算验收通过。

**完整 P3 验收状态为 BLOCKED**：下表记录实际功能和数值结果，但本机仍使用 CPU 权重 offload，且 GPU1 存在既有的无关推理服务，未达到全量 GPU0 常驻/物理独占条件。不把本次功能通过宣称为严格独占部署完成。



| 运行 | Prompt tokens | Decode steps | GPU1 Attention 次数 | GPU1 O 实际使用次数 | Decode 秒 |
|---|---|---|---|---|---|
| short-local | 30 | 32 | 0 | 0 | 42.658 |
| short-shadow_compare | 30 | 32 | 32 | 0 | 30.354 |
| short-shadow_remote | 30 | 32 | 32 | 32 | 30.077 |
| short-remote-greedy | 30 | 32 | 32 | 32 | 30.193 |
| medium-local | 512 | 128 | 0 | 0 | 120.839 |
| medium-shadow_compare | 512 | 128 | 128 | 0 | 120.926 |
| medium-shadow_remote | 512 | 128 | 128 | 128 | 121.166 |
| medium-remote-greedy | 512 | 128 | 128 | 128 | 121.104 |
| context4096-local | 4096 | 32 | 0 | 0 | 30.260 |
| context4096-shadow_compare | 4096 | 32 | 32 | 0 | 30.289 |
| context4096-shadow_remote | 4096 | 32 | 32 | 32 | 30.712 |

固定相同 token 序列对比；各步骤同时保留 Max abs/Mean abs/Max relative/RMS 和 finite。下表汇总最大绝对误差：

| 上下文 | 模式 | 状态 | 原始 O | Gate/o_proj 后 | Decoder 后 | Logits | Argmax 一致率 |
|---|---|---|---|---|---|---|---|
| short | shadow_compare | PASS | 0 | 0 | 0 | 0 | 1.0000 |
| short | shadow_remote | PASS | 0 | 0 | 0 | 0 | 1.0000 |
| medium | shadow_compare | PASS | 0 | 0 | 0 | 0 | 1.0000 |
| medium | shadow_remote | PASS | 0 | 0 | 0 | 0 | 1.0000 |
| context4096 | shadow_compare | PASS | 0 | 0 | 0 | 0 | 1.0000 |
| context4096 | shadow_remote | PASS | 0 | 0 | 0 | 0 | 1.0000 |

每次调用 GPU1 Attention 与同次 GPU0 Attention 原始 O 使用 atol=0.02/rtol=0.02；跨模型执行四级输出使用事前设定 atol=0.25/rtol=0.02。Prefill 使用原生逻辑，跨模式 logits 要求完全相同。贪心结果是否一致保存于 runs[].matches_local_generated_tokens，固定长度运行不在 EOS 提前停止。

Adapter 没有重写 Qwen forward：仅给 Layer 3 安装独立配置的 attention interface，shadow_remote 返回 GPU1 O，原 forward 继续 sigmoid gate 和 o_proj。回归测试对远端 O 注入小扰动，验证输出替换路径；完整模型 logs 记录 remote_outputs_used。其他 63 层原生执行；每步断言所有 16 个 Full Attention KV 长度。

实际 Q/K/V 元数据、projection 参数 dtype/shape、device map、GPU0/GPU1 allocated/reserved/peak、prefill/final GPU1 KV 信息均在 JSON 中。逐命令分阶段数据与计算事件在 `raw/P3_suite.jsonl`；decode 只发 Q_new/K_new/V_new。

容量限制：本机不能完整权重常驻 GPU0，采用 CPU 权重 offload，模型计算在 GPU0，GPU1 仅存本实验的 Emulator KV/scratch。未修改 FP8 checkpoint 的量化格式；CPU offload 降低吞吐。JSONL 传输字节仅统计 PNM 数据路径，不包括 CPU 权重 offload 流量。causal_conv1d 使用原生 PyTorch fallback，其他层未改写。GPU1 同时有其他进程，本结果不构成物理设备独占性能验收。

本阶段为 **Shadow Mode**：GPU0 仍保留 Layer 3 本地完整 KV，不是 P4 Strict Remote KV。下一阶段可在 native Cache bridge 上移除本地该层 KV，同时保留默认 get_seq_length 对 Layer 3 的依赖；Runtime/Backend/Transport 不需要变更。

复现：`.venv/bin/python scripts/run_qwen.py --suite --output reports/raw/P3_suite.json`。独立入口支持 `--mode local`、`--mode shadow_compare`、`--mode shadow_remote`。

