# P2：PASS

GPU0→Pinned Host→GPU1→Pinned Host→GPU0 完整路径，连续 128 步通过独立参考；Prefill 4096，最终长度 4224。历史只初始化一次；物理初始化分 256-token chunks，固定 pinned pool 复用，KV 管理器一次分配容量，Decode 不使用 torch.cat。

四段 copy/kernel 原始统计：`{"counts": {"queue": 263, "command": 263, "cuda:0_d2h": 416, "cuda:1_h2d": 416, "kv_init": 16, "kv_append": 128, "attention": 128, "cuda:1_d2h": 128, "cuda:0_h2d": 128}, "bytes": {"queue": 0, "command": 0, "cuda:0_d2h": 18874368, "cuda:1_h2d": 18874368, "kv_init": 0, "kv_append": 0, "attention": 0, "cuda:1_d2h": 1572864, "cuda:0_h2d": 1572864}, "milliseconds": {"queue": 1.7496738582849503, "command": 185.259988065809, "cuda:0_d2h": 13.629888016730547, "cuda:1_h2d": 5.756192020140588, "kv_init": 8.651615545153618, "kv_append": 5.256831929087639, "attention": 20.15859242528677, "cuda:1_d2h": 2.0125760016962886, "cuda:0_h2d": 1.7656319979578257}}`。

协议包含版本、generation、request/layer/command ID、handle、长度/位置、dtype/layout，以及可序列化 shape/stride/bytes 描述。Reset 使旧 generation Handle 失效，重复 Command ID 拒绝。Timeout Unknown 和重复 completion 使缓存 poison，查询不能自动解除 poison。取消、过期 handle、资源不足、实际 staging 异常都有测试。

证据：[P2 数据](raw/P2_emulator.json)、[逐命令日志](raw/P2_emulator.jsonl)、[测试结果](raw/unit_tests.log)。

