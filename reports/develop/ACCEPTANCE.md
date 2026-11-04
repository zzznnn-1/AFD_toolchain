# P0–P3 验收

**总体 BLOCKED；功能与数值 PASS。**

| 阶段 | 状态 | 报告 |
|---|---|---|
| P0 | PASS | [环境与加载](P0_environment.md) |
| P1 | PASS | [独立 Attention](P1_attention.md) |
| P2 | PASS | [Runtime 与模拟器](P2_emulator.md) |
| P3 | BLOCKED | [完整模型](P3_qwen_integration.md) |

完整模型共 11 次运行，GPU1 Attention 544 步，其中 352 步真正使用 GPU1 O。Greedy 对比一致=True。自动化测试 24 项通过。审计错误：无。

严格部署尚存条件：CPU 权重 offload=True；GPU1 无其他计算进程=False。实际参数 28.302 GiB，GPU0 运行时余量和占用见 P0 报告。没有停止服务器现有服务，也没有改变官方权重量化格式。

原始审计 [acceptance.json](raw/acceptance.json) / [acceptance.log](raw/acceptance.log)；完整命令与退出码 [commands.json](raw/commands.json)，依赖 [environment.lock.txt](raw/environment.lock.txt)。执行代码版本 `46e1b96e9664dcd4389b873fe057db991dedfb78`；分支 `develop`。

复现整个验证：`.venv/bin/python scripts/validate_all.py --output-dir reports/new-run`。严格部署检查：`.venv/bin/python scripts/check_placement.py --strict-deployment`，不满足时退出 2；正常功能审计退出 0，证据错误退出 1；`verify_results.py --strict-deployment` 可要求部署条件也必须通过。

当前范围仍为 P0–P3。下一步在满足容量与 GPU1 独占条件的设备上复跑同一入口；之后再单独实施 P4 Cache bridge。此原型的性能不代表真实 ASIC/DMA。


严格运行入口也已实际检查：`scripts/run_qwen.py --strict-deployment --fully-resident --steps 1` 退出 2，未加载模型、未执行任何 Decode，阻塞理由写入 [P3_strict_gate.json](raw/P3_strict_gate.json)。
