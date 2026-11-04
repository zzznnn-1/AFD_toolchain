# P3：功能 PASS；独占部署 BLOCKED

首轮完整三模式/32/128-token/4096-context 验收保存在历史 reports/develop。本轮针对共享 Runtime/传输改动，使用当前同一 checkpoint 运行 shadow_compare/shadow_remote 各 16-token 回归，并在 P4 扩展完整上下文固定序列验证。

当前回归：`[('shadow_compare', 'PASS', 16), ('shadow_remote', 'PASS', 16)]`。Prefill 原生，Adapter 退出恢复原配置，64 个 decoder 输出设备均检查为 GPU0。

证据：[当前完整模型结果](raw/P4_suite.json)。GPU0 全权重常驻和 GPU1 物理独占未满足时，P3 总验收继续保留 BLOCKED，不把 CPU offload/共享服务隐去。

