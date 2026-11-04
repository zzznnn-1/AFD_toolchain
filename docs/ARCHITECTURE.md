# PNM v1 / P0–P3

模型适配器 → Runtime → Backend.execute(Command, payload) → HostStaging → GPU1 CacheManager → Attention → HostStaging → GPU0 原生 gate/o_proj。

Runtime 只保留 CacheInfo，协议里没有 GPU1 Tensor。更换 PCIe Backend 时保持 execute/Completion 契约，设备内存和驱动实现不进入模型代码。GPUEmulator 的 manager 和 transport 仅由设备级测试检查；模型与 Runtime 都不访问它们。

Runtime 对 Backend Completion 校验 command_id、request/layer、handle、dtype/layout、容量和追加后长度。错误响应不会覆盖 Runtime 已知状态。模型加载前通过 preflight 读取 checkpoint header，生成仅 GPU0/CPU 的显式 device_map；此预检不属于 Runtime，真实 ASIC Backend 无需依赖模型权重布局。

协议 v1 使用 BF16、BHTD、batch=1、24 Q heads、4 KV heads、256 head dim。Decode 仅接受单 token。ALLOC/INIT/APPEND/INFO/FREE 均带 command_id、request_id、layer_id、handle；APPEND 还带 expected_length 和 token_position。完整描述见 src/pnm/protocol/types.py。

所有命令串行执行。命令 ID 在 session 内去重，重复命令返回 DUPLICATE_COMMAND，不重放输出也不再次追加。重试应先查询长度，再用新 ID；已成功的 token 即使用新 ID 重投，也会被长度/位置检查拒绝。失败的 append 在全部计算和返回拷贝完成前不提交长度；未提交的槽位可被覆盖。session 默认最多 100000 个命令，达到预算后拒绝，避免无限保留去重状态。释放 Runtime 后清空记录。

Pinned slots 在 backend 创建时预分配，容量固定且复用；K/V 槽支持一次完整 Prefill 初始化。单 token Q/O 槽，mask 独立槽。每次拷贝使用专用 stream、CUDA event 计时和同步。D2H 完成后才发 H2D，H2D 完成后才允许复用 host slot。同步 API 使用 non_blocking copy，但本版不宣称流水并行。输出 clone 交付调用方独立所有权。计时不含所有 CPU 调度成本；command wall time 另行记录。

非默认 source stream 通过 wait_stream 对齐，跨设备 host bridge 通过 event.synchronize 明确建立顺序；同设备 consumer 通过 wait_event 对齐。未使用跨 GPU Tensor.to。显式 mask 的浮点格式在传输层用 FP32，kernel 根据本机 cuDNN 要求转换为 Q dtype；转换行为在代码与失败日志中可查。普通无 padding 单 token Decode 不传 mask，每步逻辑 QKV 输入 16384 bytes，O 输出 12288 bytes；四段拷贝总量 57344 bytes，不等同 PCIe 实际总线流量。

Qwen 集成使用 Transformers 5.19.0 当前 Qwen3_5Attention 的可注册 attention interface。为 Layer 3 单独复制 config，注册实例专属接口，退出时恢复原始 config 并注销。原 forward 未复制，环境源码未修改。接口收到经过 Q/K Norm、partial RoPE 和 native cache.update 的 Q/K/V，只切取最后一个 token 的 K/V 传给 Runtime。返回 BTHD O 由原 forward 执行 sigmoid gate 和 o_proj。local 完全调用原 Attention，shadow_compare 返回本地 O，shadow_remote 返回 GPU1 O。

P3 保留 GPU0 Layer 3 完整历史和 GPU1 独立副本。P4 需在 cache.update / cache metadata 边界另加桥接，使全局 get_seq_length（当前会选择首个 Full Attention Layer 3）仍正确；Runtime、命令、传输和设备缓存无需重构。本版不声称 Strict Remote KV。

运行证据使用 schema v2：P0/P1/P2/pytest/P3 各自记录执行时源码哈希、Git commit 和 run_id。验收脚本是普通 Python 显式条件检查，不依赖可能被 `python -O` 移除的 assert；它从每个命令的四段 copy、kernel 和 completion 重建缓存状态。报告生成器只读同一次输出目录，严格部署状态从实际 device_map 和 GPU1 进程探测推导。早期报告没有这些字段，作为历史基线保留。
