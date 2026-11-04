# PNM v1 / P0–P5 与 P6 接入契约

模型适配器 → Runtime → Backend.execute(Command, payload) → HostStaging → GPU1 CacheManager → Attention → HostStaging → GPU0 原生 gate/o_proj。

Runtime 只保留 CacheInfo，协议里没有 GPU1 Tensor。更换 PCIe Backend 时保持 execute/Completion 契约，设备内存和驱动实现不进入模型代码。GPUEmulator 的 manager 和 transport 仅由设备级测试检查；模型与 Runtime 都不访问它们。

Runtime 对 Backend Completion 校验 command_id、request/layer、handle、dtype/layout、容量和追加后长度。错误响应不会覆盖 Runtime 已知状态。模型加载前通过 preflight 读取 checkpoint header，生成仅 GPU0/CPU 的显式 device_map；此预检不属于 Runtime，真实 ASIC Backend 无需依赖模型权重布局。

协议 v1 使用 BF16、BHTD、batch=1、24 Q heads、4 KV heads、256 head dim。Decode 仅接受单 token。ALLOC/INIT/APPEND/INFO/FREE 均带 command_id、request_id、layer_id、handle；APPEND 还带 expected_length 和 token_position。完整描述见 src/pnm/protocol/types.py。

所有命令串行执行。命令 ID 在 session 内去重，重复命令返回 DUPLICATE_COMMAND，不重放输出也不再次追加。确定失败且未提交的命令可先查询长度，再用新 ID；TIMEOUT_UNKNOWN/错误 completion 会 poison Handle，查询不自动解除，模型请求应释放并重新 Prefill。已成功 Token 即使用新 ID 重投，也会被长度/位置检查拒绝。append 在计算和返回拷贝完成前不提交长度；未提交槽位可覆盖。session 最多 100000 个命令，达到预算后拒绝，关闭 Runtime 后清空记录。

Pinned slots 创建时预分配并复用；K/V 槽默认只容纳 256 Tokens，完整 Prefill 在一个 INIT 命令内分块初始化。每个 chunk 的 D2H、H2D 和 KV 写入完成后才复用相应 host/device scratch。Q/O 单 token，mask 独立槽。non_blocking copy 配合专用 streams/events，但不宣称流水重叠。返回 O clone 给调用方独立所有权。

非默认 source stream 通过 wait_stream 对齐，跨设备 host bridge 通过 event.synchronize 明确建立顺序；同设备 consumer 通过 wait_event 对齐。未使用跨 GPU Tensor.to。显式 mask 的浮点格式在传输层用 FP32，kernel 根据本机 cuDNN 要求转换为 Q dtype；转换行为在代码与失败日志中可查。普通无 padding 单 token Decode 不传 mask，每步逻辑 QKV 输入 16384 bytes，O 输出 12288 bytes；四段拷贝总量 57344 bytes，不等同 PCIe 实际总线流量。

Qwen 集成使用 Transformers 5.19.0 当前 Qwen3_5Attention 的可注册 attention interface。为 Layer 3 单独复制 config，注册实例专属接口，退出时恢复原始 config 并注销。原 forward 未复制，环境源码未修改。接口收到经过 Q/K Norm、partial RoPE 和 native cache.update 的 Q/K/V，只切取最后一个 token 的 K/V 传给 Runtime。返回 BTHD O 由原 forward 执行 sigmoid gate 和 o_proj。local 完全调用原 Attention，shadow_compare 返回本地 O，shadow_remote 返回 GPU1 O。

P3 保留本地与远端副本。P4 RemoteCacheLayer 在 cache.update/metadata 边界提供桥接：原生 update 仅返回新 K/V，已提交长度只在远端成功后增加。strict Attention interface 完全跳过本地历史 Attention，返回远端 O。全局 get_seq_length 仍从 Layer 3 读取，其他层保持原生。旧 Prefill KV 被清除并用 weak references 检查释放，Bridge 不保存 Tensor；异常会使其 poison。

Runtime 增加 generation、Timeout Unknown poison 和取消。设备 reset 使旧 Handle 失效；不确定超时/重复 completion 后不能自动解除 poison 或盲重试。取消释放资源并禁止复用 request ID。去重集合有命令预算上限。

P5 使用同设备 CUDA events 测量 QKV、KV append、Attention 和 copy；CPU perf_counter 测量命令准备/排队、offload、Layer 3 和完整 Token。不把跨 GPU 时间分项相加为真实端到端。Host NUMA 记录是所在 VMA 的页面分布，不是精确逐页 buffer 归属。

P6 create_backend 支持 module:factory，并验证 Backend/协议版本。protocol/codec.py 是带 CRC 的模拟器 envelope，不是已冻结的 DMA ABI。真实内存注册、IOMMU/队列/驱动由外部适配器提供，缺依赖时明确失败。

运行证据使用 schema v2：P0/P1/P2/pytest/P3 各自记录执行时源码哈希、Git commit 和 run_id。验收脚本是普通 Python 显式条件检查，不依赖可能被 `python -O` 移除的 assert；它从每个命令的四段 copy、kernel 和 completion 重建缓存状态。报告生成器只读同一次输出目录，严格部署状态从实际 device_map 和 GPU1 进程探测推导。早期报告没有这些字段，作为历史基线保留。
