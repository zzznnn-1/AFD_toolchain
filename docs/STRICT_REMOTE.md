# P4 Strict Remote KV

`strict_remote` 与 `shadow_remote` 的区别是缓存所有权。原生 Prefill 完成并将目标层 K/V 初始化到 GPU1 后，CacheBridge 替换 `DynamicCache.layers[3]`，清除原生层的 `keys/values`。它只保存长度、容量、Handle、待完成标志和失效原因，不保存任何 Tensor。Weak reference 检查保证原始 Prefill K/V 对象已释放，而不是被 Adapter 的另一字段继续引用。

当前 Transformers 的 `Cache.get_seq_length()` 会把首个 Full Attention 层作为全局长度来源。因此桥接类继承 `CacheLayerMixin`，提供 `get_seq_length()`、`get_mask_sizes()`、`get_max_length()`；其他 63 层的 Cache 对象继续原生维护。

Decode 仍运行当前版本原生 Qwen Attention forward：GPU0 的 Q/K/V Projection、Norm、RoPE 不变。它调用 Bridge.update 时只收到当前 K/V，Bridge 返回它们但不存储，也不增加已提交长度。Attention interface 仅调用远端 Runtime，用完成结果提交长度后返回 O；完全不计算一份本地历史 Attention。随后原生 gate/o_proj 和 decoder 继续执行。

如果远端计算、通信、长度、generation 或 completion 发生错误，Bridge 被 poison，该请求不可继续；不会回退到本地。前面 Linear Attention 层可能已处理了当前 Token，因此即使通过查询确认远端长度，也不能把整个模型请求当作可重试的事务。应释放并重新 Prefill 新请求。

模型 API 当前只支持 batch=1、单 Token Decode。Beam search、重排、crop、缓存 offload/reset 明确拒绝，避免用 Tensor 操作接口虚构远端行为。新请求通过重新构建原生 DynamicCache 开始。

数据证据包括 Prefill/Decode 的 ownership 快照、每步 local_kv_bytes=0、旧 Prefill K/V weak references 已失效、全局和全部 Full Attention 层长度、运行时设备、GPU1 KV 字节和独立固定 Token 的四级数值比较。allocated 与 reserved 分开记录；allocator reserved 不下降不代表历史 KV 仍存活。

P4 不要求全量模型权重常驻 GPU0。CPU 仍可存放并向 GPU0 搬运官方 FP8 模型权重；这与 Layer 3 历史 KV 的 GPU1 唯一所有权是两个不同条件。物理 GPU 独占另行记录。
