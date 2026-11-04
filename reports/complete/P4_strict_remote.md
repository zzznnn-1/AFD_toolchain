# P4 Strict Remote KV：PASS

CacheBridge 安装后原生 Layer 3 keys/values 清空，旧 Prefill K/V weak references 失效；bridge 只保存 metadata。每步对象检查 local_kv_bytes=0，目标层接口只收到当前 K/V `[1,4,1,256]`。没有调用本地历史 Attention，也没有每步重传历史。GPU1 成功返回后才提交全局/Layer 3 长度，其他 63 层保持原生执行。

| 运行 | Prompt | Decode | 使用 GPU1 O | GPU0 目标 KV bytes | 最终长度 |
|---|---|---|---|---|---|
| L4-strict | 4 | 128 | 128 | 0 | 132 |
| L4-strict-greedy | 4 | 16 | 16 | 0 | 20 |
| L16-strict | 16 | 128 | 128 | 0 | 144 |
| L128-strict | 128 | 128 | 128 | 0 | 256 |
| L128-strict-greedy | 128 | 128 | 128 | 0 | 256 |
| L1024-strict | 1024 | 128 | 128 | 0 | 1152 |
| L4096-strict | 4096 | 128 | 128 | 0 | 4224 |
| L8192-strict | 8192 | 128 | 128 | 0 | 8320 |

固定 Token 序列四级 Max absolute error（完整四项误差与 cosine 保存于 JSON）：

| Context | 状态 | 原始 O | Attention 输出 | Decoder 输出 | Logits |
|---|---|---|---|---|---|
| 4 | PASS | 0 | 0 | 0 | 0 |
| 16 | PASS | 0 | 0 | 0 | 0 |
| 128 | PASS | 0 | 0 | 0 | 0 |
| 1024 | PASS | 0 | 0 | 0 | 0 |
| 4096 | PASS | 0 | 0 | 0 | 0 |
| 8192 | PASS | 0 | 0 | 0 | 0 |

远端 O 因果干预：`[{'status': 'PASS', 'kind': 'zero actual backend O after GPU1 computation', 'logit_delta': {'max_abs': 9.28125, 'mean_abs': 0.8848248720169067, 'cosine_similarity': 0.9592531323432922, 'max_relative': 1959.95654296875, 'rms': 1.126604437828064, 'relative_floor': 0.001, 'reference_abs_max': 23.375, 'finite': True}, 'remote_outputs_used': 2}]`。该实验在 GPU1 正常计算后故意把返回 O 置零，要求最终 logits 改变；它是依赖路径验证，不作为正确输出样本。

完整模型故障实验：`[{'case': 'device_reset', 'status': 'PASS', 'observed_error': 'STALE_HANDLE', 'completed_tokens_before_fault': 1, 'ownership_before_fault': {'class': 'RemoteCacheLayer', 'sequence_length': 17, 'capacity': 256, 'local_kv_bytes': 0, 'keys_is_none': True, 'values_is_none': True, 'pending': False, 'poisoned': False, 'prefill_kv_released': True, 'retired_prefill_kv_bytes': 65536}, 'device_attention_calls': 1, 'adapter_restored': True, 'returned_model_result': False}, {'case': 'device_lost', 'status': 'PASS', 'observed_error': 'DEVICE_LOST', 'completed_tokens_before_fault': 1, 'ownership_before_fault': {'class': 'RemoteCacheLayer', 'sequence_length': 17, 'capacity': 256, 'local_kv_bytes': 0, 'keys_is_none': True, 'values_is_none': True, 'pending': False, 'poisoned': False, 'prefill_kv_released': True, 'retired_prefill_kv_bytes': 65536}, 'device_attention_calls': 1, 'adapter_restored': True, 'returned_model_result': False}, {'case': 'cancel', 'status': 'PASS', 'observed_error': 'CANCELLED', 'completed_tokens_before_fault': 1, 'ownership_before_fault': {'class': 'RemoteCacheLayer', 'sequence_length': 17, 'capacity': 256, 'local_kv_bytes': 0, 'keys_is_none': True, 'values_is_none': True, 'pending': False, 'poisoned': False, 'prefill_kv_released': True, 'retired_prefill_kv_bytes': 65536}, 'device_attention_calls': 1, 'adapter_restored': True, 'returned_model_result': False}, {'case': 'timeout_before', 'status': 'PASS', 'observed_error': 'TIMEOUT_UNKNOWN', 'completed_tokens_before_fault': 1, 'ownership_before_fault': {'class': 'RemoteCacheLayer', 'sequence_length': 17, 'capacity': 256, 'local_kv_bytes': 0, 'keys_is_none': True, 'values_is_none': True, 'pending': False, 'poisoned': False, 'prefill_kv_released': True, 'retired_prefill_kv_bytes': 65536}, 'device_attention_calls': 1, 'adapter_restored': True, 'returned_model_result': False}, {'case': 'timeout_after', 'status': 'PASS', 'observed_error': 'TIMEOUT_UNKNOWN', 'completed_tokens_before_fault': 1, 'ownership_before_fault': {'class': 'RemoteCacheLayer', 'sequence_length': 17, 'capacity': 256, 'local_kv_bytes': 0, 'keys_is_none': True, 'values_is_none': True, 'pending': False, 'poisoned': False, 'prefill_kv_released': True, 'retired_prefill_kv_bytes': 65536}, 'device_attention_calls': 2, 'adapter_restored': True, 'returned_model_result': False}, {'case': 'duplicate_completion', 'status': 'PASS', 'observed_error': 'INVALID_COMPLETION', 'completed_tokens_before_fault': 1, 'ownership_before_fault': {'class': 'RemoteCacheLayer', 'sequence_length': 17, 'capacity': 256, 'local_kv_bytes': 0, 'keys_is_none': True, 'values_is_none': True, 'pending': False, 'poisoned': False, 'prefill_kv_released': True, 'retired_prefill_kv_bytes': 65536}, 'device_attention_calls': 2, 'adapter_restored': True, 'returned_model_result': False}, {'case': 'transfer_error', 'status': 'PASS', 'observed_error': 'TRANSFER_ERROR', 'completed_tokens_before_fault': 1, 'ownership_before_fault': {'class': 'RemoteCacheLayer', 'sequence_length': 17, 'capacity': 256, 'local_kv_bytes': 0, 'keys_is_none': True, 'values_is_none': True, 'pending': False, 'poisoned': False, 'prefill_kv_released': True, 'retired_prefill_kv_bytes': 65536}, 'device_attention_calls': 1, 'adapter_restored': True, 'returned_model_result': False}]`。发生异常后不返回模型结果，Adapter 恢复，旧请求失效；不能用本地回退掩盖故障。

Prefill 仍原生构建临时 GPU0 KV，Strict 约束作用于迁移后的 Decode。PyTorch reserved/峰值可能包含 Prefill，因此不用“reserved 立刻下降”证明所有权。实际释放由弱引用、CacheBridge 对象及每步长度检查证明。

证据：[P4_suite.json](raw/P4_suite.json)、[GPU1/传输记录](raw/P4_suite.jsonl)、[运行日志](raw/P4_suite.log)、[GPU 采样](raw/P4_suite_gpu.jsonl)。

