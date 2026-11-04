# 当前模型实现调查（本机实际验证）

官方模型 `Qwen/Qwen3.8-27B-FP8`，Transformers 5.19.0。根类 `Qwen3_5ForConditionalGeneration`，文本主体 `model.language_model`，64 个 decoder layers，Full Attention 在 3,7,...,63。

当前 Qwen3_5Attention.forward 保持原样；本工程注册接口调用发生于 native cache.update 后、sigmoid gate/o_proj 前。Q 的 q_proj 同时产生 gate，输出 12288 维；Q 与 gate 每 head 分开。q_proj/k_proj/v_proj/o_proj 为 FP8Linear，Q/K Norm 为 Qwen3_5RMSNorm（zero-centered weight），partial RoPE 仅 64 维，GQA 24:4。

第一次完整模型成功运行所观测的 Prefill Q/K/V：

```json
{
  "q": {
    "shape": [
      1,
      24,
      30,
      256
    ],
    "dtype": "torch.bfloat16",
    "device": "cuda:0"
  },
  "k": {
    "shape": [
      1,
      4,
      30,
      256
    ],
    "dtype": "torch.bfloat16",
    "device": "cuda:0"
  },
  "v": {
    "shape": [
      1,
      4,
      30,
      256
    ],
    "dtype": "torch.bfloat16",
    "device": "cuda:0"
  }
}
```

完整模型 Decode 接口的 Q 与原生累计 K/V：

```json
{
  "q": {
    "shape": [
      1,
      24,
      1,
      256
    ],
    "dtype": "torch.bfloat16",
    "device": "cuda:0"
  },
  "k": {
    "shape": [
      1,
      4,
      62,
      256
    ],
    "dtype": "torch.bfloat16",
    "device": "cuda:0"
  },
  "v": {
    "shape": [
      1,
      4,
      62,
      256
    ],
    "dtype": "torch.bfloat16",
    "device": "cuda:0"
  }
}
```

这里 K/V shape 是 native cache.update 后累计历史；adapter 只取 `[:, :, -1:, :]` 传输，所以设备每步收到 K_new/V_new `[1,4,1,256]`。JSONL 中每步 K/V 各 2048 bytes，Q 12288 bytes，返回 O 12288 bytes。

Cache API：`DynamicCache.layers[3].keys/values`，并用 `get_seq_length(3)` 查询。默认 `get_seq_length()` 自动跳过 LinearAttentionLayer，找到第一个 Full Attention，正是 Layer 3。当前源码 forward 没有显式 cache_position 参数；position_ids 由全局长度生成，runner 单 Token Decode 显式传入旧长度，adapter 与远端 KV 长度交叉校验。当前例子中的位置记录：

```json
{
  "position": 30,
  "native_length": 31,
  "remote_length": 31,
  "position_ids": [
    30
  ],
  "cache_position": null,
  "mask": null
}
```

Native attention gate 为 `reshape -> sigmoid(gate) multiply -> o_proj`，以本机源码为准。完整 Decoder 在此后还有残差、post_attention_layernorm、MLP、第二个残差。四级比较分别在核心 O、Attention 返回值、Decoder 返回值、最终 logits 采样，避免只比较单独重复计算而没有接入后续模型。

量化匹配：官方排除列表 `mlp.gate` 与当前 regex prefix 实现冲突，本项目只修正加载配置歧义 leaf 的匹配边界，保留官方 FP8 数值和 block scales；实际检查 400 个 FP8Linear，并要求加载诊断所有 mismatch 集合为空。Projection 元数据及完整 device_map 在 `reports/raw/P3_suite.json`。

Source SHA256 在 `reports/raw/environment.json`；FP8 kernel revision 在 `reports/raw/fp8_kernel_revision.json`。没有修改 Transformers 安装包文件。
