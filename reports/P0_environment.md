# P0 环境调查

采集时间：2026-11-04T09:09:16.908296+08:00。硬件、拓扑、版本调查 **PASS**；完整模型的实际 dtype/结构见 P3 运行数据（没有 P3 成功日志时，该项不能算通过）。

| 验收项 | 状态 | 实测 |
|---|---|---|
| 双 GPU/CC | PASS | 2×RTX 4090，CC 8.9；每卡 CUDA 可见 23.516 GiB |
| Driver/CUDA | PASS | Driver 580.178.04；PyTorch CUDA 13.0；系统 nvcc 11.5，实际 FP8 使用 Triton 内核 |
| P2P/拓扑 | PASS | 双向 peer=false；topo SYS、p2p NS，GPU0 NUMA0 / GPU1 NUMA1 |
| RAM/磁盘 | PASS | 251 GiB RAM；检查时可用磁盘 657.4 GiB |
| FP8 kernel | PASS | 官方 FP8Linear 在 GPU0 实际执行；FP8 权重和 FP32 scale，BF16 输出 |
| 权重全部常驻 GPU0 | BLOCKED | 官方全部文件约 28.77 GiB，超过单卡；采用显式 CPU 权重存储 + GPU0 计算 |
| GPU1 模型权重隔离 | PASS | 加载器显式 device map，断言本模型无 GPU1 参数；服务器其他进程单独记录 |

模型：`Qwen/Qwen3.8-27B-FP8`，Revision `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`。
版本：`{"torch": "2.14.1", "transformers": "5.19.0", "accelerate": "1.15.0", "huggingface_hub": "2.2.0", "safetensors": "0.8.0", "kernels": "0.17.0", "triton": "3.8.0", "pytest": "9.2.0", "flash-linear-attention": "0.5.2"}`。完整锁定清单：`raw/environment.lock.txt`。

官方配置来自 https://huggingface.co/Qwen/Qwen3.8-27B-FP8/blob/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a/config.json 。官方模型卡说明采用 128-block fine-grained FP8：https://huggingface.co/Qwen/Qwen3.8-27B-FP8 。本项目以下载的 config 和本机 Transformers 源码为执行依据。

`Qwen3_5ForConditionalGeneration.model.language_model.layers` 共 64 层。Full Attention 为 3,7,...,63；24 Q heads / 4 KV heads / head dim 256 / GQA 6:1 / partial RoPE 64 维。当前 forward 的顺序：q_proj 同时生成 Q 和 gate → Q/K RMSNorm → RoPE → cache.update → attention interface → reshape → sigmoid(gate) 相乘 → o_proj。不能根据 config 的其他 gate 字段猜测执行公式。

Cache 是 `DynamicCache`，目标层 `cache.layers[3].keys/values`。默认 `get_seq_length()` 遇到第 0 层 LinearAttentionLayer 时自动找首个 Full Attention，即 Layer 3；指定 `get_seq_length(3)` 直接读取其长度。当前 text model forward 不显式接收 `cache_position`，位置由 cache 长度或传入 position_ids 得到；本 runner 显式提供每步 position_ids，并校验所有 16 个 Full Attention 层长度。源码副本和 SHA256：`raw/transformers_attention_source.py.txt`、`raw/transformers_cache_source.py.txt`、`raw/environment.json`。

内存方案：常驻前 48 个 decoder layers，后 16 层、embedding/lm_head 和未用于文本的视觉模块保存在 CPU，Accelerate 需要时加载到 GPU0 执行。没有重新量化，没有把本模型权重放入 GPU1。GPU1 已有其他服务约 11 GiB 占用，未终止该进程；本报告不声称物理服务器独占。

修复记录：FP8Linear 探测最初用了过期构造参数，已按当前源码修正；自动安装的 kernels 0.18.1 不兼容 Transformers 5.19.0，改锁 kernels 0.17.0 后真实 CUDA kernel 通过。旧失败日志保留 `raw/fp8_kernel_probe_*failure.log`。

完整加载又发现官方配置 `mlp.gate` 在当前量化器中被当作正则前缀，误排除 `mlp.gate_proj`，造成 64 个 FP8 scale 未加载，常驻权重增长，首次 Prefill OOM。工程加载器依据 checkpoint scale keys 对 65 个歧义 leaf 名称（含未用于本轮的 MTP）添加结束锚点，没有改权重、scale 或环境源码。修复后检查全部 400 个 decoder FP8Linear，并拒绝任何 missing/unexpected/mismatched keys。失败证据 `raw/P3_smoke_fp8_exclusion_failure.log/json`，回归测试 `tests/test_loading.py`。首次报告序列化遇到 loading_info 中的 set 类型，亦已修复并保留 `raw/P3_smoke_report_type_failure.log`。

复现：`.venv/bin/python scripts/check_environment.py`、`.venv/bin/python scripts/probe_fp8.py`、`.venv/bin/python scripts/download_model.py`。原始日志 `raw/P0_environment.log`、`raw/fp8_kernel_probe.log`、`raw/download.log`。

