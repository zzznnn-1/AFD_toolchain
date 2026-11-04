# P0：PASS

记录时间：2026-11-04T13:28:00.960351+08:00。硬件和版本以 [environment.json](raw/environment.json) 为准。

| GPU | 型号 | CC | 总容量 GiB | 检查时空闲 GiB |
|---|---|---|---|---|
| 0 | NVIDIA GeForce RTX 4090 | [8, 9] | 23.516 | 23.117 |
| 1 | NVIDIA GeForce RTX 4090 | [8, 9] | 23.516 | 12.116 |

软件版本：`{"torch": "2.14.1", "transformers": "5.19.0", "accelerate": "1.15.0", "huggingface_hub": "2.2.0", "safetensors": "0.8.0", "kernels": "0.17.0", "triton": "3.8.0", "pytest": "9.2.0", "flash-linear-attention": "0.5.2"}`。
PyTorch CUDA=13.0；Driver、系统 nvcc、CPU/RAM、磁盘、NUMA、topo -m/-p2p 和双向 Peer Access 的实际命令输出都在环境 JSON 内。不能把驱动支持的 CUDA 上限等同系统 nvcc。

官方模型 `Qwen/Qwen3.8-27B-FP8`，Revision `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`。FP8 kernel 实测状态 **PASS**，输出 `torch.bfloat16`；400 个 decoder FP8Linear 加载一致性诊断：`{'missing_keys': [], 'unexpected_keys': [], 'mismatched_keys': [], 'error_msgs': []}`。

模型结构为 Qwen3_5ForConditionalGeneration，64 个 decoder layers，Full Attention 在 3,7,...,63；24 Q heads / 4 KV heads / D=256。目标 Layer 3 的真实 projection 类型、参数 dtype/shape 在 [P3_suite.json](raw/P3_suite.json) 的 model.projections。

Cache 为 DynamicCache，`cache.layers[3].keys/values`；默认 get_seq_length() 从首个 Full Attention Layer 3 读取。当前源码没有显式 cache_position 参数；runner 显式传 position_ids，并校验所有 Full Attention 层长度。源码摘录与原文件哈希在环境产物中。

显存预检读取 safetensors 文件头，未加载权重：参数合计 28.302 GiB；自动选择常驻 47 层，GPU0 参数 16.684 GiB、CPU 参数 11.618 GiB、最大暂存模块 2.368 GiB、运行预留 4.000 GiB。预留只是可配置预算，不能保证任意上下文均无 OOM。

完整权重常驻是否可行：False。GPU1 既有计算进程：`[{'gpu_uuid': 'GPU-ebe16217-2373-4202-8599-dd10af2b39bd', 'pid': 1189119, 'name': 'VLLM::EngineCore', 'used_mib': 11260}]`。严格部署预检结果见 [strict_placement.json](raw/strict_placement.json)；整个模型严格模式会在分配权重前阻止不满足条件的运行。

加载器修复 Transformers 5.19.0 中官方排除项 mlp.gate 对 gate_proj 的前缀误匹配，保留官方 FP8 权重和 scales，不修改 site-packages。kernel 版本固定 kernels==0.17.0。来源：[官方固定 Revision](https://huggingface.co/Qwen/Qwen3.8-27B-FP8/tree/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a)。

复现：`.venv/bin/python scripts/check_environment.py`、`.venv/bin/python scripts/check_placement.py`、`.venv/bin/python scripts/probe_fp8.py`。

