# P0：PASS

环境时间 2026-11-04T14:26:12.121807+08:00，Python 3.10.12，PyTorch CUDA 13.0。
GPU：`[{'index': 0, 'name': 'NVIDIA GeForce RTX 4090', 'total_bytes': 25250627584, 'capability': [8, 9], 'free_bytes': 24821956608}, {'index': 1, 'name': 'NVIDIA GeForce RTX 4090', 'total_bytes': 25250627584, 'capability': [8, 9], 'free_bytes': 13009420288}]`；Peer Access：`{'0->1': False, '1->0': False}`。
软件版本：`{"torch": "2.14.1", "transformers": "5.19.0", "accelerate": "1.15.0", "huggingface_hub": "2.2.0", "safetensors": "0.8.0", "kernels": "0.17.0", "triton": "3.8.0", "pytest": "9.2.0", "flash-linear-attention": "0.5.2"}`。

完整模型 `Qwen/Qwen3.8-27B-FP8`，Revision `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`，400 个 FP8Linear 检查结果 `{'missing_keys': [], 'unexpected_keys': [], 'mismatched_keys': [], 'error_msgs': []}`。FP8 probe=PASS，激活输出 `torch.bfloat16`。

预检选择 45 个 decoder 层常驻 GPU0，CPU 参数字节 13242690784；GPU1 既有计算进程 `[{'gpu_uuid': 'GPU-ebe16217-2373-4202-8599-dd10af2b39bd', 'pid': 1189119, 'name': 'VLLM::EngineCore', 'used_mib': 11260}]`。模型权重不进入 GPU1。物理独占/全量 GPU0 常驻状态 BLOCKED，与 P4 的 KV 唯一所有权区分。

原始记录：[environment.json](raw/environment.json)、[FP8 probe](raw/fp8_probe.json)、[模型完整元数据](raw/P4_suite.json)、[版本锁](raw/environment.lock.txt)。topology、CPU/NUMA/RAM/磁盘和驱动/nvcc 原始命令在 environment.json。

