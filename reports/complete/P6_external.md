# P6：BLOCKED

真实硬件检查时间 2026-11-04T14:26:49.115291+08:00；候选节点 `{'pnm': [], 'kxp': [], 'xdma': []}`；项目环境包 `{'vllm': False, 'pnm_driver': False, 'kxp_driver': False}`。

已经实现并测试 Backend 插件工厂和带版本/长度/CRC 的命令 envelope，Runtime 与模型不访问 GPU1 内部 KV Tensor。缺少真实驱动插件时明确失败，不自动回退。当前 envelope 不是冻结的硬件 DMA ABI；CUDA pinned buffer 不是已完成驱动注册的内存。

实际阻塞：['No project ASIC device/driver adapter and no finalized DMA/ioctl ABI were supplied.', 'PCIe/CXL type, registered-memory ownership, queue/reset/completion rules need hardware confirmation.', 'vLLM integration is explicitly deferred in this prototype specification; no vLLM backend was run.']。
所需输入：['PCI vendor/device IDs and device node', 'driver headers/library and version', 'DMA buffer registration/IOMMU API', 'command queue and completion ABI', 'device reset/generation semantics', 'KXP supported layout/dtype/mask/capacity']。

原规范明确后置 vLLM；本次仍采用 Transformers，未把服务器其他 vLLM 进程视为本项目集成。真实 ASIC/驱动及 vLLM PNM 接入没有实际运行，不记录 PASS。

原始检测：[P6_external.json](raw/P6_external.json)。接口契约见项目 docs/DRIVER_CONTRACT.md。

