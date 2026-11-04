# P5 性能、压力和故障：PASS

方法：Linear-interpolated sample percentiles; first 8 decode tokens excluded; 16/128 windows are prefixes of the SAME 128-token run, not independent repetitions. Device event intervals are not additive end-to-end latency. CPU weight offload traffic is excluded.。表中单位 ms，CPU 权重 offload 会主导 token 延迟。native Prefill 计算与 KV 初始化分别计时；所有首次编译/预热影响如实保留，不宣称模拟器或 ASIC 加速比。

| Context | 模式 | 输出窗口 | 统计样本 | p50 | p95 | p99 | tokens/s |
|---|---|---|---|---|---|---|---|
| 4 | local | 16 | 8 | 881.558 | 953.461 | 954.920 | 1.111 |
| 4 | local | 128 | 120 | 896.146 | 955.661 | 957.507 | 1.093 |
| 4 | strict_remote | 16 | 8 | 951.100 | 957.714 | 959.164 | 1.067 |
| 4 | strict_remote | 128 | 120 | 907.911 | 958.113 | 963.476 | 1.090 |
| 4 | strict_remote | 16 | 8 | 902.931 | 905.835 | 906.603 | 1.108 |
| 16 | local | 16 | 8 | 961.066 | 966.617 | 967.102 | 1.057 |
| 16 | local | 128 | 120 | 922.925 | 967.256 | 972.372 | 1.079 |
| 16 | strict_remote | 16 | 8 | 960.611 | 965.802 | 966.269 | 1.048 |
| 16 | strict_remote | 128 | 120 | 933.005 | 970.533 | 973.966 | 1.073 |
| 128 | local | 16 | 8 | 966.019 | 969.343 | 969.545 | 1.053 |
| 128 | local | 128 | 120 | 956.746 | 968.881 | 970.237 | 1.070 |
| 128 | strict_remote | 16 | 8 | 961.502 | 971.499 | 972.104 | 1.053 |
| 128 | strict_remote | 128 | 120 | 933.647 | 968.513 | 971.845 | 1.074 |
| 128 | strict_remote | 16 | 8 | 959.827 | 970.985 | 971.235 | 1.054 |
| 128 | strict_remote | 128 | 120 | 907.077 | 974.572 | 976.868 | 1.078 |
| 1024 | local | 16 | 8 | 893.670 | 894.928 | 894.992 | 1.121 |
| 1024 | local | 128 | 120 | 894.604 | 967.773 | 969.848 | 1.090 |
| 1024 | strict_remote | 16 | 8 | 955.378 | 964.187 | 964.480 | 1.062 |
| 1024 | strict_remote | 128 | 120 | 899.981 | 968.694 | 972.708 | 1.084 |
| 4096 | local | 16 | 8 | 900.911 | 965.215 | 966.283 | 1.086 |
| 4096 | local | 128 | 120 | 899.021 | 967.467 | 971.568 | 1.084 |
| 4096 | strict_remote | 16 | 8 | 895.941 | 924.005 | 934.397 | 1.110 |
| 4096 | strict_remote | 128 | 120 | 899.514 | 970.520 | 973.982 | 1.086 |
| 8192 | local | 16 | 8 | 896.821 | 962.526 | 964.059 | 1.096 |
| 8192 | local | 128 | 120 | 899.157 | 966.815 | 968.689 | 1.087 |
| 8192 | strict_remote | 16 | 8 | 967.503 | 972.664 | 973.107 | 1.051 |
| 8192 | strict_remote | 128 | 120 | 904.224 | 970.682 | 972.224 | 1.077 |

每步包括 QKV 准备 CUDA 时间、runtime 命令准备、队列、四段 copy、GPU1 KV append、Attention、端到端 offload、Layer 3 和完整 token 时间。不同 GPU 时钟事件不能直接相加为真实端到端延迟。四段 copy 每 token 为 57344 bytes，命令描述字节单列，不包括 CPU 权重传输，也不等于 PCIe 总线线速。

压力：16 个请求 × 128 步，Prefill=4096，所有输出对照独立 FP32 参考。固定 pool 未扩容；warm 后 GPU1 allocated 变化守卫 1 MiB，Host RSS 变化守卫 32 MiB；这些是本次有限压力运行的观测范围。关闭后 GPU1 allocated=0 bytes。去重 ID 集合增长是有预算上限的 metadata，不是 KV 泄漏。

Host pool/NUMA：`{'backend': 'gpu_emulator', 'protocol_version': 1, 'generation': 1, 'online': True, 'kv_caches': 0, 'kv_allocated_bytes': 0, 'transport': 'host_staging', 'pinned_bytes': 1138688, 'all_pinned': True, 'chunk_tokens': 256, 'preallocated_slots': 5, 'transfer_calls': 3894, 'pool_resizes': 0, 'host_numa_vmas': {'q': {'vma': '76a674600000 default file=/dev/zero\\040(deleted) dirty=512 active=0 N1=512 kernelpagesize_kB=4', 'scope': 'whole containing VMA, not exact buffer page placement'}, 'k': {'vma': '76a674600000 default file=/dev/zero\\040(deleted) dirty=512 active=0 N1=512 kernelpagesize_kB=4', 'scope': 'whole containing VMA, not exact buffer page placement'}, 'v': {'vma': '76a674600000 default file=/dev/zero\\040(deleted) dirty=512 active=0 N1=512 kernelpagesize_kB=4', 'scope': 'whole containing VMA, not exact buffer page placement'}, 'o': {'vma': '76a674600000 default file=/dev/zero\\040(deleted) dirty=512 active=0 N1=512 kernelpagesize_kB=4', 'scope': 'whole containing VMA, not exact buffer page placement'}, 'mask': {'vma': '76a674600000 default file=/dev/zero\\040(deleted) dirty=512 active=0 N1=512 kernelpagesize_kB=4', 'scope': 'whole containing VMA, not exact buffer page placement'}}}`。NUMA 数据是 buffer 所在 VMA 的页面分布，不能当作每个 buffer 的精确逐页 NUMA 归属；进程 CPU affinity 在主机上下文记录。没有修改系统 NUMA 策略。

交付：[JSON](raw/P5_performance.json)、[CSV](raw/P5_performance.csv)、[压力结果](raw/P5_stress.json)、[压力日志](raw/P5_stress.log)、[完整模型故障](raw/P4_suite.json)。所有数值均为 GPU 软件模拟，不代表未来 KXP/ASIC 吞吐。

