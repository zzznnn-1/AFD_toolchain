# P6 外部接入契约与未就绪条件

当前真实 ASIC、项目 PCIe 驱动和 DMA ABI 未提供。代码提供可执行的 Backend 工厂和命令 envelope，用于替换 GPU 模拟器；没有虚构 ioctl 编号、BAR 地址或驱动实现，也没有把普通 CUDA pinned 内存称为已完成 IOMMU/DMA 注册。

`pnm.backends.registry.create_backend` 支持明确配置的 `module:factory`。工厂返回实现 `Backend.execute(command, payload)`、`describe()`、`close()` 的对象。返回对象必须声明协议版本 1；模块缺失或配置错误立即报 DRIVER_UNAVAILABLE，不能偷偷使用 GPU 模拟器。

Runtime 只认 Command、Completion 和 CacheInfo，不访问 GPU1 Tensor。真实适配器内部负责把 GPU0 输入经受支持的方式送入驱动注册 Host buffer，再提交设备命令和等待 completion，并把 O 返回 GPU0。驱动自己的 fd、映射和 DMA 地址不进入 Qwen Adapter。

当前 `protocol/codec.py` 提供带长度、版本、CRC32 的 simulator envelope；里面是 request/layer/command ID、generation、handle、位置、dtype/layout 及 shape/stride/bytes 元数据。它已用于实际 Runtime 描述符计数和往返/损坏测试，但 **不是已经与 ASIC 约定的二进制 ABI**，不包含真实 DMA 地址。

真实 Backend 必须满足：

- INIT 逻辑一次，物理传输可分块，未完成前不暴露有效长度。
- APPEND 成功才提交长度；同一 Command ID 不重复执行，同一 Token 不能通过新 ID 绕过位置校验。
- Timeout Unknown 不表示失败回滚。请求必须 poison 或查询确切 command outcome；没有确认前禁止再次追加。
- Reset/device lost 使之前 generation 的所有 Handle 失效。旧 completion 不可被新请求接受。
- 返回的 O 拥有独立的有效生命周期；底层 DMA buffer 完成前不能复用。释放/取消必须定义在途 DMA 的 drain/cancel 规则。
- 传输失败与设备失败需要明确区分“确定未提交”和“可能已提交”，不能通过本地 Attention 回退隐藏错误。

真实硬件接入仍需要：PCI vendor/device IDs 与设备节点、驱动头文件/库和版本、页锁定/注册/注销接口、IOMMU 映射规则、队列/doorbell/completion ABI、reset 语义、KXP dtype/layout/GQA/mask/容量能力，以及 PCIe Endpoint 或 CXL Type-2 的设备类型确认。

`scripts/check_external.py` 会记录服务器实际 PCI 列表、候选设备节点和 Python 依赖，并列出缺少的输入。资源缺失时 P6 标为 BLOCKED；软件 envelope 和插件工厂测试成功不等于真实硬件通过。

vLLM 在规范中被明确列为后续集成，本轮继续使用 Transformers。后续需要锁定 vLLM 版本并适配其 Attention Backend、KV Cache Manager 和 Scheduler；KV Transfer Connector 本身不能替代远端 Attention 计算。Strict Remote 的 batch=1/no-crop 契约不能无声扩展为 vLLM paged cache 或多请求 eviction。没有运行过 vLLM PNM 接入就不记录 PASS。
