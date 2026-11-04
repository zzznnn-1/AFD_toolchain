# 双 GPU PNM 软件模拟与自研 PCIe/ASIC 适配：Agent 工程实施规范

> 文档版本：v1.0｜整理日期：2026-11-03  
> 模型：**`Qwen/Qwen3.8-27B-FP8`**（精确模型 ID；不得改用 BF16 版本或其他第三方量化版本）  
> 受众：在双 GPU Linux 服务器上负责开发、测试和提交结果的 Coding Agent  
> 项目阶段：**无真实 ASIC，双 GPU 模拟；先正确性，后优化，最后接自研驱动**  
> 实施状态：**设计规范，尚未在目标服务器执行和验收**。所有 GPU/驱动/模型加载条件均应在 P0 实测。

---

## 0. 一页摘要：目标与硬约束

### 0.1 项目目标

本项目拟开发类似 CXL-PNM/PCIe-PNM 的软件栈：团队自行设计开发 **PCIe 设备侧协议/驱动** 和 **执行 Attention 的 ASIC（KXP）**。ASIC 实物尚未就绪，先用一台双 GPU 服务器验证推理软件的数据面、控制面、缓存所有权、协议语义和数值正确性。

现阶段分工：

- **GPU0**：运行 `Qwen/Qwen3.8-27B-FP8` 模型主体；负责目标层 Q/K/V 投影、Q/K Norm、RoPE、输出 Gate 与输出投影，以及所有未卸载层。
- **CPU Host Runtime**：管理请求、设备 cache handle、序列位置、描述符/命令、Pinned Host Buffer、同步、错误处理和指标。
- **GPU1（PNM Emulator）**：显存模拟未来设备 DRAM；GPU1 CUDA/PyTorch Attention 模拟未来 KXP；以 handle 隔离其实际缓存 Tensor。
- **传输**：首版明确走 `GPU0 VRAM -> Pinned Host RAM -> GPU1 VRAM`，返回反方向。即使当前环境有 P2P，也必须能测试该路径。

**核心验收**：Prefill 的目标层历史 KV 在 GPU1 初始化一次；Decode 每 Token 只向 GPU1 发新 Q/K/V，GPU1 持有并追加历史 KV、执行 Attention，返回 O 给 GPU0；严格模式下 GPU0 不再持有该层的完整历史 KV。

### 0.2 绝对禁止的捷径

1. **不能**让 `device_map="auto"` 把模型权重偷偷分配到 GPU1；GPU1 专用于模拟 PNM。
2. **不能**通过 `q.to("cuda:1")` 之类的隐式设备间拷贝充当 Host Staging 验收；传输必须经过明确可检查的 pinned CPU buffer。
3. **不能**把“GPU1 运行一次 Attention”算成“完成 PNM 卸载”：缓存所有权、生命周期和传输协议也要被验证。
4. **不能**把 FP8 权重模型直接理解为 FP8 KV Cache；P0 要打印真实的 Q/K/V/O dtype，默认接口使用 BF16。
5. **不能**一次性重写整套 64 层、上来就改 vLLM、修改 `site-packages/transformers` 或让基础模型和模拟器混成一个脚本。
6. **不能**因为 GPU0 容量或 FP8 Kernel 不支持而静默换用另一模型/另一 GPU 布局；必须报告阻塞并优先完成不依赖模型加载的 P1/P2。
7. **不能**在数值测试未通过时，用放宽容差、跳过测试或只展示能生成一句话作为替代。

### 0.3 当前明确不做（Non-goals）

- 不实现自研 ASIC/PCIe 驱动的真实 DMA，不声称 GPU 模拟能预测其峰值性能。
- 不实现完整 vLLM、Paged KV Cache、Continuous Batching。
- 第一轮只做 **文本、batch=1、单请求、目标层索引 3、单 Token Decode**；图像/视频、长序列、多请求、Chunked Prefill 后置。
- 不在最初阶段量化 GPU1 KV 为 FP8；不卸载 Gated DeltaNet（Linear Attention）状态。
- 不把内核级 XDMA/JDMA、BAR/MMIO、IOMMU 映射规范当成现阶段已冻结的设计。

---

## 1. 官方模型事实与 FP8 数据约定

### 1.1 官方模型仓库与版本

- 官方 ID：`Qwen/Qwen3.8-27B-FP8`
- 官方文档：https://huggingface.co/Qwen/Qwen3.8-27B-FP8
- 配置：https://huggingface.co/Qwen/Qwen3.8-27B-FP8/blob/main/config.json
- Qwen3.8 模型卡：https://huggingface.co/Qwen/Qwen3.8-27B
- Transformers 代码：https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py
- PyTorch SDPA：https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html

官方模型属于 `qwen3_5` 模型类型，顶层权重为原生多模态条件生成模型（`Qwen3_5ForConditionalGeneration`），**首轮仅测试文本输入**，不得误以为官方 FP8 检查点是纯文本 CausalLM 格式。仓库主页给出的加载方式为 `AutoProcessor` + `AutoModelForMultimodalLM`；应以 P0 锁定的 Transformers 版本实际 API 为准。

从官方 `text_config` 读出的配置（仍需在服务器启动时 assert）：

| 字段 | 值 | 影响 |
|---|---:|---|
| `num_hidden_layers` | 64 | 共 64 层 |
| `layer_types` | 每 4 层：3 Linear + 1 Full | Full Attention 层索引为 3、7、11、…、63 |
| `num_attention_heads` | 24 | Query Heads |
| `num_key_value_heads` | 4 | KV Heads；每个 KV Head 服务 6 个 Q Heads（GQA） |
| `head_dim` | 256 | 每 Head 256 个元素 |
| `partial_rotary_factor` | 0.25 | 仅 64 维参与 RoPE |
| `attn_output_gate` | true | Attention O 返回 GPU0 后还要做输出 Gate |
| `text_config.dtype` | `bfloat16` | 非量化层/主要激活路径的 dtype 配置；真实张量仍以运行时检查为准 |
| `quantization_config.quant_method` | `fp8` | 使用 FP8 量化权重 |
| `quantization_config.fmt` | `e4m3` | FP8 表示类型 |
| `quantization_config.activation_scheme` | `dynamic` | 量化计算相关配置 |
| `quantization_config.weight_block_size` | `[128,128]` | 分块量化 |

**特别提醒**：官方 FP8 检查点包含不量化模块列表 `modules_to_not_convert`。不能认为每个模块的参数、激活和返回 O 都是 FP8。推理时记录：`q_proj` / `k_proj` / `v_proj` 模块类型、参数 dtype、Q/K/V dtype、KV Cache dtype、O dtype、量化 scale 数据。不要为“使用 FP8 模型”额外对 Q/K/V 做量化。

### 1.2 第一期数据契约

默认以 **BF16 Q/K/V、BF16 远端 KV、BF16 O** 作为 Runtime 协议 v1 的明确假设；若实际张量不符合，立即报告并根据模型实际类型做边界显式转换、记录转换成本。严禁静默类型改变。K 已完成 K Norm 和 RoPE；Q 已完成 Q Norm 和 RoPE；V 为该层对应 Value 表示。RoPE 和 Q/K Norm 都由 GPU0 做，GPU1 不重复执行。

目标层 Decode 单 Token，`batch=1`，典型形状：

```text
Q      : [B=1, Hq=24, Tq=1, D=256]  BF16 => 12 KiB
K_new  : [1, 4, 1, 256]              BF16 =>  2 KiB
V_new  : [1, 4, 1, 256]              BF16 =>  2 KiB
O      : [1, 24, 1, 256]             BF16 => 12 KiB
```

- Decode 单层输入负载：`Q+K+V=16 KiB`；返回 O：`12 KiB`；单 Token 逻辑双向传输总负载：`28 KiB`。
- 显式 GPU↔CPU↔GPU 路线有四段物理拷贝，按负载计总拷贝字节约为 `56 KiB`（不计描述符、对齐和协议开销）。不能把它写成 PCIe 总线实际线速流量的精确统计。
- 单个 Full Attention 层的历史 KV：每 Token `4 KiB`；4096 Tokens = `16 MiB`，32768 Tokens = `128 MiB`；其他 15 个 Full Attention 层及 48 个 Linear Attention 的状态不计在内。
- 假如以后远端 KV 真的改为 FP8，长度计算和 scale 元数据协议均需版本化；**当前权重 FP8 不意味着这些数值需要除以 2**。

---

## 2. 系统架构与数据流

### 2.1 逻辑组件

```text
                  +-----------------------------------+
                  |        Application / Runner        |
                  | prompts, prefill, decode, compare  |
                  +----------------+------------------+
                                   |
                     +-------------v-------------+
                     | Qwen3.8 Model Adapter      |
                     | Layer 3 only; BF16 Q/K/V   |
                     | Cache metadata bridge      |
                     +-------------+-------------+
                                   |
                    +--------------v--------------+
                    | CPU PNM Runtime               |
                    | sessions, handles, commands   |
                    | pinned buffers, metrics       |
                    +--------------+--------------+
                                   |
                           Backend Protocol
                         /                    \
       +-----------------v-----------+   +----v----------------+
       | GPU1 Emulator Backend       |   | Real ASIC Backend    |
       | explicit pinned staging     |   | future driver/DMA    |
       | GPU1 KV allocator           |   | BAR/MMIO/CQ          |
       | GPU1 Attention kernel       |   | device DRAM + KXP    |
       +-----------------------------+   +---------------------+
```

**数据面**（本阶段）：

```text
GPU0 Q/K_new/V_new (GPU0 VRAM)
  -> D2H into preallocated pinned Host Input
  -> H2D into GPU1 input buffer
  -> GPU1 append remote KV -> GPU1 QK^T / Softmax / AV
  -> D2H into pinned Host Output
  -> H2D into GPU0 output buffer
  -> GPU0 Gate + output projection + subsequent model layers
```

**控制面**：Model Adapter 请求 CPU Runtime -> 提交含 `command_id` 的命令 -> Backend 执行 -> Completion -> Runtime 校验状态和长度 -> Model Adapter 使用 O。

### 2.2 Prefill（第一版 GPU0 原生计算）

1. GPU0 使用官方 FP8 模型执行完整文本 Prompt 的原生 Prefill，所有层原样计算。
2. 在原生 Cache 中定位目标 Layer 3 的 **最终 K/V**（经过 K Norm、RoPE 的 K；以及 V），核对 dtype、shape、token 长度。
3. 通过 `ALLOC_CACHE` + `INIT_KV` 将 **目标层**全部历史 KV 初始化到 GPU1。需要支持分块物理传输：一次逻辑初始化不等于只能发一个 DMA 描述符。
4. GPU1 返回 cache handle、有效长度及完成状态；Runtime 维护 request -> (layer,handle,len)。
5. `shadow` 模式：暂保留 GPU0 历史 KV，对照使用；`strict` 模式：让 GPU0 不再持有该层的完整 KV Tensor，只保留必要的长度/位置元数据。后者要通过 Cache Bridge 验证，不能简单 `del` 缓存。

### 2.3 Decode（每次 1 Token）

1. GPU0 目标层计算 Q/K_new/V_new（含 Q/K Norm 和 partial RoPE），同时保留 Gate 分支。
2. Model Adapter 向 Runtime 传递 Q/K_new/V_new、request/layer、绝对 token 位置、期望旧缓存长度、`command_id`。
3. Runtime 显式 DMA 模拟传输到 GPU1。GPU1 检查旧长度一致、位置连续、Handle 有效、dtype 与形状正确。
4. GPU1 将 K_new/V_new 追加到设备历史缓存，使用 **包含当前 Token 在内的全部有效历史 KV** 做 Attention。
5. GPU1 返回 **未做 Gate / o_proj 的 Attention 核心输出 O**；Runtime 返回 GPU0。
6. GPU0 沿用原模型公式完成 `reshape -> sigmoid(gate) -> elementwise product -> o_proj`，继续 Decoder Layer 的残差、MLP 等计算。
7. Completion 与序列长度一致时提交新长度，失败不可错误增长或重复追加。

### 2.4 Causal Mask 注意事项

单 Token Decode（Q 长度=1，K 长度=历史+1，无 padding、缓存里只有有效 Token）可以使用 `scaled_dot_product_attention(..., is_causal=False, enable_gqa=True)`；这时当前 Q 能看到所有已有 K。不要对非方形 `[1, S]` 注意力直接使用 `is_causal=True`，因为 PyTorch 默认**左上对齐**，会错误屏蔽大部分历史。

Prefill 使用原生正确 causal mask；多 Token Decode、padding、多请求和 Chunked Prefill 要实现位置对齐的 mask，不能把本阶段 `is_causal=False` 的简化套用到所有场景。

### 2.5 缓存所有权与两种模式

- **Shadow**：GPU0 原始 Cache 继续更新；GPU1 持有镜像并计算 O，逐步比较。易定位错误，但不代表释放了 GPU0 历史 KV。
- **Strict Remote KV**：Layer 3 的真实历史 KV **只在 GPU1**。GPU0 上的 Cache Bridge 保留必要长度/位置/请求映射；其他 15 个 Full Attention 层和 48 个 Linear Attention 层的状态继续正常维护。所有数据访问必须通过 Runtime/Backend。

在 Qwen3.8 混合架构中，原生 Cache 对象的 `get_seq_length()`、第一个 Full Attention 层、`position_ids`/`cache_position`（如当前版本提供）、Attention Mask 生成逻辑可能耦合。进入 Strict 之前必须阅读锁定版本的实现并编写集成测试。不能假定 `past_key_values.key_cache[3]` 或 `past_key_values[3]` 在所有 Transformers 版本都可用。

---

## 3. 建议代码仓库布局（必须按职责拆分）

```text
pnm-qwen38-fp8/
├── README.md
├── pyproject.toml
├── requirements/                 # 环境解析之后冻结依赖版本
│   ├── constraints.txt
│   └── environment.lock.txt
├── configs/
│   ├── model.yaml                # 官方模型、revision、GPU0 固定、量化检测
│   ├── runtime.yaml              # dtype, pinned buffers, transport, errors
│   └── tests.yaml
├── docs/
│   ├── ARCHITECTURE.md
│   ├── MODEL_INSPECTION.md       # 源码类名、方法签名、缓存路径的检查结果
│   ├── PROTOCOL_V1.md
│   ├── ENV_REPORT.md
│   └── TEST_REPORT.md
├── scripts/
│   ├── inspect_env.py
│   ├── inspect_model_config.py
│   ├── smoke_load_fp8.py
│   ├── dump_qkv_metadata.py
│   ├── run_toy_attention.py
│   ├── run_model_baseline.py
│   ├── run_shadow_decode.py
│   ├── run_strict_remote_decode.py
│   └── benchmark.py
├── src/pnm/
│   ├── __init__.py
│   ├── protocol/
│   │   ├── types.py             # dtype/layout/command/result immutable types
│   │   ├── validate.py          # strict tensor + command input checks
│   │   ├── errors.py            # PNMError hierarchy and status values
│   │   └── codec.py             # future binary ABI; initial Python contract
│   ├── runtime/
│   │   ├── api.py               # public model-independent API
│   │   ├── session.py           # per request/layer handles, length, teardown
│   │   ├── command.py           # command IDs, state transitions, retries
│   │   ├── buffer_pool.py       # reusable pinned host buffers
│   │   ├── coordinator.py       # transport/compute completion + synchronization
│   │   └── metrics.py           # transfer bytes, latency and events
│   ├── backends/
│   │   ├── base.py              # abstract PNMBackend contract
│   │   ├── gpu_emulator.py      # calls GPU1 cache + attention + transport
│   │   ├── gpu_memory.py        # GPU1-owned allocation, length and free
│   │   ├── gpu_attention.py     # PyTorch GQA / SDPA reference kernel
│   │   ├── cuda_staging.py      # explicit D2H/H2D pinned transfer
│   │   └── pcie_asic.py         # placeholder raises NotImplementedError
│   ├── models/
│   │   └── qwen38/
│   │       ├── inspection.py    # discovery of actual model paths/classes
│   │       ├── attention_adapter.py
│   │       ├── cache_bridge.py
│   │       ├── injection.py     # reversible adapter install/remove
│   │       ├── runner.py        # prefill + teacher-forced decode
│   │       └── compare.py       # QKV/Attention/layer/logits comparison
│   └── utils/
│       ├── seed.py
│       └── trace.py
├── tests/
│   ├── unit/
│   │   ├── test_gqa.py
│   │   ├── test_decode_mask.py
│   │   ├── test_protocol.py
│   │   ├── test_command_idempotency.py
│   │   └── test_cache_length.py
│   ├── integration/
│   │   ├── test_gpu_staging.py
│   │   ├── test_gpu_emulator_cache.py
│   │   ├── test_qwen_shadow.py
│   │   └── test_qwen_strict.py
│   └── fault/
│       ├── test_out_of_order.py
│       └── test_incomplete_command.py
└── artifacts/                     # per-run JSON/CSV results; omit weights
    ├── env/
    ├── traces/
    ├── baselines/
    └── reports/
```

**依赖方向只允许：** `models -> runtime -> backend -> GPU1`；`protocol` 被各层共享；`backend` 不得 import Qwen/Transformers 类；`runtime` 不得直接访问 `past_key_values`；`models` 不得直接访问 GPU1 Tensor 存储。配置和产出要放仓库，而不是写到模型文件夹中。

---

## 4. 模块接口契约（要求 Agent 先实现这些类型/测试）

以下为**可执行方向的接口规范/示例**，不是声称已测试的成品；需要与测试一起落地。

### 4.1 统一数据结构

`protocol/types.py`：

```python
from dataclasses import dataclass
from enum import Enum
from typing import Optional

class DType(Enum):
    BF16 = "bf16"
    FP16 = "fp16"
    FP32 = "fp32"
    # FP8 only after separate versioned design and scale metadata

class Layout(Enum):
    BHSD = "B,H,S,D"   # batch, heads, sequence, head_dim

@dataclass(frozen=True)
class AttentionSpec:
    layer_id: int = 3
    num_q_heads: int = 24
    num_kv_heads: int = 4
    head_dim: int = 256
    dtype: DType = DType.BF16
    layout: Layout = Layout.BHSD

@dataclass(frozen=True)
class CacheConfig:
    request_id: str
    layer_id: int
    max_seq_len: int
    batch_size: int = 1

@dataclass(frozen=True)
class CacheHandle:
    value: int            # opaque device-assigned handle
    generation: int       # device reset invalidates old generations

@dataclass(frozen=True)
class CommandInfo:
    command_id: int
    request_id: str
    layer_id: int
    expected_seq_len: int   # previous, before append
    token_position: int     # 0-based absolute position

@dataclass(frozen=True)
class Completion:
    command_id: int
    cache_handle: CacheHandle
    committed_seq_len: int
    status: str
    detail: Optional[str] = None
```

要通过协议版本字段和校验码逐步扩展；不要现阶段声称这些 Python 类型就是 ASIC 二进制 ABI。`batch_size` 固定 1 是 P1-P4 的限制，不可静默接受其他值。

### 4.2 后端抽象：不依赖模型框架

`backends/base.py`，推荐用 `ABC`：

```python
from abc import ABC, abstractmethod
import torch

class PNMBackend(ABC):
    @abstractmethod
    def allocate(self, cfg: CacheConfig) -> CacheHandle: ...

    @abstractmethod
    def init_kv(self, handle: CacheHandle,
                k_prefill: torch.Tensor, v_prefill: torch.Tensor,
                command_id: int) -> Completion: ...

    @abstractmethod
    def append_and_attend(self, handle: CacheHandle,
                          q: torch.Tensor, k_new: torch.Tensor,
                          v_new: torch.Tensor, cmd: CommandInfo
                          ) -> tuple[torch.Tensor, Completion]: ...

    @abstractmethod
    def free(self, handle: CacheHandle) -> None: ...

    @abstractmethod
    def get_status(self, command_id: int) -> Completion | None: ...
```

接口需要明确 `torch.Tensor` 位于 GPU0，但这是**Python 模拟接口**。未来真实 ASIC 的驱动适配接口内部应改为主机缓冲区句柄/注册地址，而非将 CUDA Tensor 作为硬件 ABI。Runtime 的高层语义保持不变。

### 4.3 Runtime 对模型暴露的最小 API

`runtime/api.py`：

```python
class PNMRuntime:
    def __init__(self, backend: PNMBackend, spec: AttentionSpec): ...

    def create_session(self, request_id: str, max_seq_len: int) -> CacheHandle: ...

    def initialize_prefill(self, request_id: str, k: "Tensor", v: "Tensor") -> None: ...

    def decode_step(self, request_id: str, position: int,
                    q: "Tensor", k_new: "Tensor", v_new: "Tensor") -> "Tensor": ...

    def get_committed_length(self, request_id: str) -> int: ...

    def release_session(self, request_id: str) -> None: ...
```

Runtime 负责给命令分配稳定的 ID、校验期望长度、等待 Completion、更新时间/字节数、管理 request 生命周期、失败时恢复/报错；禁止在 Runtime 内写 `model.language_model.layers[3]` 之类的耦合代码。

### 4.4 Device Cache Manager

GPU1 `gpu_memory.py` 内应保存：

```text
handle -> {
  request_id, layer_id, capacity_tokens, committed_tokens,
  k_tensor: GPU1 [1,4,capacity,256],
  v_tensor: GPU1 [1,4,capacity,256],
  generation, last_committed_command_id, status
}
```

首版采用容量上限确定的 `torch.empty` 预分配，避免每 Token 反复 `torch.cat` 导致历史缓存重新分配/复制；有效区域为 `[..., :committed_tokens, :]`。分配失败要返回有意义的异常/状态。

**执行原子性**：`append_and_attend` 检查 `expected_seq_len==committed_tokens`、`token_position==committed_tokens`。同一个 `command_id` 重试时，返回已缓存的 Completion/O，不能重复追加。不同 ID 但同一个旧长度/位置必须拒绝。计算/传输异常发生时，要么安全回滚、要么标记 handle `POISONED` 并阻止继续使用。不可在未知状态下假装成功。

### 4.5 GPU1 Attention Reference

`gpu_attention.py` 用 PyTorch SDPA 验证，固定 `eval()`、`dropout_p=0.0`：

```python
import torch.nn.functional as F

def attend_single_decode(q, k_cache, v_cache):
    # All tensors live on cuda:1; shapes:
    # q=[1,24,1,256], k_cache/v_cache=[1,4,S,256]
    # S includes current token, with no padding/invalid slots.
    return F.scaled_dot_product_attention(
        q, k_cache, v_cache,
        is_causal=False,
        enable_gqa=True,
        dropout_p=0.0,
    )
```

同时实现一个 **独立数学参考**：`repeat_interleave` K/V -> `QK^T/sqrt(256)` -> **FP32 Softmax** -> `AV`。必须用两套实现比对，避免 GPU1 与 GPU0 恰好犯同一个 Mask/头映射错误。

### 4.6 Host Staging

`cuda_staging.py` 必须：

1. 对 Q/K/V/O 使用可复用、预分配 pinned CPU Tensor（`torch.empty(..., device='cpu', pin_memory=True)`）；按最大形状管理 Buffer Pool。
2. 显式 `gpu0_tensor -> pinned_cpu_buffer -> gpu1_tensor`，反向一样。传入之前检查 contiguous、dtype、字节数，必要时 `.contiguous()` 的成本单独计量。
3. CPU/GPU 操作必须遵循正确同步：先实现 **同步正确路径**；后续可用 stream + CUDA Event 异步重叠，但绝不能在 D2H 完成前让下一设备读取 Host Buffer。
4. 日志分别记录四段 copy 的 payload bytes、传输耗时、CPU queue/submit、GPU1 kernel 时间、整体端到端时间。
5. 每 Token 不能重新 pin/unpin/malloc Host Buffer；不能在日志中出现历史 KV 随 Decode 长度线性增长的传输记录。
6. P2P 可探测，但验收时必须能强制 `transport=host_staging`；严禁透明替换成 P2P 路径。

### 4.7 命令状态机

推荐：`NEW -> SUBMITTED -> RUNNING -> COMPLETED`；错误分支：`FAILED/POISONED/CANCELLED/TIMEOUT_UNKNOWN`。`TIMEOUT_UNKNOWN` 代表设备可能已写入缓存但 Completion 未送达，必须查询 `command_id` 或使 Handle 失效，不得盲目再次追加。

最小操作码：`ALLOC_CACHE`、`INIT_KV`、`APPEND_ATTEND`、`QUERY_STATUS`、`FREE_CACHE`。记录 `request_id`、`layer_id`、`cache_handle`、`command_id`、`expected_seq_len`、`token_position`、`dtype`、`layout`、`tensor shape/stride`、`byte length`、版本号、完成状态。未来驱动再增加 DMA 地址、对齐、队列深度、IOMMU 映射等。

---

## 5. Qwen 模型接入的具体要求

### 5.1 先查源码，不能凭经验猜类与缓存路径

`Qwen3.8-27B-FP8` 的 **FP8 是模型权重表示，模型结构走 `qwen3_5`**。在目标版本里定位并记录：

- 模型实例的实际 Python 类型、`model.language_model.layers` 或等效路径、目标 `layers[3].self_attn` 的类型。
- `Qwen3_5Attention.forward` 的参数签名；Q/K/V 与 Gate 的构造位置；Q/K Norm 与 RoPE 的调用；原生 Cache `update()` 的位置；Gate 与 `o_proj` 的次序。
- FP8 模块包装后，Q/K/V 投影是否仍可通过原模块调用而无需手工重建权重。
- 原生 Cache 类型、Layer 3 读取方式、`get_seq_length()`、位置推断、mask 构造、Stateful Linear Attention 状态；`cache_position` 是否存在及在何处生成，不能死套旧版本文档。

当前公开的 `Qwen3_5Attention` 参考实现具有：Q 投影并拆出 Gate -> Q Norm；K 投影 -> K Norm；V 投影；对 Q/K 用 partial RoPE；Cache 更新；Attention Interface；`reshape` 后乘 `sigmoid(gate)`；`o_proj`。**在服务器必须以锁定安装版本和实际模型实例为准，并保存检查结果。**

### 5.2 三种 Attention Adapter 模式

`attention_adapter.py` 设计可配置且可恢复：

- `baseline`：原生模型执行，GPU0 完整负责，生成 Golden Reference；不要修改计算。
- `shadow`：Prefill 原生、初始化 GPU1；Decode 同时维护 GPU0 本地 Cache 与 GPU1 Cache，捕获 GPU0 Norm/RoPE 后 Q/K/V，计算 remote Attention O；记录 `O_local vs O_remote`、Gate 后值、Layer Output、logits；先只比较、再可选择返回 remote O。
- `strict`：Prefill 完成后，Layer 3 的历史 KV 迁入 GPU1；本地只留必要元数据；Decode Layer 3 不再更新 GPU0 历史 KV；远端计算 O 真实参与模型输出。

严格只替换目标 `self_attn` 模块执行路径，保留模型本身的 Q/K/V 投影模块实例（包括量化权重与 scale），不重建 FP8 参数、不修改其余 63 层、不改变模型其它 dtype。加载权重与卸载层代码不得互相交织。

### 5.3 Cache Bridge 设计和验证

`cache_bridge.py` 是**单独的工作项和风险点**，不能简单留空：

- 明确把 **Layer 3 的数据缓存**（GPU1 独占）与 **全局序列位置/缓存元数据**（GPU0 仍可使用）分离。
- 保留其他 Full Attention 层和 DeltaNet 层原生 Cache 状态。
- 保证 `get_seq_length()` 和任何上层 position/mask 推断与基线一致。
- 如基于原生 Cache 定制 Proxy/Layer，严格实现锁定版本所需接口，写 `batch=1`、prefill 4/16/128、decode 连续至少 16 步测试。
- 如现有 HF Cache API 无法安全支持，采用 **显式 position_ids/attention mask + 可验证的上层 Cache Adapter**，不能通过全局 monkey-patch 隐瞒结构问题；记录变更和限制。
- Shadow Mode 的输出相同不等于 Strict Mode 合格。Strict 必须证明目标层 GPU0 Cache 中没有持续增长的历史 K/V Tensor。

### 5.4 解码生成验证策略

优先使用 **teacher forcing / 固定 Token 序列**：基线生成或保存一串 token IDs，远端路径每步喂入相同 token，逐层比较 O、Gate 后值、目标层输出及 logits；这样不被采样随机性放大。再用 `do_sample=False` 做 greedy 生成比较；如生成文本有差异，先以数值指标定位，不把文本是否完全一致当成唯一标准。

首次只做 text-only，固定 prompt、禁用 dropout、固定 RNG seed、记录 chat template、模型 revision、HF/PyTorch/CUDA 版本。生成 API 可能管理自己的 Cache/位置；严格模式不要直接调用未经适配的 `model.generate()`，应先使用可以显式控制输入与 Cache 的 Runner 验证，再接管 `generate()`。

### 5.5 FP8 加载注意事项

- 官方示例使用 `AutoProcessor` 和 `AutoModelForMultimodalLM`；建议作为加载首选，但**先检查安装版本是否具有该类**。
- 加载时对整个模型固定 `device_map={"": "cuda:0"}`（或等效 GPU0-only），**不要使用** `device_map="auto"`。
- 官方 checkpoint 自带量化配置；不得再次使用 `load_in_8bit`、`load_in_4bit`、额外 AWQ/GPTQ 或手工 `.to(torch.float16)` 把 FP8 权重处理掉。
- 检查 GPU compute capability、CUDA、Torch/Transformers 对当前 FP8 模块的支持。首次成功后冻结精确版本。因硬件/内核不支持而失败时，要有清晰错误，不得隐式加载 GPU1。
- FP8 权重降低占用，但官方仓库约 30.9 GB 是**存储仓库体量**，不是可直接套用的 CUDA 显存上限：还需要非量化模块、激活、推理状态、框架开销；GPU0 显存需 P0 实测。GPU0 容量不足时先完成 P1/P2，向用户报告 Full Model 阻塞，不改变架构目标。

---

## 6. P0-P6 里程碑与验收标准（严格串行）

### P0：服务器与模型可行性审计（第一个任务）

**目的**：确认模型、GPU、环境、FP8 加载和目录结构，不急于实现完整 Runtime。

**步骤**：

1. 读取当前仓库/目录；保留现有内容，不覆盖用户已有文件。建立上述目录，创建 `git` 分支/提交（仓库已存在则遵循已有工作流）。
2. 收集 GPU 名称、VRAM、compute capability、驱动版本、CUDA runtime、PyTorch/Transformers/accelerate/huggingface_hub 版本、CPU/NUMA、磁盘剩余容量和 P2P 状态。
3. 从 HF **先下载配置文件**，确认模型 ID、量化/层配置；锁定 commit revision（SHA）并记录，避免 `main` 漂移。
4. 做环境兼容性检查，确定支持 FP8 的实际依赖版本；版本冲突须先报告和隔离 venv。不可随意升级机器级 CUDA/NVIDIA 驱动。
5. 在 GPU0 显存与 FP8 支持满足条件时，加载官方模型，关闭训练，text-only 做短 Prompt 推理；验证 **所有权重设备位置都不在 GPU1**。输出 Loader/实际模型路径/`q_proj` dtype/使用的 quantization config/真实 QKV dtype。不能加载则报告确切原因并继续 P1/P2。
6. 固定 `MODEL_ID`、model revision、HF/PyTorch commit或版本、模型读取路径、seed、tokenizer 模板、进程环境；输出环境 `JSON + Markdown`。

推荐 shell 检查：

```bash
nvidia-smi
nvidia-smi topo -m
nvidia-smi topo -p2p p || true
nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap --format=csv
python - <<'PY'
import platform, torch
print('platform', platform.platform())
print('torch', torch.__version__, 'torch_cuda', torch.version.cuda)
print('gpu_count', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
if torch.cuda.device_count() >= 2:
    print('peer_0_to_1', torch.cuda.can_device_access_peer(0,1))
    print('peer_1_to_0', torch.cuda.can_device_access_peer(1,0))
PY
python -m pip list | grep -E 'torch|transformers|accelerate|huggingface|safetensors' || true
lscpu | head -n 25
numactl --hardware || true
df -h
```

配置审计 Python 示例（依赖 `huggingface_hub`）：

```python
import json
from huggingface_hub import HfApi, hf_hub_download
model_id = 'Qwen/Qwen3.8-27B-FP8'
revision = HfApi().model_info(model_id).sha
path = hf_hub_download(model_id, 'config.json', revision=revision)
with open(path, encoding='utf-8') as f:
    cfg = json.load(f)
tc = cfg['text_config']
assert tc['num_hidden_layers'] == 64
assert tc['num_attention_heads'] == 24
assert tc['num_key_value_heads'] == 4
assert tc['head_dim'] == 256
assert tc['layer_types'][3] == 'full_attention'
assert cfg['quantization_config']['quant_method'] == 'fp8'
print('revision', revision)
print('arch', cfg.get('architectures'))
print('dtype', tc.get('dtype'))
print('quant', {k: cfg['quantization_config'].get(k)
                for k in ('quant_method', 'fmt', 'activation_scheme', 'weight_block_size')})
```

若运行时缺 `AutoModelForMultimodalLM`，先记录并升级 **虚拟环境内** 的兼容版本；不要盲目把多模态 checkpoint 当成可直接用 `AutoModelForCausalLM` 加载。示意加载配置：

```python
from transformers import AutoProcessor, AutoModelForMultimodalLM
model_id = 'Qwen/Qwen3.8-27B-FP8'
processor = AutoProcessor.from_pretrained(model_id, revision=revision)
model = AutoModelForMultimodalLM.from_pretrained(
    model_id,
    revision=revision,
    device_map={'': 'cuda:0'},  # critical: GPU0 only
    dtype='auto',
)
model.eval()
# Follow pinned Transformers model card signature; assert no tensor on cuda:1.
```

**P0 验收**：`ENV_REPORT.md`、`MODEL_INSPECTION.md`、lock file、配置检查 JSON，明确 GO/NO-GO。只有实际加载成功才记录 Full Model 为 PASS；仅找到仓库不能算 PASS。

### P1：独立 Attention / Mask / GQA 正确性

**不加载 27B 权重**。用模拟数据 `[B=1,Hq=24,Hkv=4,D=256]`，覆盖：

1. CPU/GPU0 数学参考 Attention；GQA 对应 6:1 头映射；FP32 softmax。
2. Prefill causal mask：如 4 token 的下三角。
3. Decode：先 4 token 历史，追加位置 4，第 5 个 token Q 应看到所有 K0..K4。验证 `is_causal=True` 的错误示例与正确实现的差异。
4. 逐步追加 1、2、4、16、128 token；结果与“从头构建完整 KV”参考一致。
5. 空缓存、形状错、非连续张量、混合 dtype、非法 mask 等应明确拒绝。
6. float32 基准严格比较；BF16 先记录 max_abs、mean_abs、relative、cosine、NaN/Inf，再按实际参考确定误差门限。

**P1 验收**：pytest 所有 toy unit tests 全部 PASS，shape、K/V 尺寸、Mask 结果可复现，提供最小运行命令。

### P2：GPU1 Emulator + Host Staging + 协议

1. 编写 `PNMBackend`、GPU1 KV allocator、预分配 pinned buffer、GPU1 Attention、Runtime Session、命令状态机。
2. `ALLOC_CACHE -> INIT_KV -> APPEND_ATTEND x N -> FREE_CACHE` 完整闭环；不能每 token `torch.cat`。
3. Prefill 模拟 4096 token 的 16 MiB KV，分块通过 pinned buffer 初始化，Verify GPU1 缓存内容和长度。
4. Decode 每步 GPU0 payload 16 KiB、返回 12 KiB，额外元数据计入命令统计；四段拷贝强制 Host Staging。
5. 重复 `command_id` 不重复追加；乱序、过期 handle、越界、超时未知与资源不足必须有测试。
6. 每个请求/缓存 handle 释放后不能被误用；设备模拟 reset 后旧 generation 无效。

**P2 验收**：不加载 Qwen 的端到端数据流完全跑通，输出 JSON trace（请求、handle、token、命令状态、字节数、四段传输耗时、kernel、total）；未观察到历史 KV 逐 token 重传；异常注入覆盖基础状态机。

### P3：Qwen3.8-FP8 单层 Shadow Mode

**前置**：P0 Full Model PASS、P1/P2 PASS。

1. 原生 FP8 模型固定 CPU/GPU0 环境、准备 baseline。尽量先用显式的预填充/逐 token `forward()` Runner，而不是 `generate()`。
2. 按锁定源码找到 `layers[3].self_attn`，注入可恢复的 Adapter，保留原模块已有 FP8 权重/Quantization 与 Gate。
3. Prefill 在 GPU0 原生执行，并将 Layer 3 产生的 K/V 初始化到 GPU1。
4. Decode 捕获已 Norm+RoPE 的 Q/K_new/V_new；原本 GPU0 的 KV 仍更新、GPU1 追加计算；先 **只观察比较**，然后启用 `return_remote` 让 GPU1 生成的 O 影响后续层。
5. 比较：`q,k_new,v_new`，`O_core`，`O_after_gate`，`layer3_output`，每步 `logits`；固定 token sequence，GPU0/1 内核数值指标与性能追踪。
6. 保证其余 63 层没有发生未授权的 Forward 改动；确认 GPU1 没加载模型权重。

**P3 验收**：完整模型有稳定 Decode；相同固定 Tokens 的逐级数值误差在验收门限内；可以通过开关恢复原始 `forward()`；能证明返回的 GPU1 O 真实进入模型计算链而非只是旁路输出。

### P4：Strict Remote KV（真正完成目标）

1. 设计实现 `CacheBridge`：目标层真实 KV 只在 GPU1、GPU0 只保留必要 Cache Metadata。不要假装 GPU0 缓存仍然是权威副本。
2. 以显式 Token 绝对位置保障位置编码与 Mask 正确；上层 `get_seq_length()` 等对外语义正确。保证其他 63 层状态正常递增。
3. Runtime/Device 的提交长度一致。原生 `Layer3` 不可再向 GPU0 追加完整历史 KV。
4. 连续至少 16/128 次 Decode，短到较长 Prompt（4,16,128,1024,4096）分批开展，记录失败范围和 dtype。
5. 检查 `CacheBridge`、GPU0 目标层对象图中无逐 Token 增长的 K/V；允许 PyTorch allocator 的 reserved memory 未立刻下降，但必须区分 **allocated 与 reserved** 及真实张量所有权。
6. 模拟 device lost、duplicate completion、旧 handle 等错误时必须 fail closed，禁止静默本地回退掩盖远端依赖。

**P4 验收**：**GPU1 持有并追加 Layer 3 的全部有效历史 KV；GPU0 不保留目标层完整历史 K/V；GPU1 O 真正决定最终 logits；与 baseline 数值一致；其余层正常**。这是本项目核心验收门。

### P5：性能、压力与故障实验

首轮先用 single request + batch1 严格模式。Prompt 长度建议 128/1024/4096/8192（每项先根据 GPU0 资源及运行时间判断），输出步数 16/128；单独记录：

- Prefill GPU0 计算耗时，完整 Layer 3 KV 初始化 bytes/time。
- Decode GPU0 Q/K/V 准备时间，GPU0->Host，Host->GPU1，GPU1 append，GPU1 kernel，GPU1->Host，Host->GPU0，命令准备/排队，端到端 layer offload，完整 token 时延。
- local baseline 对照延迟、显存 allocated/reserved、GPU1 KV 使用量、设备容量占用、吞吐/上下文长度关系。
- 拷贝时间用正确的 CUDA Event、GPU 同步与 CPU 高精度计时；注意不同 GPU 时钟与异步执行，不能单纯把分项相加当作真实总时延。
- NUMA、Host buffer 位置、P2P 可用性、线程亲和性、Buffer 复用率；不能把 GPU1 SDPA 速度当做未来 KXP 吞吐。
- 注入缓存容量耗尽、重复提交、超时未知、设备重置、传输异常、请求取消。

**P5 验收**：CSV+JSON+Markdown 报告，包含机器及版本信息、统计方法、p50/p95/p99、每阶段 bytes/time、baseline 对比；清楚标注“GPU 模拟，不代表真实 ASIC 性能”。

### P6：以后接入真实 PCIe 驱动 / vLLM（不属于现阶段提交）

- 真实硬件：保留 Session/Cache/Command 语义，替换 `PNMBackend` 的传输与设备计算实现；将模拟的 Host pinned CUDA buffer 替换为**驱动可 DMA 的页锁定和 IOMMU 映射机制**（两者不是一回事）。
- 设备协议冻结前确认：究竟是普通 PCIe DMA Endpoint，还是具备 CXL.io/CXL.mem（可选 CXL.cache）的 CXL Type-2 设备；地址映射、内存一致性和 BAR/MMIO 不可混为一谈。
- vLLM：后置为 Adapter + Attention Backend + KV Cache Manager/Scheduler 集成；KV Transfer Connector 只能解决部分传输问题，不能自动实现在 PNM 计算 Attention。
- 后续计划扩展多层 Full Attention、Batch、多请求、Block/分页 KV、pipeline/async、FP8 KV/Scale 与视觉输入。

---

## 7. 验收指标：硬性检查与建议误差阈值

### 7.1 硬性不变量（不许放宽）

| 编号 | 检查项 | 要求 |
|---|---|---|
| INV-01 | 模型 ID | 严格 `Qwen/Qwen3.8-27B-FP8`；锁定 revision |
| INV-02 | 模型放置 | 权重不允许存储到 GPU1 |
| INV-03 | Layer | 第一轮只卸载 `layer_idx=3` |
| INV-04 | FP8/BF16 | 官方 FP8 权重保留；传输 dtype 实际检查，不擅自改成 FP8 |
| INV-05 | Prefill | Layer 3 历史 KV 逻辑初始化一次，允许物理分块 |
| INV-06 | Decode | 输入只含当前 Q/K/V 和元数据，不能重传历史 KV |
| INV-07 | GPU1 Cache | append 后长度正好 +1；头数和维度正确 |
| INV-08 | Mask | 单 Token Decode 允许访问全部有效历史（含当前） |
| INV-09 | Strict | GPU0 不再保存 Layer 3 的完整历史 KV |
| INV-10 | 命令 | 不重复追加、不乱序提交、不盲重试超时未知命令 |
| INV-11 | Model | 其他 63 层及 Gated DeltaNet 状态正确 |
| INV-12 | Transport | Host staging 可审计，Pinned Buffer 复用 |

### 7.2 数值阈值与签字流程

误差容忍不能凭空保证 FP8+BF16 在不同 GPU 算子实现上逐位相同。按两级执行：

1. `float32 toy/reference`：建议起始 `max_abs <= 1e-4`、`mean_abs <= 1e-5`（实现中仍需考虑维度和数值范围）；发现 NaN/Inf 或 Mask/头映射错误直接失败。
2. `BF16 QKV remote vs local`：建议先以 `atol=1e-2, rtol=1e-2` 作为初始排查尺度，**并输出最大/均方/相对误差与 cosine**。Near-zero 不以相对误差独立判断；若原生 Attention Backend 使用不同累加策略，可以经基线实测确定新的门限，但必须附误差分布、算子配置、批准理由。绝不因最终文本相同而忽略严重的中间层误差。
3. `FP8 Full Model`：必须以**同一官方 FP8 checkpoint**的 GPU0 Baseline 比较，且固定输入 Tokens；不要把 FP8 原型和 BF16 权重模型的 logits 混合当作卸载正确性基准。

---

## 8. 代码提交、测试运行与报告要求

### 8.1 日常工作节奏

每完成一个阶段：

1. 先给出变更文件列表和模块关系说明。
2. 提供精确命令，如 `python -m pytest tests/unit -q`；报告实际输出摘要（通过、失败、跳过）。
3. 把环境信息、模型 revision、运行命令、seed、输入样例写入 `artifacts/reports/Px_...md`。
4. 失败必须保留日志与复现命令；分清代码错误、硬件条件不足、依赖不匹配。
5. 提交符合 Git 习惯的分阶段 commits；不得修改安装包源码、下载权重到 Git 仓库或删除已有用户文件。
6. 未经阶段门验收，不自动推进到下一阶段；但当 Full Model 因资源不足无法进行时，允许继续与模型无关的 P1/P2。

### 8.2 日志统一格式

每步至少：

```json
{
  "request_id": "req-001",
  "layer_id": 3,
  "command_id": 1005,
  "stage": "decode",
  "token_position": 4096,
  "kv_len_before": 4096,
  "kv_len_after": 4097,
  "dtype": "bfloat16",
  "layout": "B,H,S,D",
  "q_bytes": 12288,
  "kv_new_bytes": 4096,
  "output_bytes": 12288,
  "transport": "host_staging",
  "status": "COMPLETED",
  "timings_us": {
    "gpu0_d2h": null,
    "host_to_gpu1": null,
    "gpu1_append": null,
    "gpu1_attention": null,
    "gpu1_d2h": null,
    "host_to_gpu0": null,
    "end_to_end": null
  }
}
```

`null` 表示该阶段计时尚未采集，不得用估算数字假装测量值。另保存统计后的 p50/p95/p99 与设备/版本信息。

### 8.3 首轮必须完成的测试清单

- `test_gqa.py`：24 Q Heads / 4 KV Heads、Head 映射、输出 `[1,24,1,256]`。
- `test_decode_mask.py`：4 历史 + 1 当前 Token 时所有 5 个位置可见；证明错误 `is_causal=True` 与正确输出不一致。
- `test_protocol.py`：dtype/layout/shape/capacity/position 前置校验。
- `test_cache_length.py`：Prefill length、逐步 +1、不能跳位置。
- `test_command_idempotency.py`：重复命令、乱序、超时未知状态。
- `test_gpu_staging.py`：Pinned CPU 缓冲可复用、四段显式传输、字节计数。
- `test_gpu_emulator_cache.py`：Init/append/free、GPU1 真实 KV 检查、越界。
- `test_qwen_shadow.py`：同权重相同 token 的 O/Layer/logits 比对（P3）。
- `test_qwen_strict.py`：目标层 GPU0 无完整历史、GPU1 为唯一权威、位置不乱（P4）。

---

## 9. 第一条发给服务器 Agent 的指令（复制即可）

```text
你现在是双 GPU PNM 软件项目的工程执行 Agent。

请把当前《PNM_Qwen3_8_27B_FP8_Implementation_Spec.md》作为工程规范。
项目明确使用官方模型 Qwen/Qwen3.8-27B-FP8，而不是 BF16 权重版本。

总体目标：GPU0 运行 Qwen3.8-27B-FP8；GPU1 模拟设备 DRAM 和 KXP Attention；CPU Runtime 管理请求/缓存/命令/主机 pinned buffer；未来替换为自研 PCIe 驱动和 ASIC Backend。

现在仅执行 P0 + P1：
1. 审计 GPU/显存/算力/PCIe 拓扑/驱动/CUDA/依赖/磁盘/P2P；记录准确输出。
2. 仅下载官方模型 config（如可访问），校验结构和 fp8 量化配置，锁定 revision。
3. 判断 GPU0 是否能独占加载官方 FP8 权重及当前框架是否支持；可行才做短文本 smoke test，不允许把权重放 GPU1。
4. 对安装版本检查 Qwen3_5Attention、Cache、Gate、RoPE、target layer 的实际源码和运行时签名，写 MODEL_INSPECTION.md。
5. 建立规范中的完整目录骨架、协议类型、独立 GQA/Mask/Attention 数值测试，完成 P1。
6. 所有产出附真实 pytest 命令、结果和复现方式。若 Full Model 加载不可能，要明确报告阻塞但继续能运行的 Toy 测试。

严禁直接开发 vLLM，严禁一开始重构整个模型，严禁偷偷使用 device_map=auto，严禁宣称未实际跑过的结果已通过。

P0/P1 完成后请停止推进，给出：仓库目录树、变更文件列表、实际运行命令、PASS/FAIL 清单、模型加载条件、QKV dtype 检查情况、下一阶段 P2 的具体风险。
```

---

## 10. 风险登记与决策点

| 风险 | 触发情形 | 处理方式 |
|---|---|---|
| GPU0 显存不足 | FP8 checkpoint +非量化模块/上下文无法装入 | P0 报告；P1/P2 继续；不占 GPU1 权重空间 |
| GPU0 不支持 FP8 核 | 硬件/版本/量化实现不支持 | 固定准确异常，评估兼容运行时，不私自改模型 |
| Qwen 模块 API 漂移 | Transformers 改名/多模态模型包装改变 | 保存 inspection 报告，版本锁定，Adapter 不硬编码旧源码 |
| Cache Metadata 耦合 | strict 清除 Layer3 本地 KV 后位置错误 | Cache Bridge + teacher forcing + get_seq_length 测试 |
| 多卡拓扑 `SYS` | P2P 状态未知、跨 NUMA 导致时延抖动 | 单独检查 P2P；显式 host staging；记录 NUMA |
| CPU/GPU 异步竞态 | D2H 尚未完成就开始 H2D | P2 先同步正确，再事件/流优化 |
| 重试导致重复 KV | Completion 丢失或超时未知 | command_id 幂等、版本/长度检查、poison 状态 |
| Mask 错 | Q len=1, K len>1 误设 `is_causal=True` | 定向单元测试，Valid KV 范围限定 |
| FP8 与 KV dtype 混淆 | 误把 1 Byte/Token 元素当默认 | 审计真实 dtype + 以 BF16 v1 协议计算 |
| 模拟性能误判 | GPU1 SDPA 与实际 KXP 差别大 | 分项计时并声明不是 ASIC 预测 |

**决策原则**：先得到可重现的正确性和可观测的软件协议，再决定多请求性能、真实 ASIC 接入、vLLM 集成。P4 Strict Remote KV 的通过是当前阶段最大的技术里程碑。

---

## 11. 外部来源与验证范围

文档基于 2026-11-03 公开仓库配置/源码信息整理；服务器执行时必须再次读取并锁定自己的模型和依赖版本。

1. 官方 FP8 模型：https://huggingface.co/Qwen/Qwen3.8-27B-FP8
2. 官方模型配置：https://huggingface.co/Qwen/Qwen3.8-27B-FP8/blob/main/config.json
3. 官方模型介绍：https://huggingface.co/Qwen/Qwen3.8-27B
4. Transformers `Qwen3_5Attention` 参考源码：https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py
5. Transformers Qwen3.5 模型文档：https://huggingface.co/docs/transformers/model_doc/qwen3_5
6. PyTorch SDPA 说明：https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
7. CUDA 异步与内存模型：https://docs.nvidia.com/cuda/cuda-c-programming-guide/

**最终验收口径**：本文件是工程任务规范与参考接口，不表示已在目标服务器运行任何脚本、加载完整模型、测出任何性能或验证真实 ASIC。只有目标 Agent 产出的真实日志、测试报告和代码提交才构成验收证据。
