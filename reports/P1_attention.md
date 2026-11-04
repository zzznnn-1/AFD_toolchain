# P1 独立 Attention：PASS

固定种子 20261104；BF16、batch=1、24/4 heads、D=256。独立 FP32 分组 matmul/softmax 参考实现，不调用被测 SDPA 或 GQA 复制函数。TF32 禁用。Prefill 参考按 query chunk 计算，避免 4096 长度完整 score matrix 导致显存浪费。

| 长度 | 测试 | Max abs | Mean abs | Max relative | RMS | 状态 |
|---|---|---|---|---|---|---|
| 1 | prefill | 0 | 0 | 0 | 0 | PASS |
| 1 | decode | 0 | 0 | 0 | 0 | PASS |
| 1 | append | 0.0084133148 | 0.0010076396 | 0.9459793 | 0.0014441897 | PASS |
| 16 | prefill | 0.0096290112 | 0.00064264436 | 1.1446029 | 0.00093727413 | PASS |
| 16 | decode | 0.0043504238 | 0.00055644452 | 0.80814952 | 0.00075076125 | PASS |
| 16 | append | 0.0044697523 | 0.00052923744 | 0.71724111 | 0.00071319297 | PASS |
| 128 | prefill | 0.0097527504 | 0.0003579772 | 1.903706 | 0.00052874797 | PASS |
| 128 | decode | 0.0018552542 | 0.00022837543 | 0.48215577 | 0.00030114764 | PASS |
| 128 | append | 0.0016862154 | 0.00022436159 | 0.46932694 | 0.00029434389 | PASS |
| 1024 | prefill | 0.0089657307 | 0.00015430023 | 1.765368 | 0.00024381925 | PASS |
| 1024 | decode | 0.00056217611 | 8.176238e-05 | 0.16755982 | 0.00010796974 | PASS |
| 1024 | append | 0.00058415532 | 8.273383e-05 | 0.15756486 | 0.00010880502 | PASS |
| 4096 | prefill | 0.008097887 | 8.2777507e-05 | 1.8850085 | 0.0001394 | PASS |
| 4096 | decode | 0.00029609352 | 3.9642251e-05 | 0.11361483 | 5.2086856e-05 | PASS |
| 4096 | append | 0.00027739257 | 4.1135194e-05 | 0.091136932 | 5.3772746e-05 | PASS |
| 16 | boolean_padding | 0.004625082 | 0.0005832804 | 1.3482198 | 0.00079526735 | PASS |
| 16 | additive_padding | 0.004625082 | 0.0005832804 | 1.3482198 | 0.00079526735 | PASS |
| 16 | rectangular | 0.0051054955 | 0.00053724489 | 0.92963868 | 0.00072731968 | PASS |

逐元素验收 `abs(error) <= 0.02 + 0.02*abs(reference)`，相对误差分母保护 1e-3；相对误差只报告，不独立作为近零元素的淘汰标准。所有 finite 检查通过。6:1 head sentinel、decode 上三角对齐回归、矩形因果 mask、bool/additive padding 和全遮罩行亦有测试。

首轮 additive FP32 mask 遭遇本机 cuDNN “bias dtype 必须匹配 query”错误；边界显式转换至 Q dtype 后重新运行全部 P1 通过，未放宽容差。初始失败保留 `raw/P1_attention_initial_failure.log`。

复现：`.venv/bin/python scripts/test_attention.py`；日志 `raw/P1_attention.log`、数据 `raw/P1_attention.json`，补充回归 `tests/test_attention.py`。

