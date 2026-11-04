# P2：PASS

实际 GPU0 → pinned CPU → GPU1 → pinned CPU → GPU0。Prefill=128，连续 Decode=128，最终长度=256。Prefill 初始化一次，Decode 每步只传 Q_new/K_new/V_new，历史 KV 由 GPU1 管理。

四段传输和 kernel 统计：
```json
{
  "counts": {
    "command": 263,
    "cuda:0_d2h": 386,
    "cuda:1_h2d": 386,
    "attention": 128,
    "cuda:1_d2h": 128,
    "cuda:0_h2d": 128
  },
  "bytes": {
    "command": 0,
    "cuda:0_d2h": 2621440,
    "cuda:1_h2d": 2621440,
    "attention": 0,
    "cuda:1_d2h": 1572864,
    "cuda:0_h2d": 1572864
  },
  "milliseconds": {
    "command": 158.86380337178707,
    "cuda:0_d2h": 6.240704011172056,
    "cuda:1_h2d": 4.803552022203803,
    "attention": 20.386592231690884,
    "cuda:1_d2h": 2.0098239993676543,
    "cuda:0_h2d": 1.775327993556857
  }
}
```

预分配 pinned pool 1139776 bytes，is_pinned=True。预热后/循环结束显存分别为 `{'0': {'allocated': 9695232, 'reserved': 23068672, 'max_allocated': 9885184}, '1': {'allocated': 2242048, 'reserved': 4194304, 'max_allocated': 2330112}}` / `{'0': {'allocated': 9695232, 'reserved': 23068672, 'max_allocated': 10007552}, '1': {'allocated': 2242048, 'reserved': 4194304, 'max_allocated': 2354688}}`。每步误差和显存见 [P2_emulator.json](raw/P2_emulator.json)，所有命令和拷贝见 [P2_emulator.jsonl](raw/P2_emulator.jsonl)。

Backend 通过 handle 隔离 KV；重复命令 ID、长度/位置、无效 handle、ownership、shape/dtype/mask、容量、错误回滚和非默认 stream 有测试。Runtime 现在还校验 command ID、handle、owner 和 completion length，拒绝错误 Backend 响应。释放过程即使 transport.close 报错也释放缓存管理器。

每个 Decode 逻辑负载 28 KiB、四段 copy 56 KiB；不包含 CPU 权重 offload，不是 PCIe 线速。non_blocking copy 配合专用 streams/events，但当前命令同步执行，不宣称重叠。复现：`.venv/bin/python scripts/test_emulator.py --steps 128`。

