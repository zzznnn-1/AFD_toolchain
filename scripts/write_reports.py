"""Render phase reports from one artifact directory, including failed/blocked runs."""
import _bootstrap
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifact-dir', default='reports/raw')
    parser.add_argument('--report-dir', default='reports')
    args = parser.parse_args()
    artifacts, reports = Path(args.artifact_dir), Path(args.report_dir)
    reports.mkdir(parents=True, exist_ok=True)
    def read(name):
        path = artifacts / name
        return json.loads(path.read_text()) if path.exists() else {}
    def save(name, text): (reports/name).write_text(text+'\n')
    env, fp8, p1, p2, p3, audit = [read(n) for n in ['environment.json','fp8_probe.json',
        'P1_attention.json','P2_emulator.json','P3_suite.json','acceptance.json']]
    raw = str(artifacts.resolve().relative_to(reports.resolve())) if artifacts.resolve().is_relative_to(reports.resolve()) else str(artifacts.resolve())
    def link(name): return f'[{name}]({raw}/{name})'
    facts, plan = p3.get('model',{}), p3.get('preflight',{})
    gpu_rows = '\n'.join(f"| {g['index']} | {g['name']} | {g['capability']} | {g['total_bytes']/2**30:.3f} | {g['free_bytes']/2**30:.3f} |" for g in env.get('gpus',[]))
    status = audit.get('stages',{})
    save('P0_environment.md', f'''# P0：{status.get('P0','BLOCKED')}

记录时间：{env.get('time','未采集')}。硬件和版本以 {link('environment.json')} 为准。

| GPU | 型号 | CC | 总容量 GiB | 检查时空闲 GiB |
|---|---|---|---|---|
{gpu_rows}

软件版本：`{json.dumps(env.get('versions',{}),ensure_ascii=False)}`。
PyTorch CUDA={env.get('torch_cuda')}；Driver、系统 nvcc、CPU/RAM、磁盘、NUMA、topo -m/-p2p 和双向 Peer Access 的实际命令输出都在环境 JSON 内。不能把驱动支持的 CUDA 上限等同系统 nvcc。

官方模型 `{facts.get('model_id','尚未加载')}`，Revision `{facts.get('revision','未验证')}`。FP8 kernel 实测状态 **{fp8.get('status','BLOCKED')}**，输出 `{fp8.get('dtype')}`；400 个 decoder FP8Linear 加载一致性诊断：`{facts.get('loading_info')}`。

模型结构为 Qwen3_5ForConditionalGeneration，64 个 decoder layers，Full Attention 在 3,7,...,63；24 Q heads / 4 KV heads / D=256。目标 Layer 3 的真实 projection 类型、参数 dtype/shape 在 {link('P3_suite.json')} 的 model.projections。

Cache 为 DynamicCache，`cache.layers[3].keys/values`；默认 get_seq_length() 从首个 Full Attention Layer 3 读取。当前源码没有显式 cache_position 参数；runner 显式传 position_ids，并校验所有 Full Attention 层长度。源码摘录与原文件哈希在环境产物中。

显存预检读取 safetensors 文件头，未加载权重：参数合计 {plan.get('weights',{}).get('parameter_bytes',0)/2**30:.3f} GiB；自动选择常驻 {plan.get('selected_resident_layers')} 层，GPU0 参数 {plan.get('resident_parameter_bytes',0)/2**30:.3f} GiB、CPU 参数 {plan.get('cpu_parameter_bytes',0)/2**30:.3f} GiB、最大暂存模块 {plan.get('largest_staged_module_bytes',0)/2**30:.3f} GiB、运行预留 {plan.get('workspace_reserve_bytes',0)/2**30:.3f} GiB。预留只是可配置预算，不能保证任意上下文均无 OOM。

完整权重常驻是否可行：{plan.get('fully_resident_fits')}。GPU1 既有计算进程：`{plan.get('foreign_gpu1_processes')}`。严格部署预检结果见 {link('strict_placement.json')}；整个模型严格模式会在分配权重前阻止不满足条件的运行。

加载器修复 Transformers 5.19.0 中官方排除项 mlp.gate 对 gate_proj 的前缀误匹配，保留官方 FP8 权重和 scales，不修改 site-packages。kernel 版本固定 kernels==0.17.0。来源：[官方固定 Revision](https://huggingface.co/Qwen/Qwen3.8-27B-FP8/tree/{facts.get('revision','main')})。

复现：`.venv/bin/python scripts/check_environment.py`、`.venv/bin/python scripts/check_placement.py`、`.venv/bin/python scripts/probe_fp8.py`。
''')
    rows = '\n'.join(f"| {c['length']} | {c['case']} | {c['max_abs']:.8g} | {c['mean_abs']:.8g} | {c['max_relative']:.8g} | {c['rms']:.8g} | {'PASS' if c['pass'] else 'FAIL'} |" for c in p1.get('cases',[]) if 'length' in c)
    save('P1_attention.md', f'''# P1：{status.get('P1','BLOCKED')}

种子 20261104；BF16、B=1、24/4 heads、D=256。独立 FP32 分组 matmul/softmax 参考，不调用被测 SDPA，TF32 禁用。每元素容差 atol=0.02 / rtol=0.02；相对误差分母最小 1e-3，近零值的最大相对误差不单独作为失败依据。

| 长度 | 场景 | Max abs | Mean abs | Max relative | RMS | 状态 |
|---|---|---|---|---|---|---|
{rows}

覆盖 GQA 6:1、单 token Decode 不能用左上对齐 mask、Prefill、增量追加、矩形 causal mask、布尔和 additive padding、全遮罩行。原始结果 {link('P1_attention.json')}，日志 {link('P1_attention.log')}。

复现：`.venv/bin/python scripts/test_attention.py`。
''')
    save('P2_emulator.md', f'''# P2：{status.get('P2','BLOCKED')}

实际 GPU0 → pinned CPU → GPU1 → pinned CPU → GPU0。Prefill={p2.get('prefill')}，连续 Decode={p2.get('steps')}，最终长度={p2.get('final_cache',{}).get('sequence_length')}。Prefill 初始化一次，Decode 每步只传 Q_new/K_new/V_new，历史 KV 由 GPU1 管理。

四段传输和 kernel 统计：
```json
{json.dumps(p2.get('metrics',{}),indent=2)}
```

预分配 pinned pool {p2.get('pinned_bytes')} bytes，is_pinned={p2.get('pinned')}。预热后/循环结束显存分别为 `{p2.get('warm_memory')}` / `{p2.get('final_memory')}`。每步误差和显存见 {link('P2_emulator.json')}，所有命令和拷贝见 {link('P2_emulator.jsonl')}。

Backend 通过 handle 隔离 KV；重复命令 ID、长度/位置、无效 handle、ownership、shape/dtype/mask、容量、错误回滚和非默认 stream 有测试。Runtime 现在还校验 command ID、handle、owner 和 completion length，拒绝错误 Backend 响应。释放过程即使 transport.close 报错也释放缓存管理器。

每个 Decode 逻辑负载 28 KiB、四段 copy 56 KiB；不包含 CPU 权重 offload，不是 PCIe 线速。non_blocking copy 配合专用 streams/events，但当前命令同步执行，不宣称重叠。复现：`.venv/bin/python scripts/test_emulator.py --steps 128`。
''')
    run_rows = '\n'.join(f"| {r['request_id']} | {r['prompt_tokens']} | {r['decode_steps']} | {r['remote_calls']} | {r['remote_outputs_used']} | {r['decode_seconds']:.3f} |" for r in p3.get('runs',[]))
    comp_rows = []
    for c in p3.get('comparisons',[]):
        maxima = [max(s[k]['max_abs'] for s in c['steps']) for k in ('raw_o','attention_output','decoder_output','logits')]
        comp_rows.append(f"| {c['case']} | {c['mode']} | {c['status']} | " + ' | '.join(f'{v:.8g}' for v in maxima) + ' |')
    save('P3_qwen_integration.md', f'''# P3：{status.get('P3','BLOCKED')}

功能及数值状态 **{audit.get('functional_status','未验证')}**；严格部署状态 **{audit.get('deployment_status','未验证')}**。判断来自本次预检、真实 device_map 和 trace，未写死为 PASS/BLOCKED。

{p3.get('exception','')}

| 运行 | Prompt tokens | Decode steps | GPU1 kernel 次数 | 使用远端 O 次数 | Decode 秒 |
|---|---|---|---|---|---|
{run_rows}

| 上下文 | 模式 | 状态 | Max abs 原始 O | Attention 输出 | Decoder 输出 | Logits |
|---|---|---|---|---|---|---|
{chr(10).join(comp_rows)}

相同 Prompt/token IDs 写入每个运行；固定序列逐级误差保存四项指标，原始 O 同步对比 atol=0.02/rtol=0.02，完整模型跨运行 atol=0.25/rtol=0.02；没有为复跑放宽。Greedy 序列不一致现在会导致失败。每次运行验证 64 个 decoder 的输出设备为 GPU0，卸载的权重仅存于 CPU；逐步检查所有 Full Attention 层长度。Adapter 退出后必须恢复原 config。

P3 使用本机 Qwen forward 的可注册 Attention interface，GPU1 O 在 reshape、sigmoid gate 和 o_proj 前返回。Prefill 保持原生执行。本阶段保留本地完整 KV 用于 Shadow 验证，不是 P4。

GPU1 既有进程：`{audit.get('foreign_gpu1_processes')}`。CPU 权重 offload={audit.get('cpu_weight_offload')}；CPU offload 流量不计入 PNM 四段 copy。系统条件不满足时不声称严格部署完成。显存峰值现在每次运行重新计数，记录于各 runs[].gpu_memory。

源码 Git commit `{p3.get('provenance',{}).get('git_commit')}`，分支 `{p3.get('provenance',{}).get('git_branch')}`；run_id `{p3.get('provenance',{}).get('run_id')}`。执行前后 source SHA256 必须一致，审计时还要与当前代码匹配。任何缺失的 transfer/kernel/free、串用其他运行日志、位置重复或来源版本变更都不能通过。

证据：{link('P3_suite.log')}、{link('P3_suite.json')}、{link('P3_suite.jsonl')}、{link('P3_suite_gpu.jsonl')}。复现：`.venv/bin/python scripts/run_qwen.py --suite --output {raw}/P3_suite.json`。
''')
    save('ACCEPTANCE.md', f'''# P0–P3 验收

**总体 {audit.get('status','BLOCKED')}；功能与数值 {audit.get('functional_status','未验证')}。**

| 阶段 | 状态 | 报告 |
|---|---|---|
| P0 | {status.get('P0','BLOCKED')} | [环境与加载](P0_environment.md) |
| P1 | {status.get('P1','BLOCKED')} | [独立 Attention](P1_attention.md) |
| P2 | {status.get('P2','BLOCKED')} | [Runtime 与模拟器](P2_emulator.md) |
| P3 | {status.get('P3','BLOCKED')} | [完整模型](P3_qwen_integration.md) |

完整模型共 {len(p3.get('runs',[]))} 次运行，GPU1 Attention {audit.get('remote_tokens')} 步，其中 {audit.get('remote_output_tokens')} 步真正使用 GPU1 O。Greedy 对比一致={audit.get('all_greedy_sequences_match')}。自动化测试 {audit.get('unit_tests')} 项通过。审计错误：{audit.get('error','无')}。

严格部署尚存条件：CPU 权重 offload={audit.get('cpu_weight_offload')}；GPU1 无其他计算进程={audit.get('gpu1_compute_exclusive')}。实际参数 {plan.get('weights',{}).get('parameter_bytes',0)/2**30:.3f} GiB，GPU0 运行时余量和占用见 P0 报告。没有停止服务器现有服务，也没有改变官方权重量化格式。

原始审计 {link('acceptance.json')} / {link('acceptance.log')}；完整命令与退出码 {link('commands.json')}，依赖 {link('environment.lock.txt')}。执行代码版本 `{audit.get('execution_commit')}`；分支 `{audit.get('execution_branch')}`。

复现整个验证：`.venv/bin/python scripts/validate_all.py --output-dir reports/new-run`。严格部署检查：`.venv/bin/python scripts/check_placement.py --strict-deployment`，不满足时退出 2；正常功能审计退出 0，证据错误退出 1；`verify_results.py --strict-deployment` 可要求部署条件也必须通过。

当前范围仍为 P0–P3。下一步在满足容量与 GPU1 独占条件的设备上复跑同一入口；之后再单独实施 P4 Cache bridge。此原型的性能不代表真实 ASIC/DMA。
''')


if __name__ == '__main__':
    main()
