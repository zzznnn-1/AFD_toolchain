import _bootstrap
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',default='reports/complete');a=p.parse_args()
    root=Path(a.output_dir);raw=root/'raw';root.mkdir(parents=True,exist_ok=True)
    def read(name):
        path=raw/name
        return json.loads(path.read_text()) if path.exists() else {}
    def write(name,text): (root/name).write_text(text+'\n')
    env,fp8,p1,p2,m,perf,stress,ext,audit=[read(n) for n in ['environment.json','fp8_probe.json','P1_attention.json',
        'P2_emulator.json','P4_suite.json','P5_performance.json','P5_stress.json','P6_external.json','complete_acceptance.json']]
    if not audit and read('pipeline.json').get('status')=='FAIL':
        audit={'status':'FAIL','error':'pipeline failed; see raw/commands.json and phase logs'}
    functional=audit.get('functional_stages',{});stages=audit.get('stages',{})
    status=lambda s:functional.get(s,'BLOCKED')
    facts=m.get('model',{});plan=m.get('preflight',{})
    write('P0_environment.md',f'''# P0：{status('P0')}

环境时间 {env.get('time')}，Python {env.get('python')}，PyTorch CUDA {env.get('torch_cuda')}。
GPU：`{env.get('gpus')}`；Peer Access：`{env.get('peer_access')}`。
软件版本：`{json.dumps(env.get('versions',{}))}`。

完整模型 `{facts.get('model_id')}`，Revision `{facts.get('revision')}`，400 个 FP8Linear 检查结果 `{facts.get('loading_info')}`。FP8 probe={fp8.get('status')}，激活输出 `{fp8.get('dtype')}`。

预检选择 {plan.get('selected_resident_layers')} 个 decoder 层常驻 GPU0，CPU 参数字节 {plan.get('cpu_parameter_bytes')}；GPU1 既有计算进程 `{plan.get('foreign_gpu1_processes')}`。模型权重不进入 GPU1。物理独占/全量 GPU0 常驻状态 {audit.get('deployment_status','未验证')}，与 P4 的 KV 唯一所有权区分。

原始记录：[environment.json](raw/environment.json)、[FP8 probe](raw/fp8_probe.json)、[模型完整元数据](raw/P4_suite.json)、[版本锁](raw/environment.lock.txt)。topology、CPU/NUMA/RAM/磁盘和驱动/nvcc 原始命令在 environment.json。
''')
    rows='\n'.join(f"| {c['length']} | {c['case']} | {c['max_abs']:.8g} | {c['mean_abs']:.8g} | {c['max_relative']:.8g} | {c['rms']:.8g} | {c.get('cosine_similarity',0):.8g} |" for c in p1.get('cases',[]) if 'length' in c)
    write('P1_attention.md',f'''# P1：{status('P1')}

固定种子 20261104，BF16/24 Q heads/4 KV heads/D=256/B=1；独立 FP32 分组 matmul/softmax 参考，TF32 禁用。BF16 atol=0.02/rtol=0.02；单独 FP32 回归要求 atol=1e-4/rtol=1e-5，mean_abs<1e-5。相对误差分母最小 1e-3。

| 长度 | 场景 | Max abs | Mean abs | Max relative | RMS | Cosine |
|---|---|---|---|---|---|---|
{rows}

覆盖 GQA、增量追加、矩形 causal mask、bool/additive mask、全遮罩行和错误的 Decode 左上对齐 mask。证据：[P1_attention.json](raw/P1_attention.json)、[单元测试](raw/unit_tests.xml)。
''')
    write('P2_emulator.md',f'''# P2：{status('P2')}

GPU0→Pinned Host→GPU1→Pinned Host→GPU0 完整路径，连续 {p2.get('steps')} 步通过独立参考；Prefill {p2.get('prefill')}，最终长度 {p2.get('final_cache',{}).get('sequence_length')}。历史只初始化一次；物理初始化分 256-token chunks，固定 pinned pool 复用，KV 管理器一次分配容量，Decode 不使用 torch.cat。

四段 copy/kernel 原始统计：`{json.dumps(p2.get('metrics',{}))}`。

协议包含版本、generation、request/layer/command ID、handle、长度/位置、dtype/layout，以及可序列化 shape/stride/bytes 描述。Reset 使旧 generation Handle 失效，重复 Command ID 拒绝。Timeout Unknown 和重复 completion 使缓存 poison，查询不能自动解除 poison。取消、过期 handle、资源不足、实际 staging 异常都有测试。

证据：[P2 数据](raw/P2_emulator.json)、[逐命令日志](raw/P2_emulator.jsonl)、[测试结果](raw/unit_tests.log)。
''')
    shadow=[c for c in m.get('comparisons',[]) if c['mode']!='strict_remote']
    write('P3_shadow.md',f'''# P3：功能 {status('P3')}；独占部署 {audit.get('deployment_status','未验证')}

首轮完整三模式/32/128-token/4096-context 验收保存在历史 reports/develop。本轮针对共享 Runtime/传输改动，使用当前同一 checkpoint 运行 shadow_compare/shadow_remote 各 16-token 回归，并在 P4 扩展完整上下文固定序列验证。

当前回归：`{[(c['mode'],c['status'],len(c['steps'])) for c in shadow]}`。Prefill 原生，Adapter 退出恢复原配置，64 个 decoder 输出设备均检查为 GPU0。

证据：[当前完整模型结果](raw/P4_suite.json)。GPU0 全权重常驻和 GPU1 物理独占未满足时，P3 总验收继续保留 BLOCKED，不把 CPU offload/共享服务隐去。
''')
    strict=[r for r in m.get('runs',[]) if r['mode']=='strict_remote']
    rows='\n'.join(f"| {r['request_id']} | {r['prompt_tokens']} | {r['decode_steps']} | {r['remote_outputs_used']} | {r['target_local_kv_bytes']} | {r['final_cache']['sequence_length']} |" for r in strict)
    comparisons=[]
    for c in m.get('comparisons',[]):
        if c['mode']!='strict_remote': continue
        maxima=[max(s[k]['max_abs'] for s in c['steps']) for k in ('raw_o','attention_output','decoder_output','logits')]
        comparisons.append(f"| {c['context']} | {c['status']} | "+' | '.join(f'{v:.8g}' for v in maxima)+' |')
    write('P4_strict_remote.md',f'''# P4 Strict Remote KV：{status('P4')}

CacheBridge 安装后原生 Layer 3 keys/values 清空，旧 Prefill K/V weak references 失效；bridge 只保存 metadata。每步对象检查 local_kv_bytes=0，目标层接口只收到当前 K/V `[1,4,1,256]`。没有调用本地历史 Attention，也没有每步重传历史。GPU1 成功返回后才提交全局/Layer 3 长度，其他 63 层保持原生执行。

| 运行 | Prompt | Decode | 使用 GPU1 O | GPU0 目标 KV bytes | 最终长度 |
|---|---|---|---|---|---|
{rows}

固定 Token 序列四级 Max absolute error（完整四项误差与 cosine 保存于 JSON）：

| Context | 状态 | 原始 O | Attention 输出 | Decoder 输出 | Logits |
|---|---|---|---|---|---|
{chr(10).join(comparisons)}

远端 O 因果干预：`{m.get('interventions')}`。该实验在 GPU1 正常计算后故意把返回 O 置零，要求最终 logits 改变；它是依赖路径验证，不作为正确输出样本。

完整模型故障实验：`{m.get('faults')}`。发生异常后不返回模型结果，Adapter 恢复，旧请求失效；不能用本地回退掩盖故障。

Prefill 仍原生构建临时 GPU0 KV，Strict 约束作用于迁移后的 Decode。PyTorch reserved/峰值可能包含 Prefill，因此不用“reserved 立刻下降”证明所有权。实际释放由弱引用、CacheBridge 对象及每步长度检查证明。

证据：[P4_suite.json](raw/P4_suite.json)、[GPU1/传输记录](raw/P4_suite.jsonl)、[运行日志](raw/P4_suite.log)、[GPU 采样](raw/P4_suite_gpu.jsonl)。
''')
    rows=[]
    for r in perf.get('rows',[]):
        if r['mode'] not in ('local','strict_remote'): continue
        t=r['token_ms']
        rows.append(f"| {r['context']} | {r['mode']} | {r['window_tokens']} | {t['count']} | {t['p50']:.3f} | {t['p95']:.3f} | {t['p99']:.3f} | {r['tokens_per_second']:.3f} |")
    write('P5_performance.md',f'''# P5 性能、压力和故障：{status('P5')}

方法：{perf.get('method','未运行')}。表中单位 ms，CPU 权重 offload 会主导 token 延迟。native Prefill 计算与 KV 初始化分别计时；所有首次编译/预热影响如实保留，不宣称模拟器或 ASIC 加速比。

| Context | 模式 | 输出窗口 | 统计样本 | p50 | p95 | p99 | tokens/s |
|---|---|---|---|---|---|---|---|
{chr(10).join(rows)}

每步包括 QKV 准备 CUDA 时间、runtime 命令准备、队列、四段 copy、GPU1 KV append、Attention、端到端 offload、Layer 3 和完整 token 时间。不同 GPU 时钟事件不能直接相加为真实端到端延迟。四段 copy 每 token 为 57344 bytes，命令描述字节单列，不包括 CPU 权重传输，也不等于 PCIe 总线线速。

压力：{stress.get('cycles')} 个请求 × {stress.get('steps')} 步，Prefill={stress.get('prefill')}，所有输出对照独立 FP32 参考。固定 pool 未扩容；warm 后 GPU1 allocated 变化守卫 1 MiB，Host RSS 变化守卫 32 MiB；这些是本次有限压力运行的观测范围。关闭后 GPU1 allocated={stress.get('gpu1_allocated_after_close')} bytes。去重 ID 集合增长是有预算上限的 metadata，不是 KV 泄漏。

Host pool/NUMA：`{perf.get('device')}`。NUMA 数据是 buffer 所在 VMA 的页面分布，不能当作每个 buffer 的精确逐页 NUMA 归属；进程 CPU affinity 在主机上下文记录。没有修改系统 NUMA 策略。

交付：[JSON](raw/P5_performance.json)、[CSV](raw/P5_performance.csv)、[压力结果](raw/P5_stress.json)、[压力日志](raw/P5_stress.log)、[完整模型故障](raw/P4_suite.json)。所有数值均为 GPU 软件模拟，不代表未来 KXP/ASIC 吞吐。
''')
    write('P6_external.md',f'''# P6：{ext.get('status','BLOCKED')}

真实硬件检查时间 {ext.get('time')}；候选节点 `{ext.get('candidate_device_nodes')}`；项目环境包 `{ext.get('python_packages')}`。

已经实现并测试 Backend 插件工厂和带版本/长度/CRC 的命令 envelope，Runtime 与模型不访问 GPU1 内部 KV Tensor。缺少真实驱动插件时明确失败，不自动回退。当前 envelope 不是冻结的硬件 DMA ABI；CUDA pinned buffer 不是已完成驱动注册的内存。

实际阻塞：{ext.get('blocking_reasons')}。
所需输入：{ext.get('required_inputs')}。

原规范明确后置 vLLM；本次仍采用 Transformers，未把服务器其他 vLLM 进程视为本项目集成。真实 ASIC/驱动及 vLLM PNM 接入没有实际运行，不记录 PASS。

原始检测：[P6_external.json](raw/P6_external.json)。接口契约见项目 docs/DRIVER_CONTRACT.md。
''')
    write('ACCEPTANCE.md',f'''# 整体方案执行结果

**软件模拟原型 P0–P5：{audit.get('simulator_status','未验证')}；整体外部交付：{audit.get('status','BLOCKED')}。**

| 阶段 | 功能结果 | 总验收/限制 | 报告 |
|---|---|---|---|
| P0 | {status('P0')} | 环境、官方 FP8、真实 QKV 检查 | [P0](P0_environment.md) |
| P1 | {status('P1')} | GQA/mask/独立参考 | [P1](P1_attention.md) |
| P2 | {status('P2')} | 分块 Host Staging、generation、请求生命周期 | [P2](P2_emulator.md) |
| P3 | {status('P3')} | 独占部署 {audit.get('deployment_status','未验证')} | [P3](P3_shadow.md) |
| P4 | {status('P4')} | Layer 3 GPU0 历史 KV 为零 | [P4](P4_strict_remote.md) |
| P5 | {status('P5')} | 实测统计、压力和完整模型故障 | [P5](P5_performance.md) |
| P6 | {ext.get('status','BLOCKED')} | 真实 ASIC/驱动/ABI 未提供；vLLM 后置 | [P6](P6_external.md) |

模型运行 {audit.get('model_runs')} 次，Strict Remote 使用 GPU1 O {audit.get('strict_remote_tokens')} 步；完整模型故障场景 {audit.get('faults')} 个，压力 Decode {audit.get('stress_tokens')} 步。自动化测试 {audit.get('unit_tests')} 项通过，审计错误 {audit.get('error','无')}。

执行代码 commit `{audit.get('execution_commit')}`，分支 `{audit.get('execution_branch')}`；当前独立工作区在 `/home/jinyuzhe/attention-develop`，主目录仍保留用户的 main 工作区。所有阶段记录源码 SHA256、run_id、版本与随机种子。

入口：`.venv/bin/python scripts/complete_plan.py --output-dir reports/new-complete-run`。完整命令和退出码：[commands.json](raw/commands.json)；依赖：[environment.lock.txt](raw/environment.lock.txt)；最终审计：[complete_acceptance.json](raw/complete_acceptance.json)。

GPU0 全量权重常驻和整卡 GPU1 独占仍受现有服务器限制，不影响已经验证的 GPU1 KV 唯一所有权，但不能据此宣称独占部署完成。真实 ASIC 性能和真实 DMA 也没有通过软件模拟冒充。
''')


if __name__=='__main__': main()
