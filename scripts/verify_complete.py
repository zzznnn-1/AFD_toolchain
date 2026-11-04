"""Audit the complete simulator plan, retaining real-hardware deployment blockers."""
import _bootstrap
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from pnm.acceptance import require,verify_provenance,verify_trace
from pnm.evidence import source_manifest,timestamp,sha256
from pnm.validation import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--artifact-dir',default='reports/complete/raw');a=p.parse_args()
    root=Path(a.artifact_dir);result={'status':'FAIL','verified_at':timestamp()}
    try:
        def read(name): return json.loads((root/name).read_text())
        manifest=source_manifest(_bootstrap.ROOT)
        env,fp8,p1,p2,model,stress,tests,perf=[read(n) for n in ['environment.json','fp8_probe.json',
            'P1_attention.json','P2_emulator.json','P4_suite.json','P5_stress.json','unit_tests.json','P5_performance.json']]
        for obj in (env,fp8,p1,p2,model,stress,tests,perf): verify_provenance(obj['provenance'],manifest)
        require(all(x.get('status')=='PASS' for x in (fp8,p1,p2,model,stress,tests,perf)),'one phase did not pass')
        require(len(env['gpus'])>=2 and fp8['finite'],'P0 hardware/kernel missing')
        require(all(c['pass'] for c in p1['cases']),'P1 numerical failure')
        for case in ('prefill','decode','append'):
            require({c['length'] for c in p1['cases'] if c.get('case')==case}>={1,16,128,1024,4096},'P1 coverage missing')
        require(p2['steps']>=128 and all(s['pass'] for s in p2['steps_detail']),'P2 failed')
        cfg=model['configuration'];runs=model['runs']
        require(set(cfg['contexts'])=={4,16,128,1024,4096,8192} and cfg['decode_steps']>=128,'P4 contexts/steps incomplete')
        require(model['model']['model_id']=='Qwen/Qwen3.8-27B-FP8' and model['model']['fp8_linear_count']==400,'wrong model')
        require(not any(model['model']['loading_info'].values()),'weight load mismatch')
        require(all(d not in (1,'cuda:1') for d in model['model']['hf_device_map'].values()),'model weights on GPU1')
        lookup={r['request_id']:r for r in runs}
        strict=[r for r in runs if r['mode']=='strict_remote']
        require(len(strict)==8,'missing strict/greedy runs')
        for length in cfg['contexts']:
            local,remote=lookup[f'L{length}-local'],lookup[f'L{length}-strict']
            require(local['input_tokens']==remote['input_tokens'],'teacher forcing differs')
            require(local['prompt_token_ids']==remote['prompt_token_ids'],'prompt differs')
            require(local['target_local_kv_bytes']==(length+cfg['decode_steps'])*4096,'local baseline cache not complete')
        for r in runs:
            require(r['adapter_restored'],'adapter leaked')
            require(len(r['decoder_execution_devices'])==64 and set(r['decoder_execution_devices'].values())=={'cuda:0'},'model execution moved device')
        for r in strict:
            require(r['remote_outputs_used']==r['remote_calls']==r['decode_steps'],'strict O not used')
            require(r['target_local_kv_bytes']==0,'target history remains on GPU0')
            require(r['decode_qkv']['k']['shape']==[1,4,1,256],'strict adapter received historical KV')
            require(len(r['position_records'])==r['decode_steps'],'ownership/position steps missing')
            for record in [r['prefill_ownership'],r['final_ownership']]+[p['ownership'] for p in r['position_records']]:
                require(record['local_kv_bytes']==0 and record['keys_is_none'] and record['values_is_none'],'bridge owns tensors')
                require(record['prefill_kv_released'] and not record['poisoned'] and not record['pending'],'bridge release/commit invalid')
                require(record['retired_prefill_kv_bytes']==r['prompt_tokens']*4096,'prefill KV removal bytes wrong')
            for i,position in enumerate(r['position_records']):
                require(position['position']==r['prompt_tokens']+i,'strict position wrong')
                require(position['native_length']==position['remote_length']==position['position']+1,'strict global length wrong')
            if 'matches_local_generated_tokens' in r: require(r['matches_local_generated_tokens'],'strict greedy differs')
        require(len(model['comparisons'])==8,'numerical comparison coverage missing')
        for cmp in model['comparisons']:
            require(cmp['status']=='PASS' and cmp['prefill']['max_abs']==0,'prefill/comparison differs')
            for row in cmp['steps']:
                require(all(row[k]['pass'] and row[k]['finite'] for k in ('raw_o','attention_output','decoder_output','logits')),'layer/logit mismatch')
        trace=[json.loads(line) for line in (root/'P4_suite.jsonl').read_text().splitlines()]
        counts,summary=verify_trace(trace,model['provenance']['run_id'],runs)
        require(summary['counts']==model['metrics']['counts'] and summary['bytes']==model['metrics']['bytes'],'trace summary mismatch')
        expected_faults={'device_reset','device_lost','cancel','timeout_before','timeout_after','duplicate_completion','transfer_error'}
        require({f['case'] for f in model['faults']}==expected_faults,'model faults missing')
        require(all(f['status']=='PASS' and not f['returned_model_result'] and f['adapter_restored'] for f in model['faults']),'model failed to stop on remote error')
        require(len(model['interventions'])==1 and model['interventions'][0]['logit_delta']['max_abs']>0,'no causal remote-output proof')
        require(stress['cycles']>=16 and stress['steps']>=128 and stress['prefill']>=4096,'stress workload too small')
        require(all(c['active_handles']==0 for c in stress['cycles_detail']),'stress handle leak')
        require(stress['device']['pool_resizes']==0 and stress['gpu1_allocated_after_close']==0,'transport/device resource leak')
        require({r['context'] for r in perf['rows'] if r['mode']=='strict_remote'}>={128,1024,4096,8192},'performance contexts missing')
        for row in perf['rows']:
            require(row['token_ms']['count']>0 and row['token_ms']['p99']>=row['token_ms']['p50'],'invalid percentiles')
            if row['mode']=='strict_remote':
                require(row['copy_bytes_per_token']==[57344],'decode copy bytes wrong')
                require(row['qkv_prepare_cuda_ms']['count']>0 and row['offload_wall_ms']['count']>0,'QKV/offload timing missing')
                require({'runtime_prepare','queue','kv_append','attention','cuda:0_d2h','cuda:1_h2d','cuda:1_d2h','cuda:0_h2d'}<=set(row['stages_ms']),'timing stages missing')
        xml=ET.parse(root/'unit_tests.xml').getroot();suites=list(xml) if xml.tag=='testsuites' else [xml]
        require(all(int(s.get(k,0))==0 for s in suites for k in ('failures','errors','skipped')),'unit test failure/skip')
        count=sum(int(s.get('tests',0)) for s in suites)
        require(count==tests['collected'] and count>0,'unit test count mismatch')
        external=read('P6_external.json')
        deployment='PASS' if not any(d=='cpu' for d in model['model']['hf_device_map'].values()) and model['preflight']['gpu1_compute_exclusive'] else 'BLOCKED'
        result.update(status='BLOCKED' if deployment=='BLOCKED' or external['status']=='BLOCKED' else 'PASS',
            simulator_status='PASS',functional_stages={f'P{i}':'PASS' for i in range(6)},
            stages={'P0':'PASS','P1':'PASS','P2':'PASS','P3':deployment,'P4':'PASS','P5':'PASS','P6':external['status']},
            deployment_status=deployment,unit_tests=count,model_runs=len(runs),command_counts=counts,
            strict_remote_tokens=sum(r['remote_outputs_used'] for r in strict),
            all_remote_output_tokens=sum(r['remote_outputs_used'] for r in runs),
            faults=len(model['faults']),stress_tokens=stress['cycles']*stress['steps'],
            performance_rows=len(perf['rows']),execution_commit=model['provenance']['git_commit'],
            execution_branch=model['provenance']['git_branch'],source_sha256=manifest,
            artifact_sha256={name:sha256(root/name) for name in ['environment.json','fp8_probe.json','P1_attention.json',
                'P2_emulator.json','P4_suite.json','P4_suite.jsonl','P5_stress.json','P5_performance.json','P5_performance.csv',
                'unit_tests.json','unit_tests.xml','P6_external.json']})
        code=0
    except Exception as exc:
        result['error']=repr(exc);code=1
    write_json(root/'complete_acceptance.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_sha256','artifact_sha256')},indent=2))
    return code


if __name__=='__main__': raise SystemExit(main())
