"""Full official-model P4/P5 acceptance. All comparisons use actual model executions."""
import _bootstrap
import argparse
import json
import traceback
from pathlib import Path
import torch
from pnm.evidence import provenance,finish_provenance
from pnm.models.qwen38.loader import prepare_model,load_model
from pnm.models.qwen38.runner import Runner
from pnm.models.qwen38.compare import compare_runs
from pnm.backends.registry import create_backend
from pnm.backends.faults import FaultBackend
from pnm.runtime import Runtime
from pnm.runtime.metrics import Metrics
from pnm.protocol import PNMError
from pnm.validation import check_close,errors,write_json
from pnm.performance import summarize,write_csv


def backend_options(config,capacity=None):
    return {'kind':'gpu_emulator','source':config['source_device'],'device':config['emulator_device'],
            'max_tokens':capacity or config['capacity'],'chunk_tokens':config['prefill_chunk_tokens']}


def fault_runs(model,tokenizer,config,root,run_id):
    results=[]
    cases={'device_reset':'STALE_HANDLE','device_lost':'DEVICE_LOST','cancel':'CANCELLED',
           'timeout_before':'TIMEOUT_UNKNOWN','timeout_after':'TIMEOUT_UNKNOWN',
           'duplicate_completion':'INVALID_COMPLETION','transfer_error':'TRANSFER_ERROR'}
    for case,expected in cases.items():
        metrics=Metrics(root/f'fault_{case}.jsonl',run_id)
        backend=create_backend(backend_options(config,256),metrics)
        wrapped=FaultBackend(backend)
        runtime=Runtime(wrapped,metrics)
        runner=Runner(model,tokenizer,runtime,256)
        native=model.model.language_model.layers[3].self_attn.config
        observed={}
        def trigger(step,adapter,cache):
            if step==1:
                observed.update(completed=adapter.remote_calls,ownership=adapter.ownership())
                if case=='device_reset': backend.reset()
                elif case=='device_lost': backend.lose_device()
                elif case=='cancel': runtime.cancel_request('fault-'+case)
                else: wrapped.arm(case)
        try:
            try:
                runner.run(runner.prompt(length=16),'strict_remote',3,request_id='fault-'+case,
                           step_callback=trigger,capture=False)
                raise AssertionError('fault silently returned a model result')
            except PNMError as exc:
                if exc.code!=expected: raise
                if model.model.language_model.layers[3].self_attn.config is not native:
                    raise AssertionError('failed strict request left adapter installed')
                if observed.get('completed')!=1 or observed['ownership']['local_kv_bytes']:
                    raise AssertionError('strict ownership before fault not established')
                results.append({'case':case,'status':'PASS','observed_error':exc.code,
                                'completed_tokens_before_fault':observed['completed'],
                                'ownership_before_fault':observed['ownership'],
                                'device_attention_calls':metrics.summary()['counts'].get('attention',0),
                                'adapter_restored':True,'returned_model_result':False})
                print('FAULT PASS',case,exc.code,flush=True)
        finally:
            runtime.close();metrics.close()
    return results


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/complete.json')
    p.add_argument('--output-dir',default='reports/complete/raw')
    p.add_argument('--contexts',type=int,nargs='+')
    p.add_argument('--steps',type=int)
    a=p.parse_args()
    root=Path(a.output_dir);root.mkdir(parents=True,exist_ok=True)
    config=json.loads(Path(a.config).read_text())
    if a.contexts: config['contexts']=a.contexts
    if a.steps: config['decode_steps']=a.steps
    if config['decode_steps']<16: raise ValueError('at least 16 decode steps required')
    if max(config['contexts'])+config['decode_steps']>config['capacity']: raise ValueError('cache capacity insufficient')
    evidence=provenance(_bootstrap.ROOT)
    result={'status':'RUNNING','provenance':evidence,'configuration':config,
            'runs':[],'comparisons':[],'faults':[],'interventions':[]}
    output=root/'P4_suite.json'
    metrics=Metrics(root/'P4_suite.jsonl',evidence['run_id'])
    runtime=None
    torch.manual_seed(config['seed']);torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False
    code=0
    try:
        path,plan=prepare_model(config)
        result['preflight']=plan
        model,tokenizer,facts=load_model(config,path,plan)
        result['model']=facts
        runtime=Runtime(create_backend(backend_options(config),metrics),metrics)
        runner=Runner(model,tokenizer,runtime,config['capacity'],config['attention_atol'],config['attention_rtol'])
        for length in config['contexts']:
            prompt=runner.prompt(length=length);steps=config['decode_steps']
            local,lc,lp=runner.run(prompt,'local',steps,request_id=f'L{length}-local')
            result['runs'].append(local);write_json(output,result)
            remote,rc,rp=runner.run(prompt,'strict_remote',steps,fixed_tokens=local['input_tokens'],request_id=f'L{length}-strict')
            comparison=compare_runs(lc,rc,config['logits_atol'],config['logits_rtol'])
            comparison.update(context=length,mode='strict_remote',prefill=check_close(rp,lp,atol=0,rtol=0))
            result['comparisons'].append(comparison);result['runs'].append(remote)
            write_json(output,result)
            if comparison['status']!='PASS': raise AssertionError(f'strict numerical mismatch at context {length}')
            del rc
            if length in (4,128):
                greedy_steps=16 if length==4 else steps
                greedy,_,_=runner.run(prompt,'strict_remote',greedy_steps,request_id=f'L{length}-strict-greedy',capture=False)
                greedy['matches_local_generated_tokens']=greedy['input_tokens']==local['input_tokens'][:greedy_steps]
                result['runs'].append(greedy)
                if not greedy['matches_local_generated_tokens']: raise AssertionError('strict greedy sequence differs')
            if length==128:
                for mode in ('shadow_compare','shadow_remote'):
                    shadow,sc,sp=runner.run(prompt,mode,16,fixed_tokens=local['input_tokens'][:16],request_id=f'L128-{mode}')
                    cmp=compare_runs(lc[:16],sc,config['logits_atol'],config['logits_rtol'])
                    cmp.update(context=128,mode=mode,prefill=check_close(sp,lp,atol=0,rtol=0))
                    result['comparisons'].append(cmp);result['runs'].append(shadow)
                    if cmp['status']!='PASS': raise AssertionError('shadow regression mismatch')
                    del sc
                # Corrupt a real returned O deliberately; logits must respond.
                probe_metrics=Metrics(root/'output_intervention.jsonl',evidence['run_id'])
                wrapped=FaultBackend(create_backend(backend_options(config,256),probe_metrics))
                with Runtime(wrapped,probe_metrics) as probe_rt:
                    def perturb(step,adapter,cache):
                        if step==0: wrapped.arm('zero_output')
                    probe_runner=Runner(model,tokenizer,probe_rt,256)
                    probe,pc,_=probe_runner.run(prompt,'strict_remote',2,fixed_tokens=local['input_tokens'][:2],
                                               request_id='output-intervention',step_callback=perturb)
                    delta=errors(pc[0]['logits'],lc[0]['logits'])
                    if not delta['finite'] or delta['max_abs']==0: raise AssertionError('remote O did not affect logits')
                    result['interventions'].append({'status':'PASS','kind':'zero actual backend O after GPU1 computation',
                                                   'logit_delta':delta,'remote_outputs_used':probe['remote_outputs_used']})
                probe_metrics.close();del pc
            del lc
            write_json(output,result)
        result['device']=runtime.describe()
        if result['device']['kv_caches']!=0: raise AssertionError('cache leaked between model requests')
        runtime.close();runtime=None
        result['faults']=fault_runs(model,tokenizer,config,root,evidence['run_id'])
        result['status']='PASS'
    except BaseException as exc:
        result.update(status='FAIL',exception=repr(exc),traceback=traceback.format_exc())
        traceback.print_exc();code=1
    finally:
        try:
            if runtime: runtime.close()
            finish_provenance(evidence,_bootstrap.ROOT)
        except BaseException as exc:
            result.update(status='FAIL',finalization_error=repr(exc));code=1
        metrics.close();result['metrics']=metrics.summary();write_json(output,result)
        if result['status']=='PASS':
            trace=[json.loads(line) for line in (root/'P4_suite.jsonl').read_text().splitlines()]
            performance=summarize(result['runs'],trace,config['warmup_steps'])
            performance.update(status='PASS',provenance=evidence,device=result['device'],hardware=result['preflight']['hardware'])
            write_json(root/'P5_performance.json',performance)
            write_csv(root/'P5_performance.csv',performance)
        print('P4/P5 RESULT',result['status'],output,flush=True)
    return code


if __name__=='__main__': raise SystemExit(main())
