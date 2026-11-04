import _bootstrap
import argparse
import json
import traceback
import faulthandler
import signal
from pathlib import Path
import torch
from pnm.models.qwen38.loader import load_model
from pnm.models.qwen38.runner import Runner
from pnm.models.qwen38.compare import compare_runs
from pnm.backends.gpu_emulator import GPUEmulator
from pnm.runtime import Runtime
from pnm.runtime.metrics import Metrics
from pnm.validation import write_json,check_close


def main():
    faulthandler.register(signal.SIGUSR1)
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/pnm.json')
    p.add_argument('--mode',choices=['local','shadow_compare','shadow_remote'],default='shadow_remote')
    p.add_argument('--model-path')
    p.add_argument('--steps',type=int,default=32)
    p.add_argument('--prompt')
    p.add_argument('--prompt-tokens',type=int)
    p.add_argument('--resident-layers',type=int)
    p.add_argument('--fully-resident',action='store_true')
    p.add_argument('--suite',action='store_true',help='teacher-forcing + greedy; short/512/4096 contexts')
    p.add_argument('--output',default='reports/raw/P3_run.json')
    a=p.parse_args()
    config=json.loads(Path(a.config).read_text())
    if a.resident_layers is not None: config['resident_layers']=a.resident_layers
    config['fully_resident']=a.fully_resident
    torch.manual_seed(config['seed'])
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False
    metrics=Metrics(a.output.replace('.json','.jsonl'))
    result={'status':'RUNNING','configuration':config,'runs':[],'comparisons':[]}
    runtime=None
    try:
        model,tokenizer,facts=load_model(config,a.model_path)
        result['model']=facts
        write_json(a.output,result)
        runtime=Runtime(GPUEmulator(config['source_device'],config['emulator_device'],config['capacity'],metrics))
        runner=Runner(model,tokenizer,runtime,config['capacity'],config['attention_atol'],config['attention_rtol'])
        if a.suite:
            for label,length,steps in [('short',None,32),('medium',512,128),('context4096',4096,32)]:
                prompt=runner.prompt(length=length)
                # Native local greedy provides a fixed teacher-forcing sequence.
                local,lc,lp=runner.run(prompt,'local',steps,request_id=label+'-local')
                result['runs'].append(local)
                for mode in ['shadow_compare','shadow_remote']:
                    remote,rc,rp=runner.run(prompt,mode,steps,fixed_tokens=local['input_tokens'],request_id=label+'-'+mode)
                    comparison=compare_runs(lc,rc,config['logits_atol'],config['logits_rtol'])
                    comparison.update(case=label,mode=mode,prefill=check_close(rp,lp,atol=0,rtol=0))
                    result['comparisons'].append(comparison)
                    result['runs'].append(remote)
                    write_json(a.output,result)
                    if comparison['status']!='PASS': raise AssertionError(f'{label} {mode} teacher forcing mismatch')
                    del rc
                if label in ['short','medium']:
                    greedy,_,_=runner.run(prompt,'shadow_remote',steps,request_id=label+'-remote-greedy',capture=False)
                    greedy['matches_local_generated_tokens']=greedy['input_tokens']==local['input_tokens']
                    result['runs'].append(greedy)
                    write_json(a.output,result)
                del lc
        else:
            prompt=runner.prompt(a.prompt,a.prompt_tokens)
            summary,_,_=runner.run(prompt,a.mode,a.steps,capture=False)
            result['runs'].append(summary)
        result['status']='PASS'
    except Exception as e:
        result.update(status='FAIL',exception=repr(e),traceback=traceback.format_exc())
        raise
    finally:
        if runtime: runtime.close()
        result['metrics']=metrics.summary()
        metrics.close()
        write_json(a.output,result)
        print('RESULT',result['status'],a.output,flush=True)


if __name__=='__main__': main()
