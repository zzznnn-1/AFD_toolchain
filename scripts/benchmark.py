"""Correctness-checked emulator timings at several history lengths."""
import _bootstrap
import argparse
from pathlib import Path
from test_emulator import run
from pnm.validation import write_json


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--lengths',type=int,nargs='+',default=[16,128,1024,4096])
    p.add_argument('--steps',type=int,default=128)
    p.add_argument('--output-dir',default='reports/raw/benchmark')
    a=p.parse_args()
    Path(a.output_dir).mkdir(parents=True,exist_ok=True)
    rows=[]
    for length in a.lengths:
        d=run(a.steps,length,f'{a.output_dir}/context_{length}.json')
        m=d['metrics']
        rows.append({'history':length,'steps':a.steps,'status':d['status'],'pinned_bytes':d['pinned_bytes'],
                     'attention_ms_per_step':m['milliseconds']['attention']/a.steps,
                     'transfer_bytes':{k:v for k,v in m['bytes'].items() if k.endswith(('d2h','h2d'))},
                     'transfer_ms':{k:v for k,v in m['milliseconds'].items() if k.endswith(('d2h','h2d'))}})
    write_json(f'{a.output_dir}/summary.json',rows)
