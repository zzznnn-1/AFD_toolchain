import _bootstrap
import argparse
import gc
from pathlib import Path
import torch
from pnm.backends.registry import create_backend
from pnm.protocol import CacheSpec
from pnm.runtime import Runtime
from pnm.runtime.metrics import Metrics
from pnm.evidence import provenance,finish_provenance
from pnm.validation import check_close,reference_attention,write_json


def rss_bytes():
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'): return int(line.split()[1])*1024
    return None


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cycles',type=int,default=16)
    p.add_argument('--steps',type=int,default=128)
    p.add_argument('--prefill',type=int,default=4096)
    p.add_argument('--output',default='reports/complete/raw/P5_stress.json')
    a=p.parse_args()
    if min(a.cycles,a.steps,a.prefill)<1: raise ValueError('positive workload sizes required')
    torch.manual_seed(20261104);torch.backends.cuda.matmul.allow_tf32=False
    evidence=provenance(_bootstrap.ROOT)
    metrics=Metrics(Path(a.output).with_suffix('.jsonl'),evidence['run_id'])
    capacity=a.prefill+a.steps
    backend=create_backend({'kind':'gpu_emulator','max_tokens':capacity,'chunk_tokens':256},metrics)
    pointers={n:p.data_ptr() for n,(p,_) in backend.transport.slots.items()}
    records=[];warm=None
    with Runtime(backend,metrics) as rt:
        for cycle in range(a.cycles):
            k=torch.randn(1,4,capacity,256,device='cuda:0',dtype=torch.bfloat16);v=torch.randn_like(k)
            handle=rt.alloc_cache(f'stress-{cycle}',3,CacheSpec(capacity))
            rt.init_cache(handle,k[:,:,:a.prefill],v[:,:,:a.prefill])
            worst=0.;rms=0.
            for step in range(a.steps):
                pos=a.prefill+step
                q=torch.randn(1,24,1,256,device='cuda:0',dtype=torch.bfloat16)
                output=rt.append_and_attend(handle,q,k[:,:,pos:pos+1],v[:,:,pos:pos+1],pos)
                reference=reference_attention(q,k[:,:,:pos+1],v[:,:,:pos+1],pos)
                error=check_close(output,reference)
                worst=max(worst,error['max_abs']);rms=max(rms,error['rms'])
            if rt.get_cache_info(handle).sequence_length!=capacity: raise AssertionError('incorrect final length')
            rt.free_cache(handle);gc.collect()
            snap={'cycle':cycle,'max_abs':worst,'max_rms':rms,'gpu1_allocated':torch.cuda.memory_allocated(1),
                  'gpu1_reserved':torch.cuda.memory_reserved(1),'host_rss':rss_bytes(),'active_handles':len(rt.handles)}
            records.append(snap)
            if cycle==min(2,a.cycles-1): warm=snap
            if warm and snap['gpu1_allocated']>warm['gpu1_allocated']+1024**2:
                raise AssertionError('GPU1 allocation grows across freed requests')
            if warm and snap['host_rss']>warm['host_rss']+32*1024**2:
                raise AssertionError('host RSS grows >32 MiB beyond warm state')
            if pointers!={n:p.data_ptr() for n,(p,_) in backend.transport.slots.items()}:
                raise AssertionError('pinned buffers were replaced')
            if rt.describe()['kv_caches'] or rt.handles: raise AssertionError('cache leaked')
            print('stress',cycle+1,'/',a.cycles,'max_abs',worst,'allocated',snap['gpu1_allocated'],flush=True)
        device=rt.describe()
    gc.collect()
    finish_provenance(evidence,_bootstrap.ROOT)
    result={'status':'PASS','provenance':evidence,'cycles':a.cycles,'steps':a.steps,'prefill':a.prefill,
            'seed':20261104,'warm_snapshot':warm,'cycles_detail':records,'device':device,
            'gpu1_allocated_after_close':torch.cuda.memory_allocated(1),'metrics':metrics.summary()}
    metrics.close();write_json(a.output,result)


if __name__=='__main__': main()
