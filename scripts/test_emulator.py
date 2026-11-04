import _bootstrap
import argparse
import gc
import torch
from pnm.protocol import CacheSpec, PNMError
from pnm.runtime import Runtime
from pnm.runtime.metrics import Metrics
from pnm.backends.gpu_emulator import GPUEmulator
from pnm.validation import reference_attention, check_close, write_json


def memory():
    return {str(i): {"allocated":torch.cuda.memory_allocated(i), "reserved":torch.cuda.memory_reserved(i),
                     "max_allocated":torch.cuda.max_memory_allocated(i)} for i in [0,1]}


@torch.inference_mode()
def run(steps=128, prefill=128, output="reports/raw/P2_emulator.json"):
    torch.manual_seed(20261104)
    torch.backends.cuda.matmul.allow_tf32=False
    capacity=prefill+steps+16
    metrics=Metrics(output.replace('.json','.jsonl'))
    backend=GPUEmulator(max_tokens=capacity,metrics=metrics)
    transport_bytes=backend.transport.allocated_bytes
    pinned=backend.transport.all_pinned
    records=[]
    with Runtime(backend) as rt:
        h=rt.alloc_cache('p2',3,CacheSpec(capacity))
        k=torch.randn(1,4,capacity,256,device='cuda:0',dtype=torch.bfloat16)
        v=torch.randn_like(k)
        rt.init_cache(h,k[:,:,:prefill],v[:,:,:prefill])
        initial=rt.get_cache_info(h)
        before=None
        for step in range(steps):
            pos=prefill+step
            q=torch.randn(1,24,1,256,device='cuda:0',dtype=torch.bfloat16)
            o=rt.append_and_attend(h,q,k[:,:,pos:pos+1],v[:,:,pos:pos+1],pos,command_id=f'decode-{step}')
            ref=reference_attention(q,k[:,:,:pos+1],v[:,:,:pos+1],pos)
            stats=check_close(o,ref)
            info=rt.get_cache_info(h)
            assert info.sequence_length==pos+1
            records.append(dict(step=step,length=info.sequence_length,memory=memory(),**stats))
            if step==min(15,steps-1):
                before=memory()
            if step in [0,31,63,95,127]:
                print('decode',step,'length',info.sequence_length,'max_abs',stats['max_abs'],flush=True)
        after=memory()
        assert after['1']['allocated'] <= before['1']['allocated']+1024*1024, (before,after)
        assert backend.transport.allocated_bytes==transport_bytes
        # Rejected duplicate cannot append a token; length is unchanged.
        try:
            rt.append_and_attend(h,q,k[:,:,:1],v[:,:,:1],prefill+steps,command_id='decode-0')
            raise AssertionError('duplicate accepted')
        except PNMError as e:
            assert e.code=='DUPLICATE_COMMAND'
        assert rt.get_cache_info(h).sequence_length==prefill+steps
        final=rt.get_cache_info(h)
        rt.free_cache(h)
        assert not backend.manager.entries
    gc.collect()
    summary=metrics.summary()
    metrics.close()
    # Each leg includes one prefill and new QKV only for decode.
    expected_input=prefill*4*256*2*2+steps*(24+4+4)*256*2
    assert summary['bytes']['cuda:0_d2h']==expected_input
    assert summary['bytes']['cuda:1_h2d']==expected_input
    assert summary['bytes']['cuda:1_d2h']==steps*24*256*2
    assert summary['bytes']['cuda:0_h2d']==steps*24*256*2
    result={'status':'PASS','seed':20261104,'steps':steps,'prefill':prefill,'pinned':pinned,
            'pinned_bytes':transport_bytes,'initial_cache':initial.__dict__,'final_cache':final.__dict__,
            'warm_memory':before,'final_memory':after,'metrics':summary,'steps_detail':records}
    write_json(output,result)
    print('P2 PASS',summary,flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--steps',type=int,default=128)
    p.add_argument('--prefill',type=int,default=128)
    p.add_argument('--output',default='reports/raw/P2_emulator.json')
    a=p.parse_args()
    run(a.steps,a.prefill,a.output)
