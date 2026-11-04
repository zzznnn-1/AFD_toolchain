import _bootstrap
import argparse
import torch
from pnm.backends.gpu_kernel import attend
from pnm.validation import reference_attention, check_close, write_json
from pnm.evidence import provenance, finish_provenance


@torch.inference_mode()
def run(lengths, device="cuda:0", seed=20261104):
    evidence = provenance(_bootstrap.ROOT)
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    results = []
    for length in lengths:
        q = torch.randn(1,24,length,256,device=device,dtype=torch.bfloat16)
        k = torch.randn(1,4,length,256,device=device,dtype=torch.bfloat16)
        v = torch.randn_like(k)
        for name, query, position in [("prefill",q,0),("decode",q[:,:,-1:],length-1)]:
            actual = attend(query,k,v,position)
            ref = reference_attention(query,k,v,position)
            stats = check_close(actual,ref,atol=0.02,rtol=0.02)
            results.append(dict(length=length,case=name,**stats))
            print(length,name,stats,flush=True)
        qnew = torch.randn(1,24,1,256,device=device,dtype=torch.bfloat16)
        knew = torch.randn(1,4,1,256,device=device,dtype=torch.bfloat16)
        vnew = torch.randn_like(knew)
        ka, va = torch.cat([k,knew],2), torch.cat([v,vnew],2)
        stats = check_close(attend(qnew,ka,va,length),reference_attention(qnew,ka,va,length))
        results.append(dict(length=length,case="append",**stats))
    # Rectangular multi-query chunk and padding/additive masks; each has an
    # independent position-based reference instead of SDPA's is_causal flag.
    q = torch.randn(1,24,3,256,device=device,dtype=torch.bfloat16)
    k = torch.randn(1,4,16,256,device=device,dtype=torch.bfloat16)
    v = torch.randn_like(k)
    mask = torch.ones(1,1,3,16,device=device,dtype=torch.bool)
    mask[...,2:5] = False
    for name,m in [("boolean_padding",mask),("additive_padding",torch.zeros_like(mask,dtype=torch.float32).masked_fill(~mask,-torch.inf)),("rectangular",None)]:
        stats = check_close(attend(q,k,v,13,m),reference_attention(q,k,v,13,m))
        results.append(dict(length=16,case=name,**stats))
    # A sentinel catches the common 1xS upper-left decode error decisively.
    q.zero_(); k.zero_(); v.copy_(torch.arange(16,device=device).view(1,1,16,1))
    o = attend(q[:,:,:1],k,v,15)
    torch.testing.assert_close(o,torch.full_like(o,7.5),atol=0,rtol=0)
    results.append({"case":"decode_upper_left_regression", "pass":True})
    finish_provenance(evidence, _bootstrap.ROOT)
    return {"status":"PASS", "provenance":evidence, "seed":seed, "device":device, "dtype":"bfloat16", "reference":"FP32 explicit grouped matmul/softmax", "cases":results}


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--lengths',type=int,nargs='+',default=[1,16,128,1024,4096])
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--output',default='reports/raw/P1_attention.json')
    a=p.parse_args()
    write_json(a.output,run(a.lengths,a.device))
