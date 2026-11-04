import _bootstrap
import torch
import argparse
from transformers.integrations.finegrained_fp8 import FP8Linear
from pnm.evidence import provenance, finish_provenance
from pnm.validation import write_json


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='reports/raw/fp8_probe.json')
    args=parser.parse_args()
    evidence=provenance(_bootstrap.ROOT)
    torch.manual_seed(20261104)
    m=FP8Linear(256,256,block_size=(128,128)).cuda(0)
    with torch.no_grad():
        m.weight.copy_(torch.randn(256,256,device='cuda:0').to(torch.float8_e4m3fn))
        m.weight_scale_inv.fill_(0.01)
        y=m(torch.randn(1,1,256,device='cuda:0',dtype=torch.bfloat16))
        assert torch.isfinite(y).all()
        print('FP8_LINEAR_PASS',list(y.shape),str(y.dtype),str(y.device))
        print([(n,str(p.dtype),list(p.shape)) for n,p in m.named_parameters()])
        finish_provenance(evidence,_bootstrap.ROOT)
        write_json(args.output,{'status':'PASS','provenance':evidence,'shape':list(y.shape),'dtype':str(y.dtype),
                              'device':str(y.device),'finite':bool(torch.isfinite(y).all())})
