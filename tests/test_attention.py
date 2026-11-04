import pytest
import torch
from pnm.backends.gpu_kernel import attend
from pnm.validation import reference_attention, check_close


@pytest.mark.parametrize('length',[1,16,128])
def test_gqa_and_causal(length):
    torch.manual_seed(20261104)
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
    q=torch.randn(1,24,length,256,device=device,dtype=torch.bfloat16)
    k=torch.randn(1,4,length,256,device=device,dtype=torch.bfloat16)
    v=torch.randn_like(k)
    check_close(attend(q,k,v),reference_attention(q,k,v))


def test_group_mapping():
    q=torch.zeros(1,24,1,256,dtype=torch.bfloat16)
    k=torch.zeros(1,4,16,256,dtype=torch.bfloat16)
    v=torch.arange(4,dtype=torch.bfloat16).view(1,4,1,1).expand_as(k)
    actual=attend(q,k,v,15)
    expected=torch.arange(24).div(6,rounding_mode='floor').view(1,24,1,1).expand_as(q).to(q.dtype)
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)


def test_masked_out_rows():
    q=torch.ones(1,24,1,256,dtype=torch.bfloat16)
    k=torch.ones(1,4,16,256,dtype=torch.bfloat16)
    mask=torch.zeros(1,1,1,16,dtype=torch.bool)
    o=attend(q,k,k,15,mask)
    assert torch.isfinite(o).all() and not o.any()
