from types import SimpleNamespace
import pytest
import torch
from transformers import Qwen3_5TextConfig
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention,Qwen3_5TextRotaryEmbedding
from pnm.models.qwen38.injection import Injection
from pnm.models.qwen38.cache_bridge import RemoteCacheLayer
from pnm.backends.gpu_emulator import GPUEmulator
from pnm.runtime import Runtime
from pnm.protocol import PNMError


def make_fixture():
    config=Qwen3_5TextConfig(hidden_size=256,intermediate_size=512,num_hidden_layers=4,
        num_attention_heads=24,num_key_value_heads=4,head_dim=256,
        layer_types=['linear_attention']*3+['full_attention'])
    config._attn_implementation='sdpa'
    attn=Qwen3_5Attention(config,3).to(device='cuda:0',dtype=torch.bfloat16).eval()
    model=SimpleNamespace(model=SimpleNamespace(language_model=SimpleNamespace(
        layers=[None,None,None,SimpleNamespace(self_attn=attn)])))
    rope=Qwen3_5TextRotaryEmbedding(config).to('cuda:0')
    return config,attn,model,rope


@pytest.mark.gpu
def test_strict_cache_releases_prefill_and_never_calls_local_decode():
    torch.manual_seed(20261104)
    config,attn,model,rope=make_fixture()
    original=attn.config
    with torch.inference_mode(),Runtime(GPUEmulator(max_tokens=160)) as rt:
        cache=DynamicCache(config=config)
        with Injection(model,rt,'strict_remote',160) as adapter:
            x=torch.randn(1,16,256,device='cuda:0',dtype=torch.bfloat16)
            pe=rope(x,torch.arange(16,device='cuda:0')[None,None].expand(3,1,16))
            attn(x,pe,None,past_key_values=cache)
            adapter.init_prefill(cache,'strict')
            assert isinstance(cache.layers[3],RemoteCacheLayer)
            assert adapter.ownership()['prefill_kv_released']
            def no_local(*args,**kwargs): raise AssertionError('local decode was called')
            adapter.original=no_local
            for pos in range(16,144):
                x=torch.randn(1,1,256,device='cuda:0',dtype=torch.bfloat16)
                pe=rope(x,torch.full((3,1,1),pos,device='cuda:0'))
                out,_=attn(x,pe,None,past_key_values=cache,position_ids=torch.tensor([[pos]],device='cuda:0'))
                assert torch.isfinite(out).all()
                assert cache.get_seq_length()==cache.get_seq_length(3)==pos+1
                assert cache.get_mask_sizes(1,3)==(pos+2,0)
                assert adapter.ownership()['local_kv_bytes']==0
            assert adapter.remote_outputs_used==128
        assert attn.config is original


def test_metadata_commit_and_unsupported_operations():
    b=RemoteCacheLayer(16,32,'handle')
    k=torch.zeros(1,4,1,256,dtype=torch.bfloat16)
    b.update(k,k)
    assert b.get_seq_length()==16 and b.keys is b.values is None
    with pytest.raises(PNMError,match='PENDING'): b.update(k,k)
    b.commit(17)
    with pytest.raises(PNMError,match='UNSUPPORTED'): b.crop(-1)
    b.poison('device lost')
    with pytest.raises(PNMError,match='POISONED'): b.update(k,k)
    with pytest.raises(PNMError,match='POISONED'): b.get_seq_length()
