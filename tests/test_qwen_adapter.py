"""Real installed Qwen forward + real GPU emulator, small random test fixture.

These are adapter regression tests, NEVER evidence of full 27B acceptance.
"""
import torch
import pytest
from transformers import Qwen3_5TextConfig
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention, Qwen3_5TextRotaryEmbedding, ALL_ATTENTION_FUNCTIONS
from pnm.models.qwen38.adapter import AttentionAdapter
from pnm.runtime import Runtime
from pnm.backends.gpu_emulator import GPUEmulator


@pytest.mark.gpu
def test_native_forward_remote_replacement_and_cache():
    if torch.cuda.device_count()<2: pytest.skip('two GPUs required')
    from copy import copy
    torch.manual_seed(20261104)
    config=Qwen3_5TextConfig(hidden_size=256,intermediate_size=512,num_hidden_layers=4,
                           num_attention_heads=24,num_key_value_heads=4,head_dim=256,
                           layer_types=['linear_attention']*3+['full_attention'])
    config._attn_implementation='sdpa'
    attn=Qwen3_5Attention(config,3).to(device='cuda:0',dtype=torch.bfloat16).eval()
    rope=Qwen3_5TextRotaryEmbedding(config).to('cuda:0')
    native=ALL_ATTENTION_FUNCTIONS['sdpa']
    with Runtime(GPUEmulator(max_tokens=32)) as rt,torch.inference_mode():
        adapter=AttentionAdapter(native,'shadow_remote',rt,32)
        key='pnm_adapter_regression'
        original_config=attn.config
        ALL_ATTENTION_FUNCTIONS[key]=adapter
        attn.config=copy(config)
        attn.config._attn_implementation=key
        try:
            cache=DynamicCache(config=config)
            x=torch.randn(1,16,256,device='cuda:0',dtype=torch.bfloat16)
            pe=rope(x,torch.arange(16,device='cuda:0').view(1,1,16).expand(3,1,16))
            attn(x,pe,None,past_key_values=cache)
            adapter.init_prefill(cache,'fixture')
            assert adapter.remote_calls==0
            x=torch.randn(1,1,256,device='cuda:0',dtype=torch.bfloat16)
            pe=rope(x,torch.tensor([[[16]],[[16]],[[16]]],device='cuda:0'))
            captured={}
            def proj_pre(module,args): captured['gated']=args[0].clone()
            h=attn.o_proj.register_forward_pre_hook(proj_pre)
            output,_=attn(x,pe,None,past_key_values=cache,position_ids=torch.tensor([[16]],device='cuda:0'))
            h.remove()
            assert cache.get_seq_length()==17
            assert rt.get_cache_info(adapter.handle).sequence_length==17
            assert adapter.remote_outputs_used==1
            projected=attn.q_proj(x).view(1,1,24,512)
            gate=projected.chunk(2,dim=-1)[1].reshape(1,1,-1)
            expected=adapter.last_raw.reshape(1,1,-1)*gate.sigmoid()
            torch.testing.assert_close(captured['gated'],expected,atol=0,rtol=0)
            torch.testing.assert_close(output,attn.o_proj(expected),atol=0,rtol=0)
            # Causal intervention at remote O proves native forward uses its value.
            adapter.reset('shadow_remote')
            cache=DynamicCache(config=config)
            attn(x,pe,None,past_key_values=cache)
            adapter.init_prefill(cache,'intervention')
            native_append=rt.append_and_attend
            def changed(*a,**kw): return native_append(*a,**kw)+0.001
            rt.append_and_attend=changed
            seen={}
            def capture_native(module,q,k,v,mask,**kw):
                o,w=native(module,q,k,v,mask,**kw)
                seen['local']=o.clone()
                return o,w
            adapter.original=capture_native
            attn(x,pe,None,past_key_values=cache)
            assert not torch.equal(adapter.last_raw,seen['local'])
            assert adapter.remote_outputs_used==1
        finally:
            adapter.reset()
            attn.config=original_config
            del ALL_ATTENTION_FUNCTIONS[key]
        assert attn.config is original_config and key not in ALL_ATTENTION_FUNCTIONS
