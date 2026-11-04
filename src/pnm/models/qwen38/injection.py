from copy import copy
from uuid import uuid4
from transformers.models.qwen3_5.modeling_qwen3_5 import ALL_ATTENTION_FUNCTIONS, eager_attention_forward, Qwen3_5Attention
from .adapter import AttentionAdapter


class Injection:
    """Scoped registration and per-layer config copy, restores exact original config."""
    def __init__(self, model, runtime=None, mode='local', capacity=8192, layer_id=3, atol=0.02, rtol=0.02):
        self.layer=model.model.language_model.layers[layer_id]
        self.attention=self.layer.self_attn
        if not isinstance(self.attention,Qwen3_5Attention):
            raise TypeError(type(self.attention))
        cfg=self.attention.config
        if (cfg.num_attention_heads,cfg.num_key_value_heads,cfg.head_dim)!=(24,4,256):
            raise ValueError('model dimensions differ from Qwen3.8-27B contract')
        if layer_id!=3 or cfg.layer_types[layer_id]!='full_attention':
            raise ValueError('this prototype targets Full Attention layer 3 only')
        self.original_config=cfg
        self.key='pnm_'+uuid4().hex
        native=ALL_ATTENTION_FUNCTIONS.get_interface(cfg._attn_implementation,eager_attention_forward)
        self.adapter=AttentionAdapter(native,mode,runtime,capacity,layer_id,atol,rtol)
        self.active=False

    def __enter__(self):
        if self.active: raise RuntimeError('already installed')
        ALL_ATTENTION_FUNCTIONS[self.key]=self.adapter
        cfg=copy(self.original_config)
        cfg._attn_implementation=self.key
        self.attention.config=cfg
        self.active=True
        return self.adapter

    def __exit__(self,*_):
        if self.active:
            try:
                self.adapter.reset()
            finally:
                self.attention.config=self.original_config
                del ALL_ATTENTION_FUNCTIONS[self.key]
                self.active=False
