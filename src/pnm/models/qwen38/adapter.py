"""Attention-interface adapter; the installed Transformers forward stays intact.

Native projection/norm/RoPE/cache.update precede this interface; native sigmoid
gate and o_proj follow its returned tensor. No environment source is patched.
"""
import torch
import weakref
from time import perf_counter
from pnm.protocol import CacheSpec
from pnm.validation import check_close
from .cache_bridge import RemoteCacheLayer
from pnm.measurement import Span


class AttentionAdapter:
    MODES = ('local','shadow_compare','shadow_remote','strict_remote')

    def __init__(self, original_interface, mode='local', runtime=None, capacity=8192, layer_id=3,
                 atol=0.02, rtol=0.02):
        if mode not in self.MODES:
            raise ValueError(mode)
        if mode != 'local' and runtime is None:
            raise ValueError('remote attention requires a runtime')
        self.original = original_interface
        self.mode, self.runtime = mode, runtime
        self.capacity, self.layer_id = capacity, layer_id
        self.atol,self.rtol=atol,rtol
        self.handle = None
        self.last_raw = None
        self.last_qkv = None
        self.comparisons = []
        self.remote_calls = 0
        self.remote_outputs_used = 0
        self.position_records = []
        self.bridge = None
        self.released_kv = ()
        self.retired_prefill_kv_bytes = 0
        self.decode_timings = []
        self.preparation_span = None
        self.preparation_timing = None

    def begin_preparation(self,module,args,kwargs):
        hidden=kwargs.get('hidden_states',args[0] if args else None)
        self.preparation_span=Span(hidden.device)

    def reset(self, mode=None):
        try:
            if self.handle is not None:
                self.runtime.free_cache(self.handle)
        finally:
            if self.bridge is not None:
                self.bridge.poison('request finished; remote handle released')
            self.handle = None
            self.bridge = None
            self.released_kv = ()
        self.last_raw = None
        self.last_qkv = None
        self.comparisons.clear()
        self.position_records.clear()
        self.decode_timings.clear()
        self.remote_calls = self.remote_outputs_used = 0
        if mode is not None:
            if mode not in self.MODES: raise ValueError(mode)
            self.mode = mode

    def init_prefill(self, cache, request_id):
        if self.mode == 'local': return
        if self.handle is not None: raise RuntimeError('prefill already initialized')
        layer = cache.layers[self.layer_id]
        k,v=layer.keys,layer.values
        if cache.get_seq_length()!=k.shape[2] or cache.get_seq_length(self.layer_id)!=k.shape[2]:
            raise RuntimeError('native global/layer cache lengths differ')
        h=self.runtime.alloc_cache(request_id,self.layer_id,CacheSpec(self.capacity))
        try:
            self.runtime.init_cache(h,k,v)
        except Exception:
            self.runtime.free_cache(h)
            raise
        self.handle=h
        if self.mode == 'strict_remote':
            self.retired_prefill_kv_bytes = k.numel()*k.element_size()+v.numel()*v.element_size()
            self.released_kv = (weakref.ref(k), weakref.ref(v))
            self.bridge = RemoteCacheLayer(k.shape[2], self.capacity, h)
            cache.layers[self.layer_id] = self.bridge
            layer.keys = layer.values = None
            self.last_raw = None

    def ownership(self):
        if self.bridge is None:
            return None
        info=self.bridge.metadata()
        info['prefill_kv_released']=all(ref() is None for ref in self.released_kv)
        info['retired_prefill_kv_bytes']=self.retired_prefill_kv_bytes
        if info['local_kv_bytes'] or not info['prefill_kv_released']:
            raise RuntimeError('strict layer still owns historical GPU0 KV')
        return info

    def __call__(self, module, query, key, value, attention_mask, **kwargs):
        self.preparation_timing=self.preparation_span.finish() if self.preparation_span else None
        self.preparation_span=None
        start=perf_counter()
        if module.layer_idx!=self.layer_id:
            raise RuntimeError('adapter reached a different layer')
        self.last_qkv={n:{'shape':list(t.shape),'dtype':str(t.dtype),'device':str(t.device)}
                       for n,t in [('q',query),('k',key),('v',value)]}
        if self.mode == 'strict_remote' and self.handle is not None:
            return self._strict(module,query,key,value,attention_mask,start,**kwargs)
        local, weights=self.original(module,query,key,value,attention_mask,**kwargs)
        if self.mode=='local' or self.handle is None:
            # Includes native prefill even if prompt length == 1.
            self.last_raw=local.detach()
            if self.mode=='local' and query.shape[2]==1 and key.shape[2]>1:
                self.decode_timings.append({'position':key.shape[2]-1,'qkv_prepare':self.preparation_timing})
            return local,weights
        if query.shape[2]!=1 or query.shape[0]!=1:
            raise RuntimeError('v1 remote decode requires B=1,T=1')
        pos=key.shape[2]-1
        info=self.runtime.get_cache_info(self.handle)
        if info.sequence_length!=pos:
            raise RuntimeError(f'local/remote cache mismatch: {pos} != {info.sequence_length}')
        position_ids=kwargs.get('position_ids')
        if position_ids is not None and int(position_ids.flatten()[-1])!=pos:
            raise RuntimeError(f'position_ids inconsistent with cache length {pos}')
        cache_position=kwargs.get('cache_position')
        if cache_position is not None and int(cache_position.flatten()[-1])!=pos:
            raise RuntimeError('cache_position mismatch')
        if kwargs.get('scaling',module.scaling)!=query.shape[-1]**-0.5:
            raise RuntimeError('unsupported attention scale')
        remote=self.runtime.append_and_attend(self.handle,query,key[:,:,-1:],value[:,:,-1:],
                                              token_position=pos,expected_length=pos,attention_mask=attention_mask)
        remote=remote.transpose(1,2).contiguous()  # BHTD -> native BTHD interface.
        stats=check_close(remote,local,atol=self.atol,rtol=self.rtol)
        self.comparisons.append({'position':pos,**stats})
        self.position_records.append({'position':pos,'native_length':key.shape[2],
                                      'remote_length':self.runtime.get_cache_info(self.handle).sequence_length,
                                      'position_ids':position_ids.flatten().tolist() if position_ids is not None else None,
                                      'cache_position':cache_position.flatten().tolist() if cache_position is not None else None,
                                      'mask':None if attention_mask is None else {'shape':list(attention_mask.shape),'dtype':str(attention_mask.dtype)}})
        self.remote_calls+=1
        chosen=remote if self.mode=='shadow_remote' else local
        self.remote_outputs_used+=int(self.mode=='shadow_remote')
        self.last_raw=chosen.detach()
        self.decode_timings.append({'position':pos,'offload_wall_ms':(perf_counter()-start)*1000,
                                    'qkv_prepare':self.preparation_timing})
        return chosen,weights

    def _strict(self,module,query,key,value,attention_mask,start,**kwargs):
        try:
            if self.bridge is None or not self.bridge.pending:
                raise RuntimeError('strict interface requires pending CacheBridge update')
            pos=self.bridge.get_seq_length()
            if query.shape!=(1,24,1,256) or key.shape!=(1,4,1,256) or value.shape!=key.shape:
                raise RuntimeError('strict interface received historical KV or unsupported query')
            position_ids,cache_position=kwargs.get('position_ids'),kwargs.get('cache_position')
            for positions in (position_ids,cache_position):
                if positions is not None and int(positions.flatten()[-1])!=pos:
                    raise RuntimeError('strict cache position mismatch')
            if kwargs.get('scaling',module.scaling)!=query.shape[-1]**-0.5:
                raise RuntimeError('unsupported attention scaling')
            remote=self.runtime.append_and_attend(self.handle,query,key,value,token_position=pos,
                                                  expected_length=pos,attention_mask=attention_mask)
            info=self.runtime.get_cache_info(self.handle)
            self.bridge.commit(info.sequence_length)
            chosen=remote.transpose(1,2).contiguous()
            if not torch.isfinite(chosen).all(): raise RuntimeError('nonfinite remote attention')
            self.remote_calls+=1
            self.remote_outputs_used+=1
            self.last_raw=chosen.detach()
            self.position_records.append({'position':pos,'native_length':self.bridge.length,
                'remote_length':info.sequence_length,
                'position_ids':position_ids.flatten().tolist() if position_ids is not None else None,
                'cache_position':cache_position.flatten().tolist() if cache_position is not None else None,
                'mask':None if attention_mask is None else {'shape':list(attention_mask.shape),'dtype':str(attention_mask.dtype)},
                'ownership':self.ownership()})
            self.decode_timings.append({'position':pos,'offload_wall_ms':(perf_counter()-start)*1000,
                                        'qkv_prepare':self.preparation_timing})
            return chosen,None
        except BaseException as exc:
            if self.bridge is not None: self.bridge.poison(exc)
            raise
