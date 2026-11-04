import time
import hashlib
import json
from dataclasses import asdict
import torch
from .injection import Injection
from pnm.measurement import Span


class Runner:
    def __init__(self,model,tokenizer,runtime=None,capacity=8192,attention_atol=0.02,attention_rtol=0.02):
        self.model,self.tokenizer,self.runtime=model,tokenizer,runtime
        self.capacity=capacity
        self.attention_atol,self.attention_rtol=attention_atol,attention_rtol

    def prompt(self,text=None,length=None):
        text=text or '用简短的中文解释为什么单 token 自回归 Attention 必须看到全部历史。'
        if length:
            ids=self.tokenizer(' Attention uses queries, keys and values. '* (length//5+20),return_tensors='pt').input_ids
            while ids.shape[1]<length: ids=torch.cat([ids,ids],dim=1)
            return ids[:,:length].to('cuda:0')
        rendered=self.tokenizer.apply_chat_template([{'role':'user','content':text}],tokenize=False,
                                                    add_generation_prompt=True,enable_thinking=False)
        return self.tokenizer(rendered,return_tensors='pt').input_ids.to('cuda:0')

    @torch.inference_mode()
    def run(self,input_ids,mode,steps,fixed_tokens=None,request_id='qwen',capture=True,step_callback=None):
        if steps < 1 or input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 1:
            raise ValueError('runner requires nonempty batch=1 prompt and positive decode steps')
        if mode != 'local' and self.runtime is None:
            raise ValueError('remote modes require a runtime')
        if input_ids.shape[1]+steps>self.capacity: raise ValueError('capacity insufficient')
        if fixed_tokens is not None and len(fixed_tokens)!=steps: raise ValueError('fixed token count mismatch')
        self.model.model.rope_deltas=None
        captures=[]
        saved={}
        layer_times=[]
        layer_span=[None]
        def layer_pre(module,args,kwargs):
            hidden=kwargs.get('hidden_states',args[0] if args else None)
            layer_span[0]=Span(hidden.device)
        def layer_post(module,inputs,output):
            layer_times.append(layer_span[0].finish())
            layer_span[0]=None
        def attention_hook(module,inputs,output): saved['attention_output']=output[0][:,-1:].detach()
        def decoder_hook(module,inputs,output): saved['decoder_output']=output[:,-1:].detach()
        layer=self.model.model.language_model.layers[3]
        original_config=layer.self_attn.config
        hooks=[layer.self_attn.register_forward_hook(attention_hook),layer.register_forward_hook(decoder_hook)] if capture else []
        hooks.extend([layer.register_forward_pre_hook(layer_pre,with_kwargs=True),layer.register_forward_hook(layer_post)])
        execution_devices={}
        def device_hook(index):
            def observe(module, inputs, output):
                device=str(output.device)
                execution_devices[index]=device
                if device != 'cuda:0':
                    raise RuntimeError(f'decoder layer {index} executed on {device}')
            return observe
        for i,decoder in enumerate(self.model.model.language_model.layers):
            hooks.append(decoder.register_forward_hook(device_hook(i)))
        for i in ([0,1] if self.runtime is not None else [0]):
            torch.cuda.reset_peak_memory_stats(i)
        start=time.perf_counter()
        generated=[]
        input_sequence=[]
        latencies=[]
        try:
            with Injection(self.model,self.runtime,mode,self.capacity,atol=self.attention_atol,rtol=self.attention_rtol) as adapter:
                out=self.model(input_ids=input_ids,attention_mask=torch.ones_like(input_ids),use_cache=True,logits_to_keep=1)
                if not torch.isfinite(out.logits).all(): raise RuntimeError('nonfinite prefill logits')
                torch.cuda.synchronize(0)
                prefill_model_seconds=time.perf_counter()-start
                cache=out.past_key_values
                migration_start=time.perf_counter()
                adapter.init_prefill(cache,request_id)
                prefill_ownership=adapter.ownership()
                prefill_remote_init_seconds=time.perf_counter()-migration_start if mode!='local' else None
                prefill_qkv=adapter.last_qkv
                prefill_info=asdict(self.runtime.get_cache_info(adapter.handle)) if adapter.handle else None
                prefill_logits=out.logits[:,-1].detach().float().cpu()
                next_token=out.logits[:,-1].argmax(-1,keepdim=True)
                prefill_seconds=time.perf_counter()-start
                for step in range(steps):
                    if step_callback is not None: step_callback(step,adapter,cache)
                    t=time.perf_counter()
                    token=next_token if fixed_tokens is None else torch.tensor([[fixed_tokens[step]]],device='cuda:0')
                    input_sequence.append(int(token.item()))
                    old=input_ids.shape[1]+step
                    out=self.model(input_ids=token,past_key_values=cache,use_cache=True,logits_to_keep=1,
                                   position_ids=torch.tensor([[old]],device='cuda:0'),
                                   attention_mask=torch.ones(1,old+1,device='cuda:0',dtype=torch.long))
                    if cache.get_seq_length() != old+1: raise RuntimeError('global cache length mismatch')
                    for i in range(3,64,4):
                        if cache.get_seq_length(i) != old+1: raise RuntimeError(f'layer {i} cache length mismatch')
                    if not torch.isfinite(out.logits).all(): raise RuntimeError(f'nonfinite decode logits at {step}')
                    next_token=out.logits[:,-1].argmax(-1,keepdim=True)
                    generated.append(int(next_token.item()))
                    if capture:
                        captures.append({'raw_o':adapter.last_raw[:,-1:].float().cpu(),
                                         'attention_output':saved['attention_output'].float().cpu(),
                                         'decoder_output':saved['decoder_output'].float().cpu(),
                                         'logits':out.logits[:,-1].float().cpu()})
                    torch.cuda.synchronize(0)
                    latencies.append(time.perf_counter()-t)
                    if step in [0,31,63,95,127]:
                        print(f'{request_id} {mode} step={step+1}/{steps} cache={cache.get_seq_length()} remote_used={adapter.remote_outputs_used}',flush=True)
                final_info=asdict(self.runtime.get_cache_info(adapter.handle)) if adapter.handle else None
                target_cache=cache.layers[3]
                local_kv_bytes=sum(t.numel()*t.element_size() for t in (target_cache.keys,target_cache.values) if t is not None)
                summary={'mode':mode,'request_id':request_id,'prompt_tokens':input_ids.shape[1],'decode_steps':steps,
                         'prompt_token_ids':input_ids[0].tolist(),
                         'prompt_sha256':hashlib.sha256(json.dumps(input_ids[0].tolist()).encode()).hexdigest(),
                         'input_tokens':input_sequence,'predicted_tokens':generated,'prefill_seconds':prefill_seconds,
                         'prefill_model_seconds':prefill_model_seconds,
                         'prefill_remote_init_seconds':prefill_remote_init_seconds,
                         'target_local_kv_bytes':local_kv_bytes,
                         'decode_seconds':sum(latencies),'decode_step_seconds':latencies,
                         'generated_text':self.tokenizer.decode(input_sequence,skip_special_tokens=False),
                         'prefill_qkv':prefill_qkv,'decode_qkv':adapter.last_qkv,
                         'prefill_cache':prefill_info,'final_cache':final_info,
                         'prefill_ownership':prefill_ownership,'final_ownership':adapter.ownership(),
                         'decode_timings':list(adapter.decode_timings),
                         'layer3_timings':layer_times,
                         'raw_local_remote_errors':list(adapter.comparisons),
                         'position_records':list(adapter.position_records),
                         'remote_calls':adapter.remote_calls,'remote_outputs_used':adapter.remote_outputs_used,
                         'decoder_execution_devices':execution_devices,
                         'gpu_memory':{str(i):{'allocated':torch.cuda.memory_allocated(i),
                                              'reserved':torch.cuda.memory_reserved(i),'peak':torch.cuda.max_memory_allocated(i)} for i in [0,1]}}
                if mode!='local' and adapter.remote_calls!=steps: raise RuntimeError('remote call count mismatch')
                if mode in ('shadow_remote','strict_remote') and adapter.remote_outputs_used!=steps: raise RuntimeError('remote output unused')
            summary['adapter_restored']=layer.self_attn.config is original_config
            if not summary['adapter_restored']: raise RuntimeError('adapter configuration not restored')
        finally:
            for h in hooks: h.remove()
        return summary,captures,prefill_logits
