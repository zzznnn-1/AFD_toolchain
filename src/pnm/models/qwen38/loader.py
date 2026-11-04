"""Official FP8 load with explicit CPU storage / GPU0 execution fallback."""
import json
import time
import re
from pathlib import Path
from collections import Counter
import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForConditionalGeneration


def correct_skip_prefix_collisions(disk_config, weight_map):
    """5.19.0 interprets skip names as regex prefixes: mlp.gate hits gate_proj.

    Add an end anchor ONLY to ambiguous leaf names proven to overlap checkpoint
    FP8+scale entries. Do not change any weight, scale or quantization format.
    """
    quantized=[k.removesuffix('.weight_scale_inv') for k in weight_map if k.endswith('.weight_scale_inv')]
    skip=disk_config['quantization_config']['modules_to_not_convert']
    collisions={p:[m for m in quantized if re.match(p,m) and m!=p and not m.startswith(p+'.')] for p in skip}
    collisions={p:v for p,v in collisions.items() if v}
    return [p+'$' if p in collisions else p for p in skip],collisions


def load_model(config, local_path=None):
    if config['source_device']!='cuda:0' or config['emulator_device']!='cuda:1':
        raise ValueError('model runner currently requires the audited GPU0/GPU1 layout')
    print('Resolving official FP8 snapshot',config['model_id'],config['revision'],flush=True)
    path=local_path
    if path is None:
        try:
            cached=snapshot_download(config['model_id'],revision=config['revision'],local_files_only=True)
            index=json.loads((Path(cached)/'model.safetensors.index.json').read_text())
            required=set(index['weight_map'].values())|{'config.json','tokenizer.json','tokenizer_config.json'}
            if all((Path(cached)/n).is_file() for n in required): path=cached
        except (OSError,KeyError,ValueError):
            pass
    if path is None:
        path=snapshot_download(config['model_id'],revision=config['revision'],
                               allow_patterns=['*.json','*.safetensors','*.jinja','*.txt'])
    disk_config=json.loads((Path(path)/'config.json').read_text())
    if disk_config.get('quantization_config',{}).get('quant_method')!='fp8':
        raise ValueError('official FP8 checkpoint required; BF16/NF4 substitution forbidden')
    cfg=disk_config['text_config']
    assert (cfg['num_hidden_layers'],cfg['num_attention_heads'],cfg['num_key_value_heads'],cfg['head_dim'])==(64,24,4,256)
    resident=config['resident_layers']
    if not 0<=resident<=64: raise ValueError('resident_layers must be 0..64')
    if resident==64 and config.get('fully_resident',False):
        device_map={'':0}
    else:
        device_map={'model.visual':'cpu','model.language_model.embed_tokens':'cpu','lm_head':'cpu',
                    'model.language_model.norm':0,'model.language_model.rotary_emb':0}
        device_map.update({f'model.language_model.layers.{i}':0 if i<resident else 'cpu' for i in range(64)})
    start=time.perf_counter()
    weight_map=json.loads((Path(path)/'model.safetensors.index.json').read_text())['weight_map']
    corrected_skip,collisions=correct_skip_prefix_collisions(disk_config,weight_map)
    model_config=AutoConfig.from_pretrained(path,local_files_only=True)
    model_config.quantization_config['modules_to_not_convert']=corrected_skip
    if collisions:
        print('Anchoring',len(collisions),'ambiguous FP8 exclusion names; checkpoint weights/scales unchanged',flush=True)
    print('Loading FP8 model; resident decoder layers',resident,'snapshot',path,flush=True)
    model,loading_info=Qwen3_5ForConditionalGeneration.from_pretrained(
        path,config=model_config,local_files_only=True,device_map=device_map,dtype=torch.bfloat16,
        attn_implementation='sdpa',output_loading_info=True)
    model.eval()
    print('FP8 model loaded in',time.perf_counter()-start,'seconds',flush=True)
    if getattr(model.config.quantization_config,'dequantize',False):
        raise RuntimeError('FP8 unexpectedly dequantized')
    assert not any(p.device==torch.device('cuda:1') for p in model.parameters())
    assert 1 not in model.hf_device_map.values() and 'cuda:1' not in model.hf_device_map.values()
    from transformers.integrations.finegrained_fp8 import FP8Linear
    fp8=[n for n,m in model.named_modules() if isinstance(m,FP8Linear)]
    if not fp8: raise RuntimeError('no FP8Linear modules loaded')
    expected_fp8={k.removesuffix('.weight_scale_inv') for k in weight_map
                  if k.endswith('.weight_scale_inv') and k.startswith('model.language_model.layers.')}
    if set(fp8)!=expected_fp8:
        raise RuntimeError(f'FP8 module mismatch: missing={expected_fp8-set(fp8)}, extra={set(fp8)-expected_fp8}')
    if any(loading_info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']):
        raise RuntimeError(f'checkpoint load mismatch: {loading_info}')
    target=model.model.language_model.layers[3].self_attn
    facts={'model_id':config['model_id'],'revision':config['revision'],'snapshot_path':str(path),
           'load_seconds':time.perf_counter()-start,'hf_device_map':model.hf_device_map,
           'fp8_linear_count':len(fp8), 'resident_parameter_bytes_by_device_dtype':dict(Counter()),
           'fp8_skip_prefix_corrections':collisions,
           'loading_info':{k:sorted(v) if isinstance(v,set) else v for k,v in loading_info.items()},
           'target_class':type(target).__name__,
           'projections':{name:{'class':type(getattr(target,name)).__name__,
                                'parameters':{n:{'shape':list(p.shape),'dtype':str(p.dtype),'device':str(p.device)}
                                              for n,p in getattr(target,name).named_parameters()}}
                          for name in ['q_proj','k_proj','v_proj','o_proj']},
           'full_attention_layers':[i for i,t in enumerate(cfg['layer_types']) if t=='full_attention']}
    counts=Counter()
    for p in model.parameters():
        counts[f'{p.device}/{p.dtype}']+=p.numel()*p.element_size()
    facts['resident_parameter_bytes_by_device_dtype']=dict(counts)
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
    return model,tokenizer,facts
