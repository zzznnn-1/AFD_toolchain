from copy import deepcopy
import json
import struct
import pytest
from pnm.models.qwen38.preflight import placement_plan, inventory


def fixture():
    gib = 2**30
    modules = {f'model.language_model.layers.{i}':gib//3 for i in range(64)}
    modules.update({'lm_head':2*gib, 'model.language_model.embed_tokens':2*gib,
                    'model.language_model.norm':10240, 'model.visual':gib})
    weights = {'module_bytes':modules,'parameter_bytes':sum(modules.values())}
    hw = {'gpus':[{'free_bytes':23*gib,'total_bytes':24*gib,'uuid':'GPU-0'},
                  {'free_bytes':12*gib,'total_bytes':24*gib,'uuid':'GPU-1'}],
          'compute_processes':[{'pid':99,'gpu_uuid':'GPU-1','name':'existing-service','used_mib':11000}],
          'current_pid':1,'process_probe_error':None}
    config = {'source_device':'cuda:0','emulator_device':'cuda:1','capacity':8192,
              'resident_layers':'auto','workspace_reserve_gib':4.0}
    return config, weights, hw


def test_cpu_staging_budget_excludes_gpu1_weights():
    cfg,w,hw = fixture()
    result = placement_plan(cfg,w,hw)
    assert result['status']=='PASS' and result['deployment_status']=='BLOCKED'
    assert result['required_gpu0_bytes']<=hw['gpus'][0]['free_bytes']
    assert result['largest_staged_module_bytes']==2*2**30
    assert set(result['device_map'].values())=={0,'cpu'}
    assert result['foreign_gpu1_processes']==hw['compute_processes']
    strict = placement_plan(cfg,w,hw,strict=True)
    assert strict['status']=='BLOCKED' and len(strict['blocking_reasons'])==2


def test_sufficient_empty_hardware_really_passes_strict():
    cfg,w,hw = fixture()
    cfg['fully_resident']=True
    hw['gpus'][0]['free_bytes']=80*2**30
    hw['compute_processes']=[]
    result=placement_plan(cfg,w,hw,strict=True)
    assert result['status']==result['deployment_status']=='PASS'
    assert result['cpu_parameter_bytes']==0 and result['device_map']=={'':0}


def test_capacity_and_unknown_process_inventory_fail_closed():
    cfg,w,hw=fixture()
    cfg['fully_resident']=True
    result=placement_plan(cfg,w,hw)
    assert result['status']=='BLOCKED' and 'GPU0 requires' in result['blocking_reasons'][0]
    hw['gpus'][0]['free_bytes']=80*2**30
    hw['compute_processes']=[]
    hw['process_probe_error']='nvidia-smi unavailable'
    assert placement_plan(cfg,w,hw,strict=True)['status']=='BLOCKED'


def test_inventory_uses_actual_storage_bytes_and_rejects_truncation(tmp_path):
    name='model.language_model.layers.0.weight'
    header=json.dumps({name:{'shape':[2,2],'dtype':'F8_E4M3','data_offsets':[0,4]}}).encode()
    path=tmp_path/'weights.safetensors'
    path.write_bytes(struct.pack('<Q',len(header))+header+bytes(4))
    (tmp_path/'config.json').write_text('{}')
    (tmp_path/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{name:path.name}}))
    assert inventory(tmp_path)['parameter_bytes']==4
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError,match='truncated'): inventory(tmp_path)
