"""Fail closed when any P0-P3 acceptance artifact or actual data-path evidence is missing."""
import _bootstrap
import json
import hashlib
from collections import defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET
from pnm.validation import write_json


def read(name): return json.loads((Path('reports/raw')/name).read_text())


def main():
    env,p1,p2,p3=map(read,['environment.json','P1_attention.json','P2_emulator.json','P3_suite.json'])
    assert len(env['gpus'])>=2
    assert 'FP8_LINEAR_PASS' in Path('reports/raw/fp8_kernel_probe.log').read_text()
    assert p1['status']==p2['status']==p3['status']=='PASS'
    assert {c['length'] for c in p1['cases'] if c.get('case')=='prefill'} >= {1,16,128,1024,4096}
    assert all(c['pass'] for c in p1['cases'])
    assert p2['steps']>=128 and all(c['pass'] for c in p2['steps_detail'])
    xml=ET.parse('reports/raw/unit_tests.xml').getroot()
    suites=list(xml) if xml.tag=='testsuites' else [xml]
    assert all(int(s.get('failures',0))==int(s.get('errors',0))==int(s.get('skipped',0))==0 for s in suites)
    assert p3['model']['fp8_linear_count']==400
    assert not any(p3['model']['loading_info'].values())
    assert all(v not in (1,'cuda:1') for v in p3['model']['hf_device_map'].values())
    assert {c['case'] for c in p3['comparisons']}=={'short','medium','context4096'}
    assert len(p3['comparisons'])==6 and all(c['status']=='PASS' for c in p3['comparisons'])
    modes={r['mode'] for r in p3['runs']}
    assert modes=={'local','shadow_compare','shadow_remote'}
    remotes=[r for r in p3['runs'] if r['mode']!='local']
    for r in remotes:
        assert r['remote_calls']==r['decode_steps']
        assert r['remote_outputs_used']==(r['decode_steps'] if r['mode']=='shadow_remote' else 0)
        assert r['prefill_cache']['sequence_length']==r['prompt_tokens']
        assert r['final_cache']['sequence_length']==r['prompt_tokens']+r['decode_steps']
        assert all(x['pass'] for x in r['raw_local_remote_errors'])
        assert all(x['native_length']==x['remote_length']==x['position']+1 for x in r['position_records'])
    greedy=[r for r in p3['runs'] if 'matches_local_generated_tokens' in r]
    assert {r['decode_steps'] for r in greedy} >= {32,128}
    groups=defaultdict(list)
    for line in Path('reports/raw/P3_suite.jsonl').read_text().splitlines():
        rec=json.loads(line)
        groups[rec['command_id']].append(rec)
    init_counts=defaultdict(int)
    counts=defaultdict(int)
    for command_id,records in groups.items():
        commands=[r for r in records if r['stage']=='command']
        assert len(commands)==1
        c=commands[0]
        assert c['status']=='PASS'
        counts[c['op']]+=1
        if c['op']=='init_cache': init_counts[c['handle']]+=1
        if c['op']=='append_and_attend':
            transfers=[r for r in records if r['stage']=='cuda:0_d2h']
            sizes={r['tensor']:r['bytes'] for r in transfers}
            assert sizes=={'q':12288,'k':2048,'v':2048},sizes
            assert all(r['pinned'] for r in records if 'pinned' in r)
            kernels=[r for r in records if r['stage']=='attention']
            assert len(kernels)==1 and kernels[0]['device']=='cuda:1'
            returns=[r for r in records if r['stage']=='cuda:0_h2d']
            assert len(returns)==1 and returns[0]['tensor']=='o' and returns[0]['bytes']==12288
    assert all(n==1 for n in init_counts.values()) and len(init_counts)==len(remotes)
    assert counts['append_and_attend']==sum(r['decode_steps'] for r in remotes)
    manifest={}
    for directory in ['src','tests','configs']:
        for path in sorted(Path(directory).rglob('*')):
            if path.is_file() and '__pycache__' not in str(path) and '.egg-info' not in str(path):
                manifest[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    result={'status':'BLOCKED','functional_status':'PASS',
            'stages':{'P0':'PASS','P1':'PASS','P2':'PASS','P3':'BLOCKED'},
            'P3_functional_status':'PASS',
            'P3_deployment_status':'BLOCKED: CPU weight offload and pre-existing unrelated GPU1 service',
            'unit_tests':sum(int(s.get('tests',0)) for s in suites),'command_counts':dict(counts),
            'remote_tokens':counts['append_and_attend'],'remote_output_tokens':sum(r['remote_outputs_used'] for r in remotes),
            'all_greedy_sequences_match':all(r['matches_local_generated_tokens'] for r in greedy),
            'deployment_constraints':{'all_weights_resident_gpu0':'BLOCKED: capacity; CPU weight offload used',
                                      'exclusive_physical_gpu1':'BLOCKED: pre-existing unrelated service'},
            'source_sha256':manifest}
    write_json('reports/raw/acceptance.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'},indent=2))


if __name__=='__main__': main()
