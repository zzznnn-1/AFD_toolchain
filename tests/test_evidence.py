from copy import deepcopy
import pytest
from pnm.acceptance import verify_provenance, verify_trace, VerificationError
from pnm.evidence import provenance, finish_provenance, source_manifest


def test_result_cannot_certify_modified_source(tmp_path):
    (tmp_path/'src/pnm').mkdir(parents=True)
    source=tmp_path/'src/pnm/module.py'
    source.write_text('x = 1\n')
    e=provenance(tmp_path)
    finish_provenance(e,tmp_path)
    verify_provenance(e,source_manifest(tmp_path))
    source.write_text('x = 2\n')
    with pytest.raises(VerificationError,match='differs'): verify_provenance(e,source_manifest(tmp_path))
    with pytest.raises(RuntimeError,match='changed'): finish_provenance(e,tmp_path)


def valid_trace():
    records=[]
    for i,(op,length) in enumerate([('alloc_cache',0),('init_cache',16),('append_and_attend',17),('free_cache',17)]):
        cid=str(i)
        def transfer(stage,name,size):
            records.append({'command_id':cid,'run_id':'r','stage':stage,'tensor':name,'bytes':size,'pinned':True,'ms':0.01})
        if op=='init_cache':
            for stage in ('cuda:0_d2h','cuda:1_h2d'):
                for name in ('k','v'): transfer(stage,name,16*2048)
        if op=='append_and_attend':
            for stage in ('cuda:0_d2h','cuda:1_h2d'):
                for name,size in [('q',12288),('k',2048),('v',2048)]: transfer(stage,name,size)
            records.append({'command_id':cid,'run_id':'r','stage':'attention','device':'cuda:1','length':17,'ms':0.02})
            for stage in ('cuda:1_d2h','cuda:0_h2d'): transfer(stage,'o',12288)
        records.append({'command_id':cid,'run_id':'r','stage':'command','op':op,'handle':'h',
                        'request_id':'req','layer_id':3,'status':'PASS','length':length,'dtype':'bfloat16',
                        'layout':'BHTD','expected_length':16,'token_position':16,'ms':0.1})
    runs=[{'request_id':'req','mode':'shadow_remote','prompt_tokens':16,'decode_steps':1}]
    return records,runs


def test_trace_requires_each_transfer_kernel_and_free():
    trace,runs=valid_trace()
    counts,_=verify_trace(trace,'r',runs)
    assert counts['append_and_attend']==1
    for stage in ('attention','cuda:0_d2h','cuda:1_h2d','cuda:1_d2h','cuda:0_h2d'):
        changed=deepcopy(trace)
        changed.pop(next(i for i,r in enumerate(changed) if r['command_id']=='2' and r['stage']==stage))
        with pytest.raises(VerificationError): verify_trace(changed,'r',runs)
    with pytest.raises(VerificationError,match='leaked'): verify_trace(trace[:-1],'r',runs)


def test_duplicate_or_foreign_trace_is_not_accepted():
    trace,runs=valid_trace()
    with pytest.raises(VerificationError,match='different run'): verify_trace(trace,'other',runs)
    duplicate=deepcopy(trace)
    duplicate.insert(7,dict(duplicate[7]))
    with pytest.raises(VerificationError): verify_trace(duplicate,'r',runs)
    changed=deepcopy(trace)
    next(r for r in changed if r.get('op')=='append_and_attend')['token_position']=15
    with pytest.raises(VerificationError,match='position'): verify_trace(changed,'r',runs)
