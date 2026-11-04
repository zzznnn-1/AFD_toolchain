from pnm.validation import errors
import torch


def compare_runs(local,remote,atol=0.25,rtol=0.02):
    if not local or not remote: raise ValueError('empty run is not numerical validation')
    if len(local)!=len(remote): raise ValueError('different number of steps')
    records=[]
    for i,(a,b) in enumerate(zip(local,remote)):
        row={'step':i}
        for key in ['raw_o','attention_output','decoder_output','logits']:
            # Layer outputs can accumulate BF16/FP8 rounding; exact per-layer
            # paired attention is separately checked at 0.02/0.02 in the adapter.
            stats=errors(b[key],a[key])
            stats['pass']=stats['finite'] and bool(torch.allclose(b[key],a[key],atol=atol,rtol=rtol))
            stats.update(atol=atol,rtol=rtol)
            row[key]=stats
        row['argmax_equal']=bool(a['logits'].argmax(-1).eq(b['logits'].argmax(-1)).all())
        records.append(row)
    return {'status':'PASS' if all(r[k]['pass'] for r in records for k in ['raw_o','attention_output','decoder_output','logits']) else 'FAIL',
            'steps':records,'argmax_agreement':sum(r['argmax_equal'] for r in records)/len(records)}
