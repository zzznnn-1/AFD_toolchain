"""Distribution summaries over measured observations, not sums of device clocks."""
import csv
import math
from collections import defaultdict
from pathlib import Path


def distribution(values):
    values=sorted(float(v) for v in values)
    if not values: return {'count':0,'p50':None,'p95':None,'p99':None,'mean':None}
    def percentile(p):
        x=(len(values)-1)*p
        low,high=math.floor(x),math.ceil(x)
        return values[low]+(values[high]-values[low])*(x-low)
    return {'count':len(values),'p50':percentile(.50),'p95':percentile(.95),'p99':percentile(.99),
            'mean':sum(values)/len(values),'min':values[0],'max':values[-1]}


def summarize(runs,trace,warmup_steps=8):
    grouped=defaultdict(list)
    for record in trace:
        if 'command_id' in record: grouped[record['command_id']].append(record)
    offload=defaultdict(list)
    initializations={}
    for records in grouped.values():
        completion=next((r for r in records if r['stage']=='command'),None)
        if completion is None or completion['status']!='PASS': continue
        stage_ms=defaultdict(float)
        transfer_bytes=defaultdict(int)
        descriptor_bytes=0
        for r in records:
            stage_ms[r['stage']]+=r.get('ms',0)
            transfer_bytes[r['stage']]+=r.get('bytes',0)
            descriptor_bytes+=r.get('descriptor_bytes',0)
        item={'position':completion['token_position'],'stage_ms':dict(stage_ms),
              'physical_copy_bytes':sum(transfer_bytes.values()),'descriptor_bytes':descriptor_bytes}
        if completion['op']=='append_and_attend': offload[completion['request_id']].append(item)
        if completion['op']=='init_cache': initializations[completion['request_id']]=item
    rows=[]
    for run in runs:
        for window in (16,128):
            if run['decode_steps']<window: continue
            selected=run['decode_step_seconds'][warmup_steps:window]
            record={'request_id':run['request_id'],'mode':run['mode'],'context':run['prompt_tokens'],
                    'window_tokens':window,'warmup_tokens_excluded':warmup_steps,
                    'prefill_seconds':run['prefill_seconds'],
                    'prefill_model_seconds':run['prefill_model_seconds'],
                    'prefill_remote_init_seconds':run['prefill_remote_init_seconds'],
                    'token_ms':distribution([t*1000 for t in selected]),
                    'tokens_per_second':len(selected)/sum(selected),
                    'gpu_memory':run['gpu_memory'],'prefill_transfer':initializations.get(run['request_id']),
                    'cache':run['final_cache']}
            timed=offload.get(run['request_id'],[])[warmup_steps:window]
            stages=set().union(*(t['stage_ms'].keys() for t in timed)) if timed else set()
            record['stages_ms']={stage:distribution([t['stage_ms'].get(stage,0) for t in timed]) for stage in stages}
            record['copy_bytes_per_token']=sorted(set(t['physical_copy_bytes'] for t in timed))
            record['descriptor_bytes_per_token']=distribution([t['descriptor_bytes'] for t in timed])
            prepared=run.get('decode_timings',[])[warmup_steps:window]
            record['qkv_prepare_cuda_ms']=distribution([p['qkv_prepare']['cuda_ms'] for p in prepared if p.get('qkv_prepare')])
            record['offload_wall_ms']=distribution([p['offload_wall_ms'] for p in prepared if 'offload_wall_ms' in p])
            record['layer3_wall_ms']=distribution([p['wall_ms'] for p in run['layer3_timings'][1+warmup_steps:1+window]])
            rows.append(record)
    return {'rows':rows,'method':f'Linear-interpolated sample percentiles; first {warmup_steps} decode tokens excluded; '
                '16/128 windows are prefixes of the SAME 128-token run, not independent repetitions. '
                'Device event intervals are not additive end-to-end latency. CPU weight offload traffic is excluded.'}


def write_csv(path,report):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with open(path,'w',newline='') as f:
        names=['request_id','mode','context','window_tokens','samples','token_p50_ms','token_p95_ms','token_p99_ms',
               'tokens_per_second','gpu0_peak_bytes','gpu1_peak_bytes','layer3_p50_ms','offload_p50_ms']
        writer=csv.DictWriter(f,fieldnames=names);writer.writeheader()
        for r in report['rows']:
            writer.writerow({'request_id':r['request_id'],'mode':r['mode'],'context':r['context'],
                'window_tokens':r['window_tokens'],'samples':r['token_ms']['count'],
                'token_p50_ms':r['token_ms']['p50'],'token_p95_ms':r['token_ms']['p95'],
                'token_p99_ms':r['token_ms']['p99'],'tokens_per_second':r['tokens_per_second'],
                'gpu0_peak_bytes':r['gpu_memory']['0']['peak'],'gpu1_peak_bytes':r['gpu_memory']['1']['peak'],
                'layer3_p50_ms':r['layer3_wall_ms']['p50'],'offload_p50_ms':r['offload_wall_ms']['p50']})
