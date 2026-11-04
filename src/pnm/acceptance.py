"""Validate actual traces and numerical records; never infer PASS from file existence."""
import math
from collections import Counter, defaultdict


class VerificationError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise VerificationError(message)


def verify_provenance(evidence, current):
    require(evidence.get('schema_version') == 2, 'missing execution-time provenance')
    require(evidence.get('source_unchanged') is True, 'source changed during test')
    require(evidence.get('source_sha256') == current, 'tested source differs from current source')
    require(bool(evidence.get('run_id')) and bool(evidence.get('finished_at')), 'incomplete provenance')


def verify_trace(records, run_id, runs):
    commands, states, totals = defaultdict(list), {}, Counter()
    stage_counts, stage_bytes, stage_ms = Counter(), Counter(), Counter()
    for r in records:
        require(r.get('run_id') == run_id, 'trace belongs to a different run')
        commands[r['command_id']].append(r)
        stage_counts[r['stage']] += 1
        stage_bytes[r['stage']] += r.get('bytes', 0)
        require(math.isfinite(r.get('ms', 0)) and r.get('ms', 0) >= 0, 'invalid timing')
        stage_ms[r['stage']] += r.get('ms', 0)
    remote_runs = {r['request_id']: r for r in runs if r['mode'] != 'local'}
    require(len(remote_runs) == sum(r['mode'] != 'local' for r in runs), 'duplicate request IDs')
    for group in commands.values():
        c = [r for r in group if r['stage'] == 'command']
        require(len(c) == 1 and group[-1] is c[0], 'command completion missing, duplicated or out of order')
        c = c[0]
        require(c['status'] == 'PASS', 'failed command in acceptance trace')
        require(c['dtype'] == 'bfloat16' and c['layout'] == 'BHTD' and c['layer_id'] == 3, 'wrong command contract')
        require(c['request_id'] in remote_runs, 'unknown request in trace')
        h, op, run = c['handle'], c['op'], remote_runs[c['request_id']]
        transfers = [r for r in group if r['stage'] not in ('command', 'attention')]
        actual = Counter((r['stage'], r['tensor'], r['bytes']) for r in transfers)
        require(all(r.get('pinned') is True for r in transfers), 'non-pinned transfer')
        if op == 'alloc_cache':
            require(h not in states and c['length'] == 0, 'invalid allocation')
            states[h] = {'request':c['request_id'], 'length':0, 'initialized':False, 'freed':False, 'steps':0}
            require(len(group) == 1, 'allocation unexpectedly transferred data')
        else:
            require(h in states, 'command references unallocated handle')
            s = states[h]
            require(not s['freed'] and s['request'] == c['request_id'], 'use after free or wrong owner')
            if op == 'init_cache':
                require(not s['initialized'] and c['length'] == run['prompt_tokens'], 'prefill repeated or wrong length')
                nbytes = run['prompt_tokens'] * 4 * 256 * 2
                expected = Counter((stage, name, nbytes) for stage in ('cuda:0_d2h','cuda:1_h2d') for name in ('k','v'))
                require(actual == expected and len(group) == 5, 'prefill staging mismatch')
                s.update(initialized=True, length=run['prompt_tokens'])
            elif op == 'append_and_attend':
                require(s['initialized'] and c['token_position'] == c['expected_length'] == s['length'], 'append position mismatch or replay')
                require(c['length'] == s['length'] + 1, 'append did not commit one token')
                expected = Counter((stage, name, size) for stage in ('cuda:0_d2h','cuda:1_h2d')
                                   for name,size in [('q',12288),('k',2048),('v',2048)])
                expected.update((stage,'o',12288) for stage in ('cuda:1_d2h','cuda:0_h2d'))
                require(actual == expected, 'decode transfer duplicated, missing or retransmitted history')
                kernels = [r for r in group if r['stage'] == 'attention']
                require(len(kernels) == 1 and kernels[0]['device'] == 'cuda:1' and kernels[0]['length'] == c['length'], 'GPU1 kernel evidence missing')
                require(len(group) == 10, 'unexpected decode trace records')
                s.update(length=c['length'], steps=s['steps']+1)
            elif op == 'get_cache_info':
                require(c['length'] == s['length'] and len(group) == 1, 'status length mismatch')
            elif op == 'free_cache':
                require(c['length'] == s['length'] and len(group) == 1, 'free length mismatch')
                require(s['steps'] == run['decode_steps'], 'decode count does not match trace')
                s['freed'] = True
            else:
                raise VerificationError(f'unknown operation {op}')
        totals[op] += 1
    require(len(states) == len(remote_runs) and all(s['freed'] for s in states.values()), 'cache leaked or run omitted')
    return dict(totals), {'counts':dict(stage_counts), 'bytes':dict(stage_bytes), 'milliseconds':dict(stage_ms)}


def verify_p3(result, trace, manifest):
    require(result.get('status') == 'PASS' and result.get('suite') is True, 'full P3 suite did not pass')
    verify_provenance(result['provenance'], manifest)
    require(result['preflight']['status'] == 'PASS', 'placement preflight failed')
    model = result['model']
    require(model['model_id'] == 'Qwen/Qwen3.8-27B-FP8' and model['revision'] == result['configuration']['revision'], 'wrong model/revision')
    require(model['fp8_linear_count'] == 400 and not any(model['loading_info'].values()), 'checkpoint loading mismatch')
    require(all(d not in (1, 'cuda:1') for d in model['hf_device_map'].values()), 'model assigned to GPU1')
    runs = {r['request_id']: r for r in result['runs']}
    require(len(runs) == len(result['runs']) == 11, 'missing or duplicated suite runs')
    expected = {'short':(None,32), 'medium':(512,128), 'context4096':(4096,32)}
    for case, (length, steps) in expected.items():
        local = runs[case+'-local']
        if length: require(local['prompt_tokens'] == length, 'wrong test context')
        for mode in ('local','shadow_compare','shadow_remote'):
            r = runs[case+'-'+mode]
            require(r['mode'] == mode and r['decode_steps'] == steps, 'wrong mode or decode steps')
            require(r['prompt_token_ids'] == local['prompt_token_ids'] and len(r['prompt_token_ids']) == r['prompt_tokens'], 'different prompt tokens')
            require(len(r['input_tokens']) == len(r['predicted_tokens']) == steps, 'missing decoded tokens')
            require(r['input_tokens'] == local['input_tokens'], 'teacher forcing sequence differs')
        if case != 'context4096':
            g = runs[case+'-remote-greedy']
            require(g['matches_local_generated_tokens'] and g['input_tokens'] == local['input_tokens'], 'greedy sequences differ')
            require(g['prompt_token_ids'] == local['prompt_token_ids'] and g['decode_steps'] == steps, 'greedy coverage mismatch')
    for r in runs.values():
        require(r['adapter_restored'], 'adapter was not uninstalled')
        require(len(r['decoder_execution_devices']) == 64 and set(r['decoder_execution_devices'].values()) == {'cuda:0'}, 'decoder execution not on GPU0')
        if r['mode'] == 'local':
            require(r['remote_calls'] == r['remote_outputs_used'] == 0, 'local mode used remote data')
            continue
        steps, start = r['decode_steps'], r['prompt_tokens']
        require(r['remote_calls'] == steps, 'remote calls missing')
        require(r['remote_outputs_used'] == (steps if r['mode']=='shadow_remote' else 0), 'remote output was discarded')
        require(r['prefill_cache']['sequence_length'] == start and r['final_cache']['sequence_length'] == start+steps, 'remote cache length mismatch')
        require(len(r['position_records']) == len(r['raw_local_remote_errors']) == steps, 'per-step verification missing')
        for i, (p,e) in enumerate(zip(r['position_records'], r['raw_local_remote_errors'])):
            require(p['position'] == e['position'] == start+i, 'wrong per-step position')
            require(p['native_length'] == p['remote_length'] == start+i+1, 'native/remote cache mismatch')
            require(e['pass'] and e['finite'], 'per-step attention mismatch')
    pairs = {(c['case'], c['mode']) for c in result['comparisons']}
    require(len(result['comparisons']) == 6 and pairs == {(case,mode) for case in expected for mode in ('shadow_compare','shadow_remote')}, 'missing numerical comparisons')
    for c in result['comparisons']:
        require(c['status'] == 'PASS' and len(c['steps']) == expected[c['case']][1], 'numerical comparison incomplete')
        require(c['prefill']['pass'] and c['prefill']['max_abs'] == 0, 'prefill differed')
        for row in c['steps']:
            for key in ('raw_o','attention_output','decoder_output','logits'):
                e = row[key]
                require(e['pass'] and e['finite'], f'{key} failed')
                require(all(math.isfinite(e[n]) for n in ('max_abs','mean_abs','max_relative','rms')), 'nonfinite error statistics')
    counts, summary = verify_trace(trace, result['provenance']['run_id'], result['runs'])
    for group in ('counts','bytes','milliseconds'):
        require(set(summary[group]) == set(result['metrics'][group]), 'metric stages mismatch')
        for stage,value in summary[group].items():
            require(math.isclose(value, result['metrics'][group][stage], rel_tol=1e-8, abs_tol=1e-8), 'trace/summary mismatch')
    plan = result['preflight']
    cpu_weights = any(d == 'cpu' for d in model['hf_device_map'].values())
    isolated = plan['gpu1_compute_exclusive']
    return {'functional_status':'PASS', 'deployment_status':'PASS' if not cpu_weights and isolated else 'BLOCKED',
            'cpu_weight_offload':cpu_weights, 'gpu1_compute_exclusive':isolated,
            'foreign_gpu1_processes':plan['foreign_gpu1_processes'], 'command_counts':counts,
            'remote_tokens':counts['append_and_attend'],
            'remote_output_tokens':sum(r['remote_outputs_used'] for r in runs.values()),
            'all_greedy_sequences_match':True}
