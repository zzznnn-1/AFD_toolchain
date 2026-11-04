import _bootstrap
import argparse
import faulthandler
import json
import signal
import traceback
from pathlib import Path
import torch
from pnm.evidence import provenance, finish_provenance
from pnm.models.qwen38.loader import load_model, prepare_model
from pnm.models.qwen38.preflight import DeploymentBlocked
from pnm.models.qwen38.runner import Runner
from pnm.models.qwen38.compare import compare_runs
from pnm.backends.gpu_emulator import GPUEmulator
from pnm.runtime import Runtime
from pnm.runtime.metrics import Metrics
from pnm.validation import write_json, check_close


def main():
    faulthandler.register(signal.SIGUSR1)
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/pnm.json')
    p.add_argument('--mode', choices=['local','shadow_compare','shadow_remote'], default='shadow_remote')
    p.add_argument('--model-path')
    p.add_argument('--steps', type=int, default=32)
    p.add_argument('--prompt')
    p.add_argument('--prompt-tokens', type=int)
    p.add_argument('--resident-layers', help='auto or 0..64; CPU storage with execution on GPU0')
    p.add_argument('--fully-resident', action='store_true')
    p.add_argument('--strict-deployment', action='store_true', help='require all weights on GPU0 and exclusive GPU1 compute')
    p.add_argument('--suite', action='store_true', help='teacher-forcing + greedy; short/512/4096 contexts')
    p.add_argument('--output', default='reports/raw/P3_run.json')
    a = p.parse_args()
    if a.steps < 1 or (a.prompt_tokens is not None and a.prompt_tokens < 1):
        p.error('steps and prompt tokens must be positive')
    config = json.loads(Path(a.config).read_text())
    if a.resident_layers is not None:
        config['resident_layers'] = 'auto' if a.resident_layers == 'auto' else int(a.resident_layers)
    if config.get('layer_id') != 3 or config.get('dtype') != 'bfloat16':
        p.error('this runner supports layer_id=3 and bfloat16 only')
    config['fully_resident'] = a.fully_resident
    torch.manual_seed(config['seed'])
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    evidence = provenance(_bootstrap.ROOT)
    output = Path(a.output)
    if output.suffix != '.json': p.error('--output must end with .json')
    metrics = Metrics(output.with_suffix('.jsonl'), run_id=evidence['run_id'])
    result = {'status':'RUNNING', 'suite':a.suite, 'configuration':config,
              'provenance':evidence, 'runs':[], 'comparisons':[]}
    runtime, code = None, 0
    write_json(output, result)
    try:
        path, preflight = prepare_model(config, a.model_path, strict=a.strict_deployment,
                                        needs_emulator=a.suite or a.mode!='local')
        result['preflight'] = preflight
        write_json(output, result)
        if preflight['status'] != 'PASS': raise DeploymentBlocked(preflight)
        model, tokenizer, facts = load_model(config, path, preflight)
        result['model'] = facts
        write_json(output, result)
        if a.suite or a.mode != 'local':
            runtime = Runtime(GPUEmulator(config['source_device'], config['emulator_device'], config['capacity'], metrics))
        runner = Runner(model, tokenizer, runtime, config['capacity'], config['attention_atol'], config['attention_rtol'])
        if a.suite:
            for label, length, steps in [('short',None,32), ('medium',512,128), ('context4096',4096,32)]:
                prompt = runner.prompt(length=length)
                local, lc, lp = runner.run(prompt, 'local', steps, request_id=label+'-local')
                result['runs'].append(local)
                write_json(output, result)
                for mode in ['shadow_compare','shadow_remote']:
                    remote, rc, rp = runner.run(prompt, mode, steps, fixed_tokens=local['input_tokens'], request_id=label+'-'+mode)
                    comparison = compare_runs(lc, rc, config['logits_atol'], config['logits_rtol'])
                    comparison.update(case=label, mode=mode, prefill=check_close(rp, lp, atol=0, rtol=0))
                    result['comparisons'].append(comparison)
                    result['runs'].append(remote)
                    write_json(output, result)
                    if comparison['status'] != 'PASS': raise RuntimeError(f'{label} {mode} teacher forcing mismatch')
                    del rc
                if label in ['short','medium']:
                    greedy, _, _ = runner.run(prompt, 'shadow_remote', steps, request_id=label+'-remote-greedy', capture=False)
                    greedy['matches_local_generated_tokens'] = greedy['input_tokens'] == local['input_tokens']
                    result['runs'].append(greedy)
                    write_json(output, result)
                    if not greedy['matches_local_generated_tokens']:
                        raise RuntimeError(f'{label} local/remote greedy token sequences diverged')
                del lc
        else:
            prompt = runner.prompt(a.prompt, a.prompt_tokens)
            summary, _, _ = runner.run(prompt, a.mode, a.steps, capture=False)
            result['runs'].append(summary)
        result['status'] = 'PASS'
    except DeploymentBlocked as e:
        result.update(status='BLOCKED', exception=str(e), preflight=e.report)
        code = 2
    except BaseException as e:
        result.update(status='FAIL', exception=repr(e), traceback=traceback.format_exc())
        traceback.print_exc()
        code = 1
    finally:
        try:
            if runtime: runtime.close()
            finish_provenance(evidence, _bootstrap.ROOT)
        except BaseException as e:
            result.update(status='FAIL', finalization_error=repr(e))
            code = 1
        finally:
            metrics.close()
            result['metrics'] = metrics.summary()
            write_json(output, result)
            print('RESULT', result['status'], output, flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
