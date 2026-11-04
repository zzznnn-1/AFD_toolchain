"""Run P0->P3 sequentially and retain every exit code; never overwrite a prior run."""
import _bootstrap
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from pnm.evidence import provenance, finish_provenance, timestamp
from pnm.validation import write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output-dir', required=True)
    a = p.parse_args()
    root = Path(a.output_dir).resolve()
    if root.exists(): p.error('output directory exists; choose a new directory to preserve old evidence')
    raw = root/'raw'
    raw.mkdir(parents=True)
    evidence = provenance(_bootstrap.ROOT)
    command_log = []
    env = dict(os.environ, HF_HUB_DISABLE_PROGRESS_BARS='1', TOKENIZERS_PARALLELISM='false')
    python = sys.executable
    tasks = [
        ('P0_environment',[python,'scripts/check_environment.py','--output',str(raw/'environment.json')],{0}),
        ('placement',[python,'scripts/check_placement.py','--output',str(raw/'placement.json')],{0}),
        ('strict_placement',[python,'scripts/check_placement.py','--strict-deployment','--output',str(raw/'strict_placement.json')],{0,2}),
        ('fp8_kernel_probe',[python,'scripts/probe_fp8.py','--output',str(raw/'fp8_probe.json')],{0}),
        ('P1_attention',[python,'scripts/test_attention.py','--output',str(raw/'P1_attention.json')],{0}),
        ('P2_emulator',[python,'scripts/test_emulator.py','--output',str(raw/'P2_emulator.json')],{0}),
        ('unit_tests',[python,'-m','pytest','-q','--junitxml='+str(raw/'unit_tests.xml'),'--evidence-json='+str(raw/'unit_tests.json')],{0}),
        ('P3_suite',[python,'-u','scripts/run_qwen.py','--suite','--output',str(raw/'P3_suite.json')],{0}),
        ('acceptance',[python,'scripts/verify_results.py','--artifact-dir',str(raw)],{0}),
    ]
    code = 0
    try:
        with (raw/'environment.lock.txt').open('w') as f:
            subprocess.run([python,'-m','pip','freeze'],stdout=f,check=True)
        for name,cmd,accepted in tasks:
            print('RUN', name, flush=True)
            started, start = timestamp(), time.perf_counter()
            monitor = None
            with (raw/(name+'.log')).open('w') as f:
                process = subprocess.Popen(cmd,cwd=_bootstrap.ROOT,stdout=f,stderr=subprocess.STDOUT,env=env)
                try:
                    if name == 'P3_suite':
                        monitor = subprocess.Popen([python,'scripts/monitor_gpus.py','--pid',str(process.pid),
                                '--output',str(raw/'P3_suite_gpu.jsonl')],cwd=_bootstrap.ROOT,
                                stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                    returncode = process.wait()
                finally:
                    if process.poll() is None:
                        process.terminate()
                        process.wait(timeout=30)
                    if monitor:
                        try: monitor.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            monitor.terminate()
                            monitor.wait(timeout=5)
            command_log.append({'name':name,'argv':cmd,'started_at':started,'seconds':time.perf_counter()-start,
                                'exit_code':returncode,'accepted_exit_codes':sorted(accepted)})
            write_json(raw/'commands.json',command_log)
            print('DONE',name,'exit',returncode,flush=True)
            if returncode not in accepted:
                code = 1
                break
        finish_provenance(evidence,_bootstrap.ROOT)
    except BaseException as e:
        code = 1
        evidence['error'] = repr(e)
    finally:
        write_json(raw/'pipeline.json',{'status':'PASS' if code==0 else 'FAIL','provenance':evidence})
        subprocess.run([python,'scripts/write_reports.py','--artifact-dir',str(raw),'--report-dir',str(root)],cwd=_bootstrap.ROOT,check=True)
    print('FUNCTIONAL PIPELINE', 'PASS' if code==0 else 'FAIL',root,flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
