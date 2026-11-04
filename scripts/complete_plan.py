"""Execute all available P0-P5 simulator work and record P6 external prerequisites."""
import _bootstrap
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from pnm.evidence import provenance,finish_provenance,timestamp
from pnm.validation import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);a=p.parse_args()
    root=Path(a.output_dir).resolve()
    if root.exists(): p.error('choose a new output directory; existing evidence is preserved')
    raw=root/'raw';raw.mkdir(parents=True)
    python=sys.executable
    env=dict(os.environ,HF_HUB_DISABLE_PROGRESS_BARS='1',TOKENIZERS_PARALLELISM='false')
    evidence=provenance(_bootstrap.ROOT)
    tasks=[
        ('P0_environment',[python,'scripts/check_environment.py','--output',str(raw/'environment.json')]),
        ('placement',[python,'scripts/check_placement.py','--config','configs/complete.json','--output',str(raw/'placement.json')]),
        ('fp8_kernel_probe',[python,'scripts/probe_fp8.py','--output',str(raw/'fp8_probe.json')]),
        ('P1_attention',[python,'scripts/test_attention.py','--output',str(raw/'P1_attention.json')]),
        ('P2_emulator',[python,'scripts/test_emulator.py','--prefill','4096','--steps','128','--output',str(raw/'P2_emulator.json')]),
        ('unit_tests',[python,'-m','pytest','-q','--junitxml='+str(raw/'unit_tests.xml'),'--evidence-json='+str(raw/'unit_tests.json')]),
        ('P5_stress',[python,'-u','scripts/stress_emulator.py','--output',str(raw/'P5_stress.json')]),
        ('P6_external',[python,'scripts/check_external.py','--output',str(raw/'P6_external.json')]),
        ('P4_suite',[python,'-u','scripts/run_strict_suite.py','--output-dir',str(raw)]),
        ('complete_acceptance',[python,'scripts/verify_complete.py','--artifact-dir',str(raw)])]
    commands=[];code=0
    try:
        with (raw/'environment.lock.txt').open('w') as f:
            subprocess.run([python,'-m','pip','freeze'],stdout=f,check=True)
        write_json(raw/'host_context.json',{'cpu_affinity':sorted(os.sched_getaffinity(0)),
            'numa_policy':'not changed','cwd':str(_bootstrap.ROOT),'pid':os.getpid()})
        for name,argv in tasks:
            print('RUN',name,flush=True)
            started,start=timestamp(),time.perf_counter();monitor=None
            with (raw/(name+'.log')).open('w') as log:
                process=subprocess.Popen(argv,cwd=_bootstrap.ROOT,stdout=log,stderr=subprocess.STDOUT,env=env)
                try:
                    if name=='P4_suite':
                        monitor=subprocess.Popen([python,'scripts/monitor_gpus.py','--pid',str(process.pid),
                            '--output',str(raw/'P4_suite_gpu.jsonl')],cwd=_bootstrap.ROOT,
                            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                    rc=process.wait()
                finally:
                    if process.poll() is None:
                        process.terminate();process.wait(timeout=30)
                    if monitor:
                        try: monitor.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            monitor.terminate();monitor.wait(timeout=5)
            commands.append({'name':name,'argv':argv,'started_at':started,'seconds':time.perf_counter()-start,'exit_code':rc})
            write_json(raw/'commands.json',commands)
            print('DONE',name,'exit',rc,flush=True)
            if rc!=0: code=1;break
        finish_provenance(evidence,_bootstrap.ROOT)
    except BaseException as exc:
        code=1;evidence['error']=repr(exc)
    finally:
        write_json(raw/'pipeline.json',{'status':'PASS' if code==0 else 'FAIL','provenance':evidence})
        subprocess.run([python,'scripts/write_complete_report.py','--output-dir',str(root)],cwd=_bootstrap.ROOT,check=True)
    print('SOFTWARE PLAN', 'PASS' if code==0 else 'FAIL',root,flush=True)
    return code


if __name__=='__main__': raise SystemExit(main())
