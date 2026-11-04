"""Record actual whole-device usage while a specified experiment PID is alive."""
import _bootstrap
import argparse
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--pid',type=int,required=True)
    p.add_argument('--output',default='reports/raw/gpu_monitor.jsonl')
    p.add_argument('--interval',type=float,default=2.)
    a=p.parse_args()
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    with open(a.output,'w',buffering=1) as f:
        while True:
            try: os.kill(a.pid,0)
            except ProcessLookupError: break
            result=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu,power.draw',
                                   '--format=csv,noheader,nounits'],text=True,capture_output=True,timeout=20)
            f.write(json.dumps({'time':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'pid':a.pid,
                                'csv':result.stdout,'returncode':result.returncode})+'\n')
            time.sleep(a.interval)
