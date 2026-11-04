import _bootstrap
import argparse
import hashlib
import importlib.metadata
import inspect
import json
import platform
import shutil
import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import torch
import transformers
from pnm.validation import write_json


def command(args):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=30)
        return {"returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    except Exception as e:
        return {"error": repr(e)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="reports/raw/environment.json")
    args = p.parse_args()
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention, Qwen3_5TextModel
    from transformers.cache_utils import Cache, DynamicCache, DynamicLayer
    classes = [Qwen3_5Attention, Qwen3_5TextModel, Cache, DynamicCache, DynamicLayer]
    versions = {}
    for name in ["torch", "transformers", "accelerate", "huggingface_hub", "safetensors", "kernels", "triton", "pytest", "flash-linear-attention"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "MISSING"
    d = {"time": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "python": platform.python_version(),
         "versions": versions, "torch_cuda": torch.version.cuda,
         "disk": shutil.disk_usage('.')._asdict(), "gpus": [], "source": {}}
    for i in range(torch.cuda.device_count()):
        prop = torch.cuda.get_device_properties(i)
        d["gpus"].append({"index": i, "name": prop.name, "total_bytes": prop.total_memory,
                          "capability": torch.cuda.get_device_capability(i), "free_bytes": torch.cuda.mem_get_info(i)[0]})
    d["peer_access"] = {f"{i}->{j}": torch.cuda.can_device_access_peer(i,j)
                         for i in range(torch.cuda.device_count()) for j in range(torch.cuda.device_count()) if i != j}
    for name, cmd in {"smi": ["nvidia-smi"], "topology": ["nvidia-smi", "topo", "-m"],
                      "p2p": ["nvidia-smi", "topo", "-p2p", "p"], "cpu": ["lscpu"],
                      "ram": ["free", "-b"], "nvcc": ["nvcc", "--version"]}.items():
        d[name] = command(cmd)
    for cls in classes:
        src = Path(inspect.getfile(cls))
        d["source"][cls.__name__] = {"path": str(src), "sha256": hashlib.sha256(src.read_bytes()).hexdigest()}
    Path('reports/raw/transformers_attention_source.py.txt').write_text(inspect.getsource(Qwen3_5Attention))
    Path('reports/raw/transformers_cache_source.py.txt').write_text(inspect.getsource(Cache.get_seq_length) + inspect.getsource(DynamicLayer))
    write_json(args.output, d)
    print(json.dumps({"output": args.output, "versions": versions, "gpus": d["gpus"], "peer": d["peer_access"]}, indent=2))


if __name__ == '__main__':
    main()
