"""Inspect checkpoint headers and measured device capacity before weight allocation."""
import csv
import json
import os
import struct
import subprocess
from collections import defaultdict
from pathlib import Path
import torch
from pnm.evidence import sha256, timestamp


class DeploymentBlocked(RuntimeError):
    def __init__(self, report):
        self.report = report
        super().__init__('; '.join(report['blocking_reasons']))


def inventory(snapshot):
    snapshot = Path(snapshot)
    index = json.loads((snapshot / 'model.safetensors.index.json').read_text())['weight_map']
    modules, files = defaultdict(int), {}
    for filename in sorted(set(index.values())):
        path = snapshot / filename
        with path.open('rb') as f:
            size = struct.unpack('<Q', f.read(8))[0]
            if size > 32 * 1024**2:
                raise ValueError(f'invalid safetensors header size: {filename}')
            header = json.loads(f.read(size))
        files[filename] = path.stat().st_size
        for name, entry in header.items():
            if name == '__metadata__' or name.startswith('mtp.'):
                continue  # Transformers text/vision model has no MTP module.
            if index.get(name) != filename:
                raise ValueError(f'checkpoint index/header disagreement: {name}')
            offsets = entry['data_offsets']
            if not 0 <= offsets[0] <= offsets[1] <= files[filename] - 8 - size:
                raise ValueError(f'truncated checkpoint: {filename}/{name}')
            if name.startswith('model.language_model.layers.'):
                group = '.'.join(name.split('.')[:4])
            elif name.startswith('model.visual.'):
                group = 'model.visual'
            else:
                group = name.rsplit('.', 1)[0]
            modules[group] += offsets[1] - offsets[0]
        missing = [name for name, shard in index.items() if shard == filename and name not in header]
        if missing:
            raise ValueError(f'checkpoint missing tensors: {missing}')
    return {'parameter_bytes': sum(modules.values()), 'module_bytes': dict(modules),
            'files': files, 'config_sha256': sha256(snapshot / 'config.json'),
            'index_sha256': sha256(snapshot / 'model.safetensors.index.json')}


def hardware_snapshot():
    gpus = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        uuid = str(props.uuid)
        gpus.append({'index': i, 'name': props.name, 'uuid': uuid if uuid.startswith('GPU-') else 'GPU-' + uuid,
                     'free_bytes': free, 'total_bytes': total, 'capability': [props.major, props.minor]})
    cmd = ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory',
           '--format=csv,noheader,nounits']
    processes, probe_error = [], None
    try:
        p = subprocess.run(cmd, text=True, capture_output=True, check=True, timeout=20)
        for row in csv.reader(p.stdout.splitlines()):
            if not row:
                continue
            uuid, pid, name, mib = [x.strip() for x in row]
            processes.append({'gpu_uuid': uuid, 'pid': int(pid), 'name': name,
                              'used_mib': int(mib) if mib.isdigit() else None})
    except (OSError, ValueError, subprocess.SubprocessError) as e:
        probe_error = str(e)
    return {'time': timestamp(), 'gpus': gpus, 'compute_processes': processes,
            'process_probe_error': probe_error, 'current_pid': os.getpid()}


def placement_plan(config, weights, hardware, strict=False, needs_emulator=True):
    """Pure planning function, tested with both constrained and sufficient hardware."""
    if config['source_device'] != 'cuda:0' or config['emulator_device'] != 'cuda:1':
        raise ValueError('audited mapping requires model cuda:0 and emulator cuda:1')
    report = {'checked_at': timestamp(), 'hardware': hardware, 'weights': weights,
              'policy': 'strict' if strict else 'functional', 'blocking_reasons': []}
    gpus = hardware['gpus']
    if len(gpus) < (2 if needs_emulator else 1):
        report.update(status='BLOCKED', blocking_reasons=['required CUDA devices unavailable'])
        return report
    source = gpus[0]
    reserve = int(config.get('workspace_reserve_gib', 4.0) * 2**30)
    if reserve < 0:
        raise ValueError('workspace reserve must be nonnegative')
    modules = weights['module_bytes']
    expected = {f'model.language_model.layers.{i}' for i in range(64)}
    if not expected.issubset(modules):
        raise ValueError('checkpoint does not have all 64 decoder layers')
    baseline = {'model.visual': 'cpu', 'model.language_model.embed_tokens': 'cpu', 'lm_head': 'cpu',
                'model.language_model.norm': 0, 'model.language_model.rotary_emb': 0}

    def allocation(resident):
        mapping = dict(baseline)
        mapping.update({f'model.language_model.layers.{i}': 0 if i < resident else 'cpu' for i in range(64)})
        resident_bytes = sum(size for name, size in modules.items() if mapping.get(name) == 0)
        cpu_bytes = sum(size for name, size in modules.items() if mapping.get(name) == 'cpu')
        if resident_bytes + cpu_bytes != weights['parameter_bytes']:
            raise ValueError('unmapped checkpoint module')
        # Text only: visual weights are never materialized. Accelerate stages each
        # needed module onto GPU0; whole-layer maximum is a conservative bound.
        transient = max((size for name, size in modules.items()
                         if mapping.get(name) == 'cpu' and name != 'model.visual'), default=0)
        return mapping, resident_bytes, cpu_bytes, transient

    feasible = [n for n in range(65) if sum(allocation(n)[i] for i in (1, 3)) + reserve <= source['free_bytes']]
    requested = config.get('resident_layers', 'auto')
    if requested != 'auto' and (type(requested) is not int or not 0 <= requested <= 64):
        raise ValueError('resident_layers must be auto or an integer 0..64')
    if config.get('fully_resident', False):
        selected, mapping = 64, {'': 0}
        resident_bytes, cpu_bytes, transient = weights['parameter_bytes'], 0, 0
    else:
        selected = max(feasible, default=0) if requested == 'auto' else requested
        mapping, resident_bytes, cpu_bytes, transient = allocation(selected)
    required = resident_bytes + transient + reserve
    full_fit = weights['parameter_bytes'] + reserve <= source['free_bytes']
    gpu1_bytes = 2 * config['capacity'] * 4 * 256 * 2 * 2 + 256 * 2**20
    foreign = []
    if len(gpus) > 1:
        foreign = [p for p in hardware['compute_processes'] if p['gpu_uuid'] == gpus[1]['uuid']
                   and p['pid'] != hardware['current_pid']]
    isolated = hardware['process_probe_error'] is None and not foreign and len(gpus) > 1
    report.update(selected_resident_layers=selected, requested_resident_layers=requested,
                  device_map=mapping, resident_parameter_bytes=resident_bytes,
                  cpu_parameter_bytes=cpu_bytes, largest_staged_module_bytes=transient,
                  workspace_reserve_bytes=reserve, required_gpu0_bytes=required,
                  estimated_gpu1_bytes=gpu1_bytes if needs_emulator else 0,
                  fully_resident_fits=full_fit, foreign_gpu1_processes=foreign,
                  gpu1_compute_exclusive=isolated,
                  deployment_status='PASS' if cpu_bytes == 0 and isolated else 'BLOCKED')
    reasons = report['blocking_reasons']
    if required > source['free_bytes']:
        reasons.append(f'GPU0 requires {required} bytes including staging/reserve, free={source["free_bytes"]}')
    if needs_emulator and gpu1_bytes > gpus[1]['free_bytes']:
        reasons.append(f'GPU1 emulator budget {gpu1_bytes} exceeds free memory {gpus[1]["free_bytes"]}')
    if strict:
        if cpu_bytes:
            reasons.append(f'strict deployment forbids CPU weight storage ({cpu_bytes} bytes); fully resident fits={full_fit}')
        if not isolated:
            reasons.append(f'GPU1 compute exclusivity unverified: processes={foreign}, probe_error={hardware["process_probe_error"]}')
    report['status'] = 'BLOCKED' if reasons else 'PASS'
    return report
