"""Record execution-time provenance; verification never stamps old results as new."""
import hashlib
import importlib.metadata
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_manifest(root):
    root = Path(root)
    paths = []
    for directory, pattern in [('src/pnm', '*.py'), ('scripts', '*.py'), ('tests', '*.py'), ('configs', '*.json')]:
        paths.extend((root / directory).rglob(pattern))
    paths.extend(root / n for n in ['pyproject.toml', 'requirements-pnm.txt'])
    return {str(p.relative_to(root)): sha256(p) for p in sorted(paths) if p.is_file()}


def timestamp():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def provenance(root):
    root = Path(root)
    def git(*args):
        p = subprocess.run(['git', *args], cwd=root, text=True, capture_output=True)
        return p.stdout.strip() if p.returncode == 0 else None
    versions = {}
    for name in ['torch', 'transformers', 'accelerate', 'huggingface_hub', 'safetensors', 'kernels', 'triton']:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {'schema_version': 2, 'run_id': uuid4().hex, 'started_at': timestamp(),
            'python': platform.python_version(), 'executable': sys.executable, 'argv': sys.argv,
            'pid': os.getpid(), 'git_commit': git('rev-parse', 'HEAD'),
            'git_branch': git('branch', '--show-current'), 'versions': versions,
            'source_sha256': source_manifest(root)}


def finish_provenance(evidence, root):
    evidence['finished_at'] = timestamp()
    evidence['source_unchanged'] = evidence['source_sha256'] == source_manifest(root)
    if not evidence['source_unchanged']:
        raise RuntimeError('source changed during execution; results cannot certify the current code')
