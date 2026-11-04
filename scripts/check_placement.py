"""No model weight allocation; exit 2 means a measured deployment blocker."""
import _bootstrap
import argparse
import json
from pathlib import Path
from pnm.models.qwen38.loader import prepare_model
from pnm.validation import write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/pnm.json')
    p.add_argument('--model-path')
    p.add_argument('--strict-deployment', action='store_true')
    p.add_argument('--fully-resident', action='store_true')
    p.add_argument('--output', default='reports/raw/placement.json')
    a = p.parse_args()
    config = json.loads(Path(a.config).read_text())
    config['fully_resident'] = a.fully_resident
    _, report = prepare_model(config, a.model_path, strict=a.strict_deployment)
    write_json(a.output, report)
    print(json.dumps({k: report[k] for k in ['status', 'selected_resident_layers',
          'resident_parameter_bytes', 'cpu_parameter_bytes', 'required_gpu0_bytes',
          'fully_resident_fits', 'foreign_gpu1_processes', 'blocking_reasons'] if k in report}, indent=2))
    return 0 if report['status'] == 'PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
