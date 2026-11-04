"""Exit 0: functional evidence passes; 1: invalid evidence; 2: strict deployment blocked."""
import _bootstrap
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from pnm.acceptance import require, verify_p3, verify_provenance
from pnm.evidence import source_manifest, sha256, timestamp
from pnm.validation import write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--artifact-dir', default='reports/raw')
    p.add_argument('--strict-deployment', action='store_true')
    a = p.parse_args()
    root = Path(a.artifact_dir)
    result = {'status':'FAIL', 'verified_at':timestamp()}
    try:
        current = source_manifest(_bootstrap.ROOT)
        def read(name): return json.loads((root / name).read_text())
        env, fp8, p1, p2, p3, tests = [read(n) for n in ['environment.json','fp8_probe.json',
                'P1_attention.json','P2_emulator.json','P3_suite.json','unit_tests.json']]
        for item in (env, fp8, p1, p2, tests): verify_provenance(item['provenance'], current)
        require(len(env['gpus']) >= 2 and fp8['status'] == 'PASS' and fp8['finite'], 'P0 failed')
        require(p1['status'] == 'PASS' and all(c['pass'] for c in p1['cases']), 'P1 failed')
        for case in ('prefill','decode','append'):
            require({c['length'] for c in p1['cases'] if c.get('case') == case} >= {1,16,128,1024,4096}, 'P1 coverage missing')
        require(p2['status'] == 'PASS' and p2['steps'] >= 128 and len(p2['steps_detail']) == p2['steps'], 'P2 incomplete')
        require(p2['pinned'] and all(c['pass'] for c in p2['steps_detail']), 'P2 data path failed')
        require(tests['status'] == 'PASS' and tests['exit_code'] == 0, 'unit tests failed')
        xml = ET.parse(root / 'unit_tests.xml').getroot()
        suites = list(xml) if xml.tag == 'testsuites' else [xml]
        require(all(int(s.get(k,0)) == 0 for s in suites for k in ('failures','errors','skipped')), 'failed/skipped unit tests')
        test_count = sum(int(s.get('tests',0)) for s in suites)
        require(test_count > 0 and test_count == tests['collected'], 'test count mismatch')
        trace = [json.loads(line) for line in (root/'P3_suite.jsonl').read_text().splitlines()]
        validated = verify_p3(p3, trace, current)
        overall = validated['deployment_status']
        result.update(validated, status=overall, stages={'P0':'PASS','P1':'PASS','P2':'PASS','P3':overall},
                      unit_tests=test_count, source_sha256=current,
                      execution_commit=p3['provenance']['git_commit'], execution_branch=p3['provenance']['git_branch'],
                      run_id=p3['provenance']['run_id'],
                      artifact_sha256={name:sha256(root/name) for name in ['environment.json','fp8_probe.json',
                          'P1_attention.json','P2_emulator.json','P3_suite.json','P3_suite.jsonl','unit_tests.json','unit_tests.xml']})
        code = 2 if a.strict_deployment and overall == 'BLOCKED' else 0
    except Exception as e:
        result['error'] = str(e)
        code = 1
    write_json(root/'acceptance.json', result)  # Always replace stale PASS, including on errors.
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_sha256','artifact_sha256')}, indent=2))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
