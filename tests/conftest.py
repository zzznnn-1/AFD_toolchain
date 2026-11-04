from pathlib import Path
import pytest
from pnm.evidence import provenance, finish_provenance
from pnm.validation import write_json


def pytest_addoption(parser):
    parser.addoption('--evidence-json', help='save execution-time source hashes and test outcome')


@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart(session):
    if session.config.getoption('--evidence-json'):
        session.pnm_evidence = provenance(Path(__file__).resolve().parents[1])


def pytest_sessionfinish(session, exitstatus):
    output = session.config.getoption('--evidence-json')
    if output:
        finish_provenance(session.pnm_evidence, Path(__file__).resolve().parents[1])
        write_json(output, {'status':'PASS' if exitstatus==0 else 'FAIL', 'exit_code':int(exitstatus),
                            'collected':session.testscollected, 'provenance':session.pnm_evidence})
