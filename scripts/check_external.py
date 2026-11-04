"""Record actual P6 prerequisites. No invented ioctl numbers or fake DMA execution."""
import _bootstrap
import argparse
import glob
import importlib.util
import json
import subprocess
from pnm.validation import write_json
from pnm.evidence import timestamp


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='reports/raw/P6_external.json')
    a=p.parse_args()
    probe=subprocess.run(['lspci','-nn'],capture_output=True,text=True)
    nodes={name:glob.glob('/dev/'+name+'*') for name in ('pnm','kxp','xdma')}
    result={'status':'BLOCKED','time':timestamp(),'lspci':probe.stdout,'lspci_exit_code':probe.returncode,
            'candidate_device_nodes':nodes,
            'python_packages':{n:importlib.util.find_spec(n) is not None for n in ['vllm','pnm_driver','kxp_driver']},
            'software_contract':'PASS: versioned command envelope and configurable Backend factory',
            'blocking_reasons':['No project ASIC device/driver adapter and no finalized DMA/ioctl ABI were supplied.',
                                'PCIe/CXL type, registered-memory ownership, queue/reset/completion rules need hardware confirmation.',
                                'vLLM integration is explicitly deferred in this prototype specification; no vLLM backend was run.'],
            'required_inputs':['PCI vendor/device IDs and device node','driver headers/library and version',
                'DMA buffer registration/IOMMU API','command queue and completion ABI',
                'device reset/generation semantics','KXP supported layout/dtype/mask/capacity']}
    write_json(a.output,result)
    print(json.dumps({k:v for k,v in result.items() if k!='lspci'},indent=2))
