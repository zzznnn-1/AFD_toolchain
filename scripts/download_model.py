import _bootstrap
import json
from pathlib import Path
from huggingface_hub import HfApi,snapshot_download
from pnm.validation import write_json


if __name__=='__main__':
    config=json.loads(Path('configs/pnm.json').read_text())
    info=HfApi().model_info(config['model_id'],revision=config['revision'],files_metadata=True)
    assert info.sha==config['revision']
    write_json('reports/raw/model_metadata.json',{'model_id':info.id,'revision':info.sha,
                'files':[{'name':s.rfilename,'bytes':s.size} for s in info.siblings]})
    path=snapshot_download(info.id,revision=info.sha,allow_patterns=['*.json','*.safetensors','*.jinja','*.txt'],max_workers=8)
    Path('reports/raw/model_path.txt').write_text(path+'\n')
    print('COMPLETE',path)
