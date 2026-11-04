"""A backend with no CUDA dependencies proves the Runtime handle boundary."""
import pytest
from pnm.runtime import Runtime
from pnm.protocol import CacheSpec,CacheInfo,Completion,Op,PNMError
from pnm.backends.base import Backend


class InMemoryControlBackend(Backend):
    def __init__(self):
        self.commands=[]
        self.closed=False
    def execute(self,c,payload):
        self.commands.append((c,payload))
        return Completion(c.command_id,'PASS',CacheInfo('opaque','request',3,0,32,'bfloat16','BHTD',131072,False))
    def close(self): self.closed=True


def test_control_only_backend_and_cleanup():
    backend=InMemoryControlBackend()
    with Runtime(backend) as rt:
        handle=rt.alloc_cache('request',3,CacheSpec(32))
        assert handle=='opaque'
        assert rt.get_cache_info(handle).capacity==32
    assert [c.op for c,_ in backend.commands]==[Op.ALLOC,Op.INFO,Op.FREE]
    assert backend.closed
    with pytest.raises(PNMError,match='CLOSED'):
        rt.alloc_cache('request',3,CacheSpec(32))
