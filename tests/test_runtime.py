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


@pytest.mark.parametrize('corruption',['command_id','handle','owner','length'])
def test_runtime_rejects_inconsistent_backend_completion(corruption):
    from dataclasses import replace
    backend=InMemoryControlBackend()
    rt=Runtime(backend)
    h=rt.alloc_cache('request',3,CacheSpec(32))
    original=backend.execute
    def wrong(c,p):
        result=original(c,p)
        if corruption=='command_id': result.command_id='different'
        if corruption=='handle': result.info=replace(result.info,handle='different')
        if corruption=='owner': result.info=replace(result.info,request_id='different')
        if corruption=='length': result.info=replace(result.info,sequence_length=999)
        return result
    backend.execute=wrong
    with pytest.raises(PNMError,match='INVALID_COMPLETION'): rt.get_cache_info(h)
    assert rt.handles[h].sequence_length==0
    backend.execute=original
    rt.close()
