import json
import pytest
import torch
from pnm.backends.gpu_emulator import GPUEmulator
from pnm.backends.faults import FaultBackend
from pnm.protocol import CacheSpec,Command,Op,PNMError
from pnm.protocol.codec import encode,decode
from pnm.runtime import Runtime


def data():
    torch.manual_seed(20261104)
    return (torch.randn(1,24,1,256,device='cuda:0',dtype=torch.bfloat16),
            torch.randn(1,4,1,256,device='cuda:0',dtype=torch.bfloat16),
            torch.randn(1,4,1,256,device='cuda:0',dtype=torch.bfloat16))


@pytest.mark.parametrize('fault,code,committed',[
    ('timeout_before','TIMEOUT_UNKNOWN',False),('timeout_after','TIMEOUT_UNKNOWN',True),
    ('duplicate_completion','INVALID_COMPLETION',True)])
def test_uncertain_completion_poison_prevents_double_append(fault,code,committed):
    q,k,v=data()
    wrapped=FaultBackend(GPUEmulator(max_tokens=8))
    with Runtime(wrapped) as rt:
        h=rt.alloc_cache('fault',3,CacheSpec(8)); rt.init_cache(h,k,v)
        wrapped.arm(fault)
        with pytest.raises(PNMError,match=code): rt.append_and_attend(h,q,k,v,1)
        assert rt.get_cache_info(h).sequence_length==(2 if committed else 1)
        with pytest.raises(PNMError,match='POISONED'):
            rt.append_and_attend(h,q,k,v,2 if committed else 1)
        assert rt.get_cache_info(h).sequence_length==(2 if committed else 1)


def test_reset_lost_cancel_and_resource_limit():
    q,k,v=data()
    backend=GPUEmulator(max_tokens=4,max_cache_bytes=4*4096)
    with Runtime(backend) as rt:
        h=rt.alloc_cache('first',3,CacheSpec(4)); rt.init_cache(h,k,v)
        with pytest.raises(PNMError,match='RESOURCE_EXHAUSTED'): rt.alloc_cache('too-many',3,CacheSpec(1))
        backend.reset()
        with pytest.raises(PNMError,match='STALE_HANDLE'): rt.get_cache_info(h)
        h2=rt.alloc_cache('second',3,CacheSpec(4)); rt.init_cache(h2,k,v)
        assert rt.get_cache_info(h2).generation==2
        rt.cancel_request('second')
        assert not backend.manager.entries
        with pytest.raises(PNMError,match='CANCELLED'): rt.alloc_cache('second',3,CacheSpec(4))
        h3=rt.alloc_cache('third',3,CacheSpec(4)); rt.init_cache(h3,k,v)
        backend.lose_device()
        with pytest.raises(PNMError,match='DEVICE_LOST'): rt.append_and_attend(h3,q,k,v,1)
        with pytest.raises(PNMError,match='POISONED'): rt.append_and_attend(h3,q,k,v,1)
    assert backend.closed and not backend.manager.entries


def test_command_envelope_round_trip_and_corruption():
    q,k,v=data()
    c=Command('cmd',Op.APPEND,'req',3,'handle',expected_length=7,token_position=7,generation=2)
    packet=encode(c,{'q':q,'k':k,'v':v,'mask':None})
    body=decode(packet)
    assert body['command']['generation']==2
    assert sum(t['bytes'] for t in body['tensors'].values())==16384
    assert body['tensors']['q']['shape']==[1,24,1,256]
    with pytest.raises(PNMError,match='CHECKSUM'): decode(packet[:-1]+bytes([packet[-1]^1]))
    with pytest.raises(PNMError,match='FRAME'): decode(packet[:4])


def test_actual_staging_exception_does_not_commit_length(monkeypatch):
    q,k,v=data()
    backend=GPUEmulator(max_tokens=8)
    with Runtime(backend) as rt:
        h=rt.alloc_cache('copy-failure',3,CacheSpec(8));rt.init_cache(h,k,v)
        copy=backend.transport.transfer
        def fail(tensor,name,command_id):
            if name=='q': raise RuntimeError('injected failure at Q staging after writing pending KV')
            return copy(tensor,name,command_id)
        monkeypatch.setattr(backend.transport,'transfer',fail)
        with pytest.raises(PNMError,match='DEVICE_ERROR'): rt.append_and_attend(h,q,k,v,1)
        assert rt.get_cache_info(h).sequence_length==1
        monkeypatch.setattr(backend.transport,'transfer',copy)
        rt.append_and_attend(h,q,k,v,1)
        assert rt.get_cache_info(h).sequence_length==2
