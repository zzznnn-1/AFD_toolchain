import pytest
from pnm.protocol import CacheSpec, PNMError
from pnm.runtime import Runtime
from pnm.backends.gpu_emulator import GPUEmulator
import torch


def test_invalid_specs():
    for spec in [CacheSpec(0),CacheSpec(10,dtype='float8'),CacheSpec(10,layout='BTHD'),CacheSpec(10,query_heads=23)]:
        with pytest.raises(PNMError):
            spec.validate()


@pytest.fixture
def rt():
    if torch.cuda.device_count()<2:
        pytest.skip('two GPUs required')
    with Runtime(GPUEmulator(max_tokens=32)) as runtime:
        yield runtime


def tensors():
    return (torch.randn(1,24,1,256,device='cuda:0',dtype=torch.bfloat16),
            torch.randn(1,4,1,256,device='cuda:0',dtype=torch.bfloat16),
            torch.randn(1,4,1,256,device='cuda:0',dtype=torch.bfloat16))


@pytest.mark.gpu
def test_lifecycle_errors_and_idempotency(rt):
    torch.manual_seed(20261104)
    q,k,v=tensors()
    h=rt.alloc_cache('a',3,CacheSpec(3))
    with pytest.raises(PNMError,match='NOT_INITIALIZED'):
        rt.append_and_attend(h,q,k,v,0)
    rt.init_cache(h,k,v,command_id='init')
    for error,fn in [
        ('ALREADY_INITIALIZED',lambda:rt.init_cache(h,k,v)),
        ('DUPLICATE_COMMAND',lambda:rt.init_cache(h,k,v,command_id='init')),
        ('INVALID_HANDLE',lambda:rt.get_cache_info('bad')),
        ('LENGTH_POSITION',lambda:rt.append_and_attend(h,q,k,v,0)),
        ('LENGTH_POSITION',lambda:rt.append_and_attend(h,q,k,v,1,expected_length=0)),
        ('DTYPE_DEVICE',lambda:rt.append_and_attend(h,q.float(),k,v,1)),
        ('SHAPE',lambda:rt.append_and_attend(h,q,k[:,:2],v,1)),
        ('MASK',lambda:rt.append_and_attend(h,q,k,v,1,attention_mask=torch.ones(2,device='cuda:0'))),
    ]:
        with pytest.raises(PNMError,match=error): fn()
        assert rt.get_cache_info(h).sequence_length==1
    o1=rt.append_and_attend(h,q,k,v,1,command_id='append')
    saved=o1.clone()
    with pytest.raises(PNMError,match='DUPLICATE_COMMAND'):
        rt.append_and_attend(h,q,k,v,2,command_id='append')
    assert rt.get_cache_info(h).sequence_length==2
    rt.append_and_attend(h,q*2,k,v*2,2)
    torch.testing.assert_close(o1,saved,atol=0,rtol=0)
    with pytest.raises(PNMError,match='CAPACITY'): rt.append_and_attend(h,q,k,v,3)
    rt.free_cache(h)
    with pytest.raises(PNMError,match='INVALID_HANDLE'): rt.free_cache(h)


@pytest.mark.gpu
def test_ownership_and_failure_rollback(rt,monkeypatch):
    from pnm.protocol import Command,Op
    import pnm.backends.gpu_emulator as emulator
    q,k,v=tensors()
    h=rt.alloc_cache('owner',3,CacheSpec(4))
    rt.init_cache(h,k,v)
    result=rt.backend.execute(Command('wrong-owner',Op.INFO,'other',3,h),{})
    assert result.error_code=='OWNERSHIP'
    original=emulator.attend
    def fail(*a,**kw): raise RuntimeError('injected compute failure')
    monkeypatch.setattr(emulator,'attend',fail)
    with pytest.raises(PNMError,match='DEVICE_ERROR'):
        rt.append_and_attend(h,q,k,v,1)
    assert rt.get_cache_info(h).sequence_length==1
    monkeypatch.setattr(emulator,'attend',original)
    rt.append_and_attend(h,q,k,v,1)
    assert rt.get_cache_info(h).sequence_length==2


@pytest.mark.gpu
def test_padding_mask_and_repeated_alloc_free(rt):
    from pnm.validation import reference_attention,check_close
    q,k,v=tensors()
    baseline=None
    for cycle in range(12):
        h=rt.alloc_cache(str(cycle),3,CacheSpec(4))
        rt.init_cache(h,k,v)
        mask=torch.tensor([[[[False,True]]]],device='cuda:0')
        o=rt.append_and_attend(h,q,k,v*2,1,attention_mask=mask)
        check_close(o,reference_attention(q,torch.cat([k,k],2),torch.cat([v,v*2],2),1,mask))
        rt.free_cache(h)
        current=torch.cuda.memory_allocated(1)
        if baseline is None: baseline=current
        assert current<=baseline+1024*1024


@pytest.mark.gpu
def test_nondefault_producer_stream_and_handle_isolation(rt):
    from pnm.validation import reference_attention,check_close
    torch.manual_seed(20261104)
    stream=torch.cuda.Stream(device=0)
    with torch.cuda.stream(stream):
        q,k,v=tensors()
        h1=rt.alloc_cache('one',3,CacheSpec(4))
        h2=rt.alloc_cache('two',3,CacheSpec(4))
        rt.init_cache(h1,k,v)
        rt.init_cache(h2,k,v*3)
        o=rt.append_and_attend(h1,q,k,v,1)
        check_close(o,reference_attention(q,torch.cat([k,k],2),torch.cat([v,v],2),1))
        assert rt.get_cache_info(h1).sequence_length==2
        assert rt.get_cache_info(h2).sequence_length==1
        o=rt.append_and_attend(h2,q,k,v*3,1)
        check_close(o,reference_attention(q,torch.cat([k,k],2),torch.cat([v*3,v*3],2),1))
