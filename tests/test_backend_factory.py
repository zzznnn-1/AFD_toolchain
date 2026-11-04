import pytest
from pnm.backends.registry import create_backend
from pnm.runtime import Runtime
from pnm.protocol import CacheSpec,PNMError


def test_backend_factory_runs_same_runtime_contract():
    with Runtime(create_backend({'kind':'gpu_emulator','max_tokens':16})) as rt:
        handle=rt.alloc_cache('factory',3,CacheSpec(16))
        assert rt.get_cache_info(handle).capacity==16
        assert rt.describe()['protocol_version']==1


def test_missing_real_driver_fails_without_emulator_fallback():
    with pytest.raises(PNMError,match='DRIVER_UNAVAILABLE'):
        create_backend({'kind':'plugin'})
    with pytest.raises(PNMError,match='DRIVER_UNAVAILABLE'):
        create_backend({'kind':'plugin','entrypoint':'nonexistent_pnm_asic_driver:open'})


def test_plugin_entrypoint_runs_real_backend(monkeypatch):
    import sys
    from types import ModuleType
    from pnm.backends.gpu_emulator import GPUEmulator
    module=ModuleType('pnm_test_plugin')
    module.open=lambda options,metrics=None: GPUEmulator(max_tokens=options['capacity'],metrics=metrics)
    monkeypatch.setitem(sys.modules,'pnm_test_plugin',module)
    with Runtime(create_backend({'kind':'plugin','entrypoint':'pnm_test_plugin:open','options':{'capacity':8}})) as rt:
        h=rt.alloc_cache('plugin',3,CacheSpec(8))
        assert rt.get_cache_info(h).capacity==8
