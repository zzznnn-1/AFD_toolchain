"""Load an explicitly configured Backend implementation, without model/runtime changes."""
from importlib import import_module
from .base import Backend
from .gpu_emulator import GPUEmulator
from pnm.protocol import PNMError


def create_backend(config,metrics=None):
    kind=config.get('kind','gpu_emulator')
    if kind=='gpu_emulator':
        options={k:v for k,v in config.items() if k!='kind'}
        return GPUEmulator(metrics=metrics,**options)
    if kind!='plugin':
        raise PNMError('BACKEND_CONFIGURATION',f'unknown backend kind {kind}')
    entrypoint=config.get('entrypoint')
    if not entrypoint or ':' not in entrypoint:
        raise PNMError('DRIVER_UNAVAILABLE','provide the real driver adapter module:factory entrypoint')
    module,name=entrypoint.split(':',1)
    try:
        factory=getattr(import_module(module),name)
    except (ImportError,AttributeError) as exc:
        raise PNMError('DRIVER_UNAVAILABLE',str(exc)) from exc
    backend=factory(config.get('options',{}),metrics=metrics)
    if not isinstance(backend,Backend):
        close=getattr(backend,'close',None)
        if close: close()
        raise PNMError('BACKEND_CONTRACT','driver adapter must implement Backend')
    if backend.describe().get('protocol_version')!=1:
        backend.close()
        raise PNMError('PROTOCOL_VERSION','backend does not support protocol v1')
    return backend
