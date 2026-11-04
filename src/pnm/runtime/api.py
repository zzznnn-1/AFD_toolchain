from uuid import uuid4
from time import perf_counter
from pnm.protocol import CacheSpec, Command, Op, PNMError
from pnm.protocol.codec import encode


class Runtime:
    """Backend-neutral synchronous v1 API. Only immutable metadata crosses ownership boundary.

    Duplicate command IDs are rejected, never replayed. IDs are scoped to a Runtime
    session, whose bounded command budget is specified by the backend.
    """
    def __init__(self, backend, metrics=None):
        self.backend = backend
        self.handles = {}
        self.closed = False
        self.metrics = metrics
        self.poisoned = {}
        self.cancelled = set()
        self.cancelled_handles = set()
        self.cleanup_errors = []

    def _call(self, op, request_id, layer_id, handle=None, command_id=None, **kwargs):
        start=perf_counter()
        if self.closed:
            raise PNMError("CLOSED", "runtime closed")
        if request_id in self.cancelled and op != Op.FREE:
            raise PNMError('CANCELLED','request has been cancelled')
        if handle in self.poisoned and op in (Op.INIT,Op.APPEND):
            raise PNMError('POISONED_CACHE',self.poisoned[handle])
        payload = kwargs.pop("payload", {})
        if handle in self.handles:
            kwargs['generation']=self.handles[handle].generation
        c = Command(command_id or uuid4().hex, op, request_id, layer_id, handle, **kwargs)
        if self.metrics:
            packet=encode(c,payload)
            self.metrics.emit({'stage':'runtime_prepare','command_id':c.command_id,'ms':(perf_counter()-start)*1000,
                               'descriptor_bytes':len(packet),'state':'SUBMITTED'})
        try:
            result = self.backend.execute(c, payload)
        except TimeoutError as exc:
            if handle is not None:
                self.poisoned[handle]='TIMEOUT_UNKNOWN: device may have committed this token'
            raise PNMError('TIMEOUT_UNKNOWN',str(exc)) from exc
        if result.command_id != c.command_id:
            if handle is not None: self.poisoned[handle]='invalid or duplicate completion'
            raise PNMError('INVALID_COMPLETION', 'backend returned a different command ID')
        if result.status != "PASS":
            if handle is not None and result.error_code in ('DEVICE_LOST','STALE_HANDLE','TIMEOUT_UNKNOWN'):
                self.poisoned[handle]=str(result.error_code)
            raise PNMError(result.error_code, result.error_message)
        info = result.info
        if info is None or (info.request_id, info.layer_id, info.dtype, info.layout) != (
                c.request_id, c.layer_id, c.dtype, c.layout):
            raise PNMError('INVALID_COMPLETION', 'missing or mismatched cache metadata')
        if c.handle is not None and info.handle != c.handle:
            self.poisoned[handle]='completion handle mismatch'
            raise PNMError('INVALID_COMPLETION', 'backend returned a different handle')
        if handle is not None and info.generation != c.generation:
            self.poisoned[handle]='device generation mismatch'
            raise PNMError('INVALID_COMPLETION','device generation changed')
        if not 0 <= info.sequence_length <= info.capacity:
            raise PNMError('INVALID_COMPLETION', 'invalid sequence length')
        if op == Op.APPEND and (info.sequence_length != c.expected_length + 1 or result.output is None):
            self.poisoned[handle]='append completion inconsistent'
            raise PNMError('INVALID_COMPLETION', 'append output or committed length missing')
        if op == Op.ALLOC and (info.sequence_length != 0 or info.capacity != c.spec.capacity or info.initialized):
            raise PNMError('INVALID_COMPLETION', 'allocated cache does not match requested spec')
        if result.info and op != Op.FREE:
            self.handles[result.info.handle] = result.info
        if op == Op.FREE:
            self.handles.pop(handle, None)
            self.poisoned.pop(handle,None)
        return result

    def alloc_cache(self, request_id, layer_id, spec: CacheSpec, command_id=None):
        return self._call(Op.ALLOC, request_id, layer_id, spec=spec, command_id=command_id).info.handle

    def _owned(self, handle):
        if self.closed:
            raise PNMError("CLOSED", "runtime closed")
        if handle in self.cancelled_handles:
            raise PNMError('CANCELLED','request owning this handle was cancelled')
        if handle not in self.handles:
            raise PNMError("INVALID_HANDLE", str(handle))
        i = self.handles[handle]
        return i.request_id, i.layer_id

    def init_cache(self, handle, key, value, command_id=None):
        return self._call(Op.INIT, *self._owned(handle), handle=handle, command_id=command_id,
                          payload={"k": key, "v": value}).info

    def append_and_attend(self, handle, query, key, value, token_position, expected_length=None,
                          attention_mask=None, command_id=None):
        length = self.handles[handle].sequence_length if expected_length is None and handle in self.handles else expected_length
        return self._call(Op.APPEND, *self._owned(handle), handle=handle, command_id=command_id,
                          token_position=token_position, expected_length=length,
                          payload={"q": query, "k": key, "v": value, "mask": attention_mask}).output

    def get_cache_info(self, handle, command_id=None):
        return self._call(Op.INFO, *self._owned(handle), handle=handle, command_id=command_id).info

    def free_cache(self, handle, command_id=None):
        self._call(Op.FREE, *self._owned(handle), handle=handle, command_id=command_id)

    def cancel_request(self, request_id):
        self.cancelled.add(request_id)
        for handle,info in list(self.handles.items()):
            if info.request_id==request_id:
                self.free_cache(handle)
                self.cancelled_handles.add(handle)

    def describe(self):
        return self.backend.describe()

    def close(self):
        if not self.closed:
            try:
                for handle in list(self.handles):
                    try:
                        self.free_cache(handle)
                    except PNMError as exc:
                        self.cleanup_errors.append(str(exc))
            finally:
                self.backend.close()
                self.handles.clear()
                self.poisoned.clear()
                self.cancelled_handles.clear()
                self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
