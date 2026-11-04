from uuid import uuid4
from pnm.protocol import CacheSpec, Command, Op, PNMError


class Runtime:
    """Backend-neutral synchronous v1 API. Only immutable metadata crosses ownership boundary.

    Duplicate command IDs are rejected, never replayed. IDs are scoped to a Runtime
    session, whose bounded command budget is specified by the backend.
    """
    def __init__(self, backend):
        self.backend = backend
        self.handles = {}
        self.closed = False

    def _call(self, op, request_id, layer_id, handle=None, command_id=None, **kwargs):
        if self.closed:
            raise PNMError("CLOSED", "runtime closed")
        payload = kwargs.pop("payload", {})
        c = Command(command_id or uuid4().hex, op, request_id, layer_id, handle, **kwargs)
        result = self.backend.execute(c, payload)
        if result.status != "PASS":
            raise PNMError(result.error_code, result.error_message)
        if result.info and op != Op.FREE:
            self.handles[result.info.handle] = result.info
        if op == Op.FREE:
            self.handles.pop(handle, None)
        return result

    def alloc_cache(self, request_id, layer_id, spec: CacheSpec, command_id=None):
        return self._call(Op.ALLOC, request_id, layer_id, spec=spec, command_id=command_id).info.handle

    def _owned(self, handle):
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

    def close(self):
        if not self.closed:
            try:
                for handle in list(self.handles):
                    self.free_cache(handle)
            finally:
                self.backend.close()
                self.handles.clear()
                self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
