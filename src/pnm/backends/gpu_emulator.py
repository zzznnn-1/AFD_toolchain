from threading import RLock
from time import perf_counter
import torch
from pnm.protocol import Completion, Op, PNMError, CacheSpec
from .base import Backend
from .cuda_staging import HostStaging
from .gpu_memory import CacheManager
from .gpu_kernel import attend
from pnm.runtime.metrics import Metrics


class GPUEmulator(Backend):
    def __init__(self, source="cuda:0", device="cuda:1", max_tokens=8192, metrics=None,
                 spec=None, command_budget=100000, chunk_tokens=256, max_cache_bytes=512*2**20):
        self.source, self.device = torch.device(source), torch.device(device)
        self.max_tokens = max_tokens
        self.spec = spec or CacheSpec(max_tokens)
        self.spec.validate()
        self.metrics = metrics or Metrics()
        self.manager = CacheManager(self.device)
        self.transport = HostStaging(source, device, max_tokens, self.spec, self.metrics,chunk_tokens)
        self.max_cache_bytes = max_cache_bytes
        self.seen = set()
        self.command_budget = command_budget
        self.lock = RLock()
        self.closed = False
        self.online = True

    def execute(self, command, payload):
        submitted = perf_counter()
        with self.lock, torch.inference_mode():
            c, t = command, perf_counter()
            self.metrics.emit({'stage':'queue','command_id':c.command_id,'ms':(t-submitted)*1000,'state':'RUNNING'})
            try:
                if self.closed:
                    raise PNMError("CLOSED", "backend closed")
                if not self.online:
                    raise PNMError('DEVICE_LOST','emulator device is unavailable')
                if c.version != 1:
                    raise PNMError('PROTOCOL_VERSION',str(c.version))
                if c.op != Op.ALLOC and c.generation not in (0,self.manager.generation):
                    raise PNMError('STALE_HANDLE','device generation changed')
                if c.command_id in self.seen:
                    raise PNMError("DUPLICATE_COMMAND", c.command_id)
                if len(self.seen) >= self.command_budget:
                    raise PNMError("COMMAND_BUDGET", "start a new session")
                self.seen.add(c.command_id)
                if c.dtype != self.spec.dtype or c.layout != self.spec.layout:
                    raise PNMError("CONTRACT", "command dtype/layout differs from backend")
                result = self._execute(c, payload)
            except PNMError as exc:
                result = Completion(c.command_id, "FAIL", error_code=exc.code, error_message=str(exc))
            except Exception as exc:
                result = Completion(c.command_id, "FAIL", error_code="DEVICE_ERROR", error_message=repr(exc))
            self.metrics.emit({"stage": "command", "command_id": c.command_id, "op": c.op.value,
                               "request_id": c.request_id, "layer_id": c.layer_id,
                               "handle": result.info.handle if result.info else c.handle,
                               "token_position": c.token_position, "expected_length": c.expected_length,
                               "dtype": c.dtype, "layout": c.layout, "status": result.status,
                               "version":c.version,"generation":self.manager.generation,
                               "state":"COMPLETED" if result.status=='PASS' else "FAILED",
                               "error_code": result.error_code, "length": result.info.sequence_length if result.info else None,
                               "ms": (perf_counter() - t) * 1000})
            return result

    def _validate_tensor(self, x, heads, tokens):
        if not isinstance(x, torch.Tensor) or tuple(x.shape) != (1, heads, tokens, self.spec.head_dim):
            raise PNMError("SHAPE", f"expected [1,{heads},{tokens},{self.spec.head_dim}]")
        if x.device != self.source or x.dtype != torch.bfloat16:
            raise PNMError("DTYPE_DEVICE", "inputs must be BF16 on source GPU")

    def _execute(self, c, p):
        if c.op == Op.ALLOC:
            if c.spec is None:
                raise PNMError("INVALID_SPEC", "missing spec")
            c.spec.validate()
            if c.spec.capacity > self.max_tokens or any(getattr(c.spec, n) != getattr(self.spec, n) for n in
                    ["batch", "query_heads", "kv_heads", "head_dim", "dtype", "layout"]):
                raise PNMError("INVALID_SPEC", "spec exceeds transport contract")
            used=sum(2*e.key.numel()*e.key.element_size() for e in self.manager.entries.values())
            required=2*c.spec.batch*c.spec.kv_heads*c.spec.capacity*c.spec.head_dim*2
            if used+required>self.max_cache_bytes:
                raise PNMError('RESOURCE_EXHAUSTED',f'KV budget {self.max_cache_bytes} exceeded')
            h = self.manager.allocate(c.request_id, c.layer_id, c.spec)
            return Completion(c.command_id, "PASS", self.manager.info(h))
        e = self.manager.get(c.handle, c.request_id, c.layer_id)
        if c.op == Op.INFO:
            return Completion(c.command_id, "PASS", self.manager.info(c.handle))
        if c.op == Op.FREE:
            info = self.manager.info(c.handle)
            self.manager.free(c.handle)
            return Completion(c.command_id, "PASS", info)
        if c.op == Op.INIT:
            if e.initialized:
                raise PNMError("ALREADY_INITIALIZED", "prefill may be initialized once")
            n = p["k"].shape[2]
            if n < 1 or n > e.spec.capacity:
                raise PNMError("CAPACITY", str(n))
        elif c.op == Op.APPEND:
            if not e.initialized:
                raise PNMError("NOT_INITIALIZED", "init prefill first")
            if c.expected_length != e.length or c.token_position != e.length:
                raise PNMError("LENGTH_POSITION", f"device={e.length}, expected={c.expected_length}, pos={c.token_position}")
            if e.length + 1 > e.spec.capacity:
                raise PNMError("CAPACITY", "cache full")
            n = 1
            self._validate_tensor(p["q"], e.spec.query_heads, 1)
            mask = p.get("mask")
            if mask is not None:
                if tuple(mask.shape) != (1, 1, 1, e.length + 1) or mask.device != self.source:
                    raise PNMError("MASK", "decode mask must be [1,1,1,KV_length] on source")
                if mask.dtype not in (torch.bool, torch.bfloat16, torch.float32):
                    raise PNMError("MASK", "bool or floating additive mask required")
        else:
            raise PNMError("UNKNOWN_OP", str(c.op))
        self._validate_tensor(p["k"], e.spec.kv_heads, n)
        self._validate_tensor(p["v"], e.spec.kv_heads, n)
        old = e.length
        for offset in range(0,n,self.transport.chunk_tokens):
            end_offset=min(offset+self.transport.chunk_tokens,n)
            k=self.transport.transfer(p['k'][:,:,offset:end_offset], 'k', c.command_id)
            v=self.transport.transfer(p['v'][:,:,offset:end_offset], 'v', c.command_id)
            with torch.cuda.device(self.device):
                start_event,end_event=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start_event.record()
                e.key[:,:,old+offset:old+end_offset].copy_(k)
                e.value[:,:,old+offset:old+end_offset].copy_(v)
                end_event.record()
            end_event.synchronize()  # Staging scratch is safe to reuse for the next chunk.
            self.metrics.emit({'stage':'kv_init' if c.op==Op.INIT else 'kv_append',
                               'command_id':c.command_id,'ms':start_event.elapsed_time(end_event),
                               'tokens':end_offset-offset,'position':old+offset})
        output = None
        if c.op == Op.APPEND:
            q = self.transport.transfer(p["q"], "q", c.command_id)
            if mask is not None:
                if mask.dtype == torch.bool:
                    mask = torch.zeros_like(mask, dtype=torch.float32).masked_fill(~mask, -torch.inf)
                else:
                    mask = mask.float()
                mask = self.transport.transfer(mask, "mask", c.command_id)
            with torch.cuda.device(self.device):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                o = attend(q, e.key[:, :, :old+1], e.value[:, :, :old+1], query_start=old, mask=mask)
                end.record()
            end.synchronize()
            self.metrics.emit({"stage": "attention", "command_id": c.command_id, "device": str(self.device),
                               "length": old+1, "ms": start.elapsed_time(end)})
            # Clone gives callers ownership beyond the next command/slot reuse.
            output = self.transport.transfer(o, "o", c.command_id).clone()
            torch.cuda.current_stream(self.source).synchronize()
        else:
            torch.cuda.current_stream(self.device).synchronize()
        # Commit only after successful device work AND return transfer. Failed append
        # leaves old length visible; the uncommitted slot is overwritten on retry.
        e.length, e.initialized = old + n, True
        return Completion(c.command_id, "PASS", self.manager.info(c.handle), output)

    def describe(self):
        return {'backend':'gpu_emulator','protocol_version':1,'generation':self.manager.generation,
                'online':self.online,'kv_caches':len(self.manager.entries),
                'kv_allocated_bytes':sum(2*e.key.numel()*e.key.element_size() for e in self.manager.entries.values()),
                **self.transport.describe()}

    def reset(self):
        with self.lock:
            if self.closed: raise PNMError('CLOSED','cannot reset a closed backend')
            torch.cuda.synchronize(self.source)
            torch.cuda.synchronize(self.device)
            self.manager.reset()
            self.online=True

    def lose_device(self):
        self.reset()
        self.online=False

    def close(self):
        with self.lock:
            if not self.closed:
                try:
                    self.transport.close()
                finally:
                    self.manager.close()
                    self.seen.clear()
                    self.closed = True
                    self.online = False
