from dataclasses import dataclass
from uuid import uuid4
import torch
from pnm.protocol import CacheInfo, CacheSpec, PNMError


@dataclass
class Entry:
    request_id: str
    layer_id: int
    spec: CacheSpec
    key: torch.Tensor
    value: torch.Tensor
    length: int = 0
    initialized: bool = False


class CacheManager:
    def __init__(self, device):
        self.device = torch.device(device)
        self.entries = {}

    def allocate(self, request_id, layer_id, spec):
        spec.validate()
        shape = (spec.batch, spec.kv_heads, spec.capacity, spec.head_dim)
        h = uuid4().hex
        self.entries[h] = Entry(request_id, layer_id, spec,
                                torch.empty(shape, dtype=torch.bfloat16, device=self.device),
                                torch.empty(shape, dtype=torch.bfloat16, device=self.device))
        return h

    def get(self, handle, request_id, layer_id):
        if handle not in self.entries:
            raise PNMError("INVALID_HANDLE", str(handle))
        e = self.entries[handle]
        if (e.request_id, e.layer_id) != (request_id, layer_id):
            raise PNMError("OWNERSHIP", "request/layer does not own handle")
        return e

    def info(self, handle):
        e = self.entries[handle]
        return CacheInfo(handle, e.request_id, e.layer_id, e.length, e.spec.capacity,
                         e.spec.dtype, e.spec.layout, 2 * e.key.numel() * e.key.element_size(), e.initialized)

    def free(self, handle):
        del self.entries[handle]

    def close(self):
        self.entries.clear()
