"""Metadata-only Full Attention cache layer; no historical tensor is retained."""
import torch
from transformers.cache_utils import CacheLayerMixin
from pnm.protocol import PNMError


class RemoteCacheLayer(CacheLayerMixin):
    is_sliding = False
    is_croppable = False
    supports_early_init = False

    def __init__(self, length, capacity, handle):
        super().__init__()
        self.length, self.capacity = int(length), int(capacity)
        self.handle = handle
        self.pending = False
        self.poison_reason = None
        self.is_initialized = True
        self.batch_size = 1

    def _healthy(self):
        if self.poison_reason:
            raise PNMError('POISONED_CACHE', self.poison_reason)

    def lazy_initialization(self, key_states, value_states):
        raise PNMError('CACHE_BRIDGE', 'install remote layer after native prefill')

    def update(self, key_states, value_states, *args, **kwargs):
        self._healthy()
        if self.pending:
            raise PNMError('PENDING_APPEND', 'preceding token has no completion')
        if key_states.shape != (1,4,1,256) or value_states.shape != key_states.shape:
            raise PNMError('SHAPE', 'strict decode accepts one new KV token')
        if key_states.dtype != torch.bfloat16 or value_states.dtype != torch.bfloat16:
            raise PNMError('DTYPE_DEVICE', 'strict KV must be BF16')
        if self.length >= self.capacity:
            raise PNMError('CAPACITY', 'remote cache full')
        self.pending = True
        return key_states, value_states

    def commit(self, new_length):
        self._healthy()
        if not self.pending or new_length != self.length+1:
            raise PNMError('CACHE_BRIDGE', 'invalid remote completion length')
        self.length, self.pending = new_length, False

    def poison(self, reason):
        self.poison_reason = str(reason)

    def get_seq_length(self):
        self._healthy()
        return self.length

    def get_mask_sizes(self, query_length):
        self._healthy()
        return self.length+query_length, 0

    def get_max_length(self):
        return self.capacity

    def metadata(self):
        tensors=[v for v in vars(self).values() if isinstance(v,torch.Tensor)]
        return {'class':type(self).__name__,'sequence_length':self.length,'capacity':self.capacity,
                'local_kv_bytes':sum(t.numel()*t.element_size() for t in tensors),
                'keys_is_none':self.keys is None,'values_is_none':self.values is None,
                'pending':self.pending,'poisoned':self.poison_reason is not None}

    def _unsupported(self,*args,**kwargs):
        self.poison('unsupported cache mutation; discard this request')
        raise PNMError('UNSUPPORTED_CACHE_OPERATION','create a new request; remote crop/reorder/offload unsupported')

    reset = crop = reorder_cache = batch_repeat_interleave = batch_select_indices = offload = prefetch = _unsupported
